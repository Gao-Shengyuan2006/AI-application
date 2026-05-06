import cv2
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator


def smart_annotation(annotations, video_path):
    """
    Args:
        annotations: 字典，key是帧号(int)，value是tuple (center_x, center_y, width, height)
                    例如: {0: (100, 200, 50, 80), 50: (150, 220, 55, 85), ...}
        video_path: 视频文件路径
    
    Returns:
        字典，key是帧号(int)，value是tuple (center_x, center_y, width, height)
        包含视频中所有帧的预测位置
    """
    if len(annotations) < 4:
        raise ValueError("至少需要4个标注才能进行预测")
    
    # 打开视频获取信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 转换标注格式并排序
    positions = []
    for frame_idx, bbox in annotations.items():
        if len(bbox) != 4:
            raise ValueError(f"标注格式错误，应为(center_x, center_y, width, height)，得到: {bbox}")
        center_x, center_y, bbox_width, bbox_height = bbox
        positions.append({
            'frame': int(frame_idx),
            'center_x': float(center_x),
            'center_y': float(center_y),
            'width': float(bbox_width),
            'height': float(bbox_height)
        })
    
    positions_sorted = sorted(positions, key=lambda p: p['frame'])
    frames = np.array([pos['frame'] for pos in positions_sorted])
    x_coords = np.array([pos['center_x'] for pos in positions_sorted])
    y_coords = np.array([pos['center_y'] for pos in positions_sorted])
    widths = np.array([pos['width'] for pos in positions_sorted])
    heights = np.array([pos['height'] for pos in positions_sorted])
    
    # 检查帧号范围
    if frames.min() < 0 or frames.max() >= total_frames:
        raise ValueError(f"标注帧号超出范围 [0, {total_frames-1}]")
    
    # 计算变化率（用于自适应混合插值）
    def compute_change_ratios():
        change_ratios = []
        for i in range(len(frames) - 1):
            frame_diff = frames[i+1] - frames[i]
            if frame_diff == 0:
                change_ratios.append(0)
                continue
            
            pos_diff = np.sqrt((x_coords[i+1] - x_coords[i])**2 + 
                              (y_coords[i+1] - y_coords[i])**2)
            size_diff = np.sqrt((widths[i+1] - widths[i])**2 + 
                               (heights[i+1] - heights[i])**2)
            
            pos_change_rate = pos_diff / frame_diff if frame_diff > 0 else 0
            size_change_rate = size_diff / frame_diff if frame_diff > 0 else 0
            total_change_rate = pos_change_rate + size_change_rate * 0.1
            change_ratios.append(total_change_rate)
        
        return np.array(change_ratios)
    
    change_ratios = compute_change_ratios()
    use_adaptive = False
    
    if len(change_ratios) > 0:
        avg_change = np.mean(change_ratios)
        max_change = np.max(change_ratios)
        use_adaptive = max_change > avg_change * 3.0
        
    # 创建插值函数（默认使用PCHIP，变化大时使用自适应混合）
    if use_adaptive:
        # 自适应混合插值：变化大的区间混合线性插值，变化小的区间使用PCHIP
        x_func_pchip = PchipInterpolator(frames, x_coords)
        y_func_pchip = PchipInterpolator(frames, y_coords)
        width_func_pchip = PchipInterpolator(frames, widths)
        height_func_pchip = PchipInterpolator(frames, heights)
        
        x_func_linear = interp1d(frames, x_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
        y_func_linear = interp1d(frames, y_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
        width_func_linear = interp1d(frames, widths, kind='linear', bounds_error=False, fill_value='extrapolate')
        height_func_linear = interp1d(frames, heights, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        threshold = avg_change * 2.0
        
        def create_mixed_func(func_pchip, func_linear):
            def mixed_func(frame_idx):
                if frame_idx <= frames[0]:
                    return float(func_pchip(frame_idx))
                if frame_idx >= frames[-1]:
                    return float(func_pchip(frame_idx))
                
                interval_idx = np.searchsorted(frames, frame_idx) - 1
                interval_idx = max(0, min(interval_idx, len(change_ratios) - 1))
                
                change_rate = change_ratios[interval_idx]
                if change_rate > threshold:
                    linear_weight = min(1.0, (change_rate - threshold) / threshold)
                    pchip_weight = 1.0 - linear_weight
                else:
                    pchip_weight = 1.0
                    linear_weight = 0.0
                
                pred_pchip = float(func_pchip(frame_idx))
                pred_linear = float(func_linear(frame_idx))
                return pchip_weight * pred_pchip + linear_weight * pred_linear
            
            return mixed_func
        
        x_func = create_mixed_func(x_func_pchip, x_func_linear)
        y_func = create_mixed_func(y_func_pchip, y_func_linear)
        width_func = create_mixed_func(width_func_pchip, width_func_linear)
        height_func = create_mixed_func(height_func_pchip, height_func_linear)
    else:
        # 标准PCHIP插值
        x_func = PchipInterpolator(frames, x_coords)
        y_func = PchipInterpolator(frames, y_coords)
        width_func = PchipInterpolator(frames, widths)
        height_func = PchipInterpolator(frames, heights)
    
    # 预测所有帧的位置
    result = {}
    for frame_idx in range(total_frames):
        # 如果该帧已有标注，使用标注值
        if frame_idx in annotations:
            result[frame_idx] = annotations[frame_idx]
        else:
            # 预测位置
            pred_x = max(0, min(width, float(x_func(frame_idx))))
            pred_y = max(0, min(height, float(y_func(frame_idx))))
            pred_w = max(10, min(width, float(width_func(frame_idx))))
            pred_h = max(10, min(height, float(height_func(frame_idx))))
            
            result[frame_idx] = (pred_x, pred_y, pred_w, pred_h)
    
    return result

'''
# 使用示例

    # 示例：手动标注了几帧
    manual_annotations = {
        0: (100, 200, 50, 80),    # 第0帧: (center_x, center_y, width, height)
        50: (150, 220, 55, 85),   # 第50帧
        100: (200, 240, 60, 90),  # 第100帧
        150: (250, 260, 65, 95),  # 第150帧
    }
    
    video_path = "/Users/lubitong/Desktop/ob/79_1762357671.mp4"
    
    # 调用智能标注函数
    all_annotations = smart_annotation(
        annotations=manual_annotations,
        video_path=video_path
    )
    
    # 查看结果
    print(f"\n前5帧的预测结果:")
    for i in range(5):
        if i in all_annotations:
            print(f"帧 {i}: {all_annotations[i]}")

'''
