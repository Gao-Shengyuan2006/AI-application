import base64
import json
import requests

def image_to_base64(image_path):
    """将本地图片转换为base64编码字符串"""
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        return {"error": f"图片读取失败: {str(e)}"}

def build_compatible_request(image_path, object_box_desc, api_key):
    """构建OpenAI兼容模式的千问多模态请求（官网推荐方式）"""
    # 1. 图片转base64
    base64_image = image_to_base64(image_path)
    if isinstance(base64_image, dict) and "error" in base64_image:
        return base64_image
    
    # 2. 提示词（和GPT-4V格式一致）
    prompt = f"""
    识别图片中被框选的物体类别，框选区域描述：{object_box_desc}。
    要求：仅返回物体核心类别名称（如手机、杯子），不要多余文字，无法识别返回"未知物体"。
    """
    
    # 3. 完全复用OpenAI的请求格式（兼容模式核心）
    request_data = {
        "model": "qwen-vl-plus",  # 千问多模态模型名
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    
    return {
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"  # 兼容模式用Bearer Token（和OpenAI一致）
        },
        "data": json.dumps(request_data, ensure_ascii=False)
    }

def call_qwen_compatible_api(base_url, request_config):
    """调用千问兼容模式API（官网推荐的Method 1）"""
    # 拼接完整接口地址（兼容模式固定为/chat/completions）
    api_url = f"{base_url}/chat/completions"
    
    try:
        response = requests.post(
            api_url,
            headers=request_config["headers"],
            data=request_config["data"],
            timeout=90
        )
        response.raise_for_status()
        result = response.json()
        
        # 解析结果（和OpenAI格式完全一致）
        if "choices" in result and len(result["choices"]) > 0:
            category = result["choices"][0]["message"]["content"].strip()
            return {"success": True, "category": category, "raw_result": result}
        else:
            return {"success": False, "error": "返回格式异常", "raw_result": result}
    
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API调用失败: {str(e)}"}

# ------------------- 官网推荐的使用方式 -------------------
if __name__ == "__main__":
    # 配置官网指定的参数（替换为你的信息）
    CONFIG = {
        "image_path": "test.jpg",  # 你的图片路径
        "object_box_desc": "图片中红色框选的物体",  # 框选描述
        "api_key": "your-dashscope-api-key",  # 千问API Key（仅需api_key，无需api_secret）
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
    # 1. 构建请求（完全复用OpenAI逻辑）
    request_config = build_compatible_request(
        CONFIG["image_path"],
        CONFIG["object_box_desc"],
        CONFIG["api_key"]
    )
    if "error" in request_config:
        print(f"请求构建失败：{request_config['error']}")
    else:
        # 2. 调用API
        result = call_qwen_compatible_api(CONFIG["base_url"], request_config)
        
        # 3. 输出结果
        if result["success"]:
            print(f"识别结果：{result['category']}")
        else:
            print(f"失败原因：{result['error']}")
            print("原始返回：", json.dumps(result['raw_result'], indent=2, ensure_ascii=False))