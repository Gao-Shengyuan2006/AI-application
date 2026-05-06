import logging
from fastapi import FastAPI, Request, status, HTTPException, UploadFile, File, BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel
from typing import Dict, List
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from mysql.connector import Error
import shutil
import uuid
import cv2
import pandas as pd
import os
from cv_models import KCF, SAM
import numpy as np
import base64
import aiofiles
from ultralytics import YOLO
import yaml
import random
import mysql.connector
from shutil import copy2, rmtree
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    username: str
    password: str
        
class UserRequest(BaseModel):
    userID: str

class ProjectRequest(BaseModel):
    projectID: str

class CreateProjectRequest(BaseModel):
    userID: str
    project_name: str
    project_type: str = "YOLO object detection"

class VideoRequest(BaseModel):
    projectID: str
    videoID: str

class AnnotationRequest(BaseModel):
    projectID: str
    videoID: str
    frame_num: int
    coordinates: list

class UserLogin():
    def __init__(self, username, password, status=True):
        self.username = username
        self.password = password  # In production, hash this!
        self.status = status      # True if active
        self.login_attempts = 0

    def get_password_hash(self):
        ### db ###
        # Find hashed password from database
        cursor=self.db.cursor()
        query = "SELECT password FROM user WHERE username = %s"
        cursor.execute(query, (username,))
        row = cursor.fetchone()
        cursor.close()
        
    def verify_password(self, username: str, password: str) -> bool:
        hash_from_db = self.get_password_hash(username)
        if hash_from_db is None:
            return False
        return pwd_context.verify(password, hash_from_db)
    
    def get_userID(self):
        ### db ###
        # Find userID given self.username
        cursor=self.db.cursor()
        query="SELECT user_id FROM user WHERE username=%s"
        cursor.execute(query,(use_id,))
        row=cursor.fetchone()
        cursor.close()
        return userID

    def login(self, max_attempts=3):
        if not self.status:
            return False, "Account locked."
        
        # Hash the password input
        salt, pwd_hash = self._hash_password(self.password)
        stored_hash = self.get_password_hash()
        is_correct = self._verify_password(stored_hash, salt, self.password)

        if is_correct:
            self.login_attempts = 0
            return True, "Login successful."

        else:
            self.login_attempts += 1
            if self.login_attempts >= max_attempts:
                self.status = False
                return "Login attempts exceeded. Account locked."
            return False, "Invalid password."
        
    @staticmethod
    def _hash_password(self, password, salt=None):
        # Generate a random salt if not provided
        if salt is None:
            salt = os.urandom(16)
        # Use PBKDF2-HMAC-SHA256 as the hashing algorithm
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000)
        return salt, pwd_hash

    @staticmethod
    def _verify_password(self, stored_hash, stored_salt, provided_password):
        # Hash the provided password using the stored salt
        _, pwd_hash = self._hash_password(provided_password, stored_salt)
        # Use hmac.compare_digest to avoid timing attacks
        return hmac.compare_digest(pwd_hash, stored_hash)

class User():
    def __init__(self, userID: str):
        self.userID = userID
        self.username = self.get_username()
    
    def get_username(self):
        ### db ###
        DB_CONFIG ={
            'host': 'localhost',
            'database': 'your_database',
            'user': 'your_username',
            'password': 'your_password'
        }
        connection= None
        username=None
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            if connection.is_connected():
                cursor = connection.cursor()
                query= "SELECT username FROM user WHERE user_id =%s"
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                if result:
                    username=result[0]
        except Error as e:
            print(f"Database Error: {e}")
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()  
        return username

    def get_owned_projects(self):
        ### db ###
        owned_projects = []  # list of project IDs
        self.owned_projects = owned_projects
        return owned_projects
    
    def get_shared_projects(self):
        ### db ###
        shared_projects = []  # list of project IDs
        self.shared_projects = shared_projects
        return shared_projects

class Project():
    def __init__(self, projectID: str, initialize=False):
        if initialize:
            self.initialize()
        else:
            self.projectID = projectID
            self.project_name = self.get_project_name()
            self.project_type = self.get_project_type()
            self.videos = self.get_videos()
            self.video_count = self.get_video_count()
            self.owner = self.get_owner()
            self.shared_users = self.get_shared_users()
            self.project_status = self.get_project_status()

    def initialize(self, project_name, project_type, owner):
        self.project_name = project_name
        if self.project_name.strip() == '':
            self.project_name = "Untitled"
            i = 1
            while self.project_name_exists():
                self.project_name = f"{self.project_name} {i}"
                i += 1
        self.save_project_name()
        self.project_type = project_type
        self.owner = owner
        self.save_owner()
        self.initialize_classes()
        self.project_status = "Not started" # can be "Awaiting Labeling", "Labeling in progress", "Data is ready", "Training in progress", "Trained"
        self.save_project_status()
        
        # Create project directory
        self.get_project_path()

        # Get project ID
        projectID = self.get_projectID()

        return projectID

    def get_projectID(self):
        ### db ###
        return projectID
    
    def project_name_exists(self):
        if self.project_name == self.get_project_name():
            return True
        return False
    
    def get_project_name(self):
        ### db ###
        query = "SELECT project_name FROM project WHERE project_name = %s"
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (self.project_name,))
                row = cur.fetchone()
        return project_name
    
    def get_project_type(self):
        ### db ###
        project_type = "YOLO object detection"
        query = "SELECT project_type FROM project WHERE project_id = %s"
        with self._get_connection() as conn:
            with conn.cursor()as conn:
                cur.execute(query,(self.project_type,))
                row=cur.fetchone
        return project_type
    
    def get_videos(self):
        
        conn = None
        cursor= None
        conn=mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        cursor=conn.cursor()
        if self.project_id is not None:
            query = "SELECT video_id FROM video WHERE project_id = %s ORDER BY video_id ASC"
            cursor.execute(query, (self.project_id,))
        else:
            query = "SELECT video_id FROM video ORDER BY video_id ASC"
            cursor.execute(query)
        rows = cursor.fetchall()
        return [video1, video2, ...]
    
    def get_video_count(self):
        video_count = len(self.videos)
        return video_count
    
    def get_owner(self):
        # return owner userID
        ### db ###
        query = "SELECT project_owner_id FROM project WHERE project_id = %s"
        with conn.cursor()as cur:
            cur.execute(query,(self.project_owner_id))
        row=cur.fetchone()
        ownerID = "### owner ID ###"
        return ownerID
    
    def get_shared_users(self):
        # return shared users' userID (excluding owner)
        ### db ###
        conn= None
        cursor= None
        shared_users = ["### user1 ID ###", "### user2 ID ###", ...]
        conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        cursor = conn.cursor()
        query = """
                SELECT DISTINCT user_id
                FROM project_shared_users
                WHERE project_id = %s AND user_id <> %s
            """
        cursor.execute(query, (self.project_id, self.owner_user_id))
        rows = cursor.fetchall()
        shared_users = ["### user1 ID ###", "### user2 ID ###", ...]
        return shared_users
    
    def get_classes(self):
        ### db ###
        classes = {"class 1": "blue"}
        conn= None
        cursor= None
        conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        cursor = conn.cursor()
        query="SELECT class_name, color FROM project_classes WHERE class_id=%s"
        cursor.execute(query,(self.class_name,self.color))
        rows=cursor.fetchall()
        return classes
    
    def get_project_status(self):
        ### db ###
        ### Project_status can be "Not started", "Awaiting Labeling", "Labeling in progress", "Data is ready", "Training in progress", "Training completed" ###
        query = "SELECT project_status FROM project WHERE project_id = %s"
        cursor.execute(query,(self.project_status))
        row=cursor.fetchone()
        return project_status
        
    def get_project_path(self):
        with db_connection.cursor() as cursor:
            cursor.execute("""
                SELECT model_path, dataset_path
                FROM your_table_name
                WHERE projectID = %s
                LIMIT 1
            """, (self.projectID,))
            row = cursor.fetchone()
            project_path = f"./{self.projectID}/"

            if not os.path.exists(project_path):
                os.makedirs(project_path)
            return project_path
    
    def change_project_name(self, new_name: str):
        self.project_name = new_name
        self.save_project_name()
        return True
    
    def check_class_exists(self, class_name: str):
        return class_name in self.classes.values()
    
    def initialize_classes(self):
        self.classes = {}
    
    def get_new_class_ID(self):
        # get new class ID
        classID = 80
        while classID in self.classes:
            classID += 1
        return classID
    
    def add_class(self, class_name: str, colour: str):
        self.classes = self.get_classes()
        self.classes[class_name] = colour
        ### db ###
        self_classes=self.get_glasses()
        self_classes[class_name]=colour
        sql = """
        INSERT INTO `class` (`class_name`, `colour`)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE `colour` = VALUES(`colour`)
    """
        with db_connection.cursor() as cursor:
            cursor.execute(sql,(class_name,colour))
            db.connection.commit
        # Only add one class, do not replace all class info
        return True
    
    def modify_class(self, original_class_name: str, new_class_name: str):
        self.classes = self.get_classes()
        self.classes[new_class_name] = self.classes.pop(original_class_name)
                ### db ###
        self.classes[new_class_name] = colour
        del self.classes[original_class_name]
        with db_connection.cursor() as cursor:
            cursor.execute(sql, (new_class_name, original_class_name))
        db_connection.commit()
                # Only modify one class, do not replace all class info
        return True
    
    def delete_class(self, class_name: str):
        self.classes.pop(class_name)
                ### db ###
        self.classes = self.get_classes()
        self.classes.pop(class_name, None)
        sql = "DELETE FROM `class` WHERE `class_name` = %s"
        with db_connection.cursor() as cursor:
            cursor.execute(sql, (class_name,))
        db_connection.commit()
        return True

    def create_dataset(self):
        label_dir = f"{self.get_project_path()}/datasets/labels/"
        image_dir = f"{self.get_project_path()}/datasets/images/"

        # Create the class_ID, class_name dictionary
        self.classes = self.get_classes()
        class_list = [key for key in self.classes]
        class_list.sort()
        print("classes sorted in alphabetical order")
        class_id_dict = {}
        id = 0
        for _class in class_list:
            class_id_dict[_class] = id
            id += 1
        print("Class id defined as: "+class_id_dict)
        self.save_class_ids(class_id_dict)

        for videoID in self.videos:
            video = Video(self.projectID, videoID)

            # Write labels
            bbox_data = video.get_bbox_data()
            for data in bbox_data:
                # bbox_data = [[frame_num, class_name, coordinates]]
                frame_num = data[0]
                class_name = data[1]
                coordinates = data[2]

                if not isinstance(coordinates, str):
                    raise ValueError(f"Video {video.videoID} has unannotated frames. Please complete annotation before creating dataset.")
                
                ### db change path if necessary ###
                filename = f"{label_dir}{video.videoID}_frame_{frame_num}.txt"

                with open(filename, 'a') as file:
                    file.write(f"{class_id_dict[class_name]} {coordinates}\n")

            # Write images
            cap = cv2.VideoCapture(video.get_video_path())
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image_path = f"{image_dir}{video.videoID}_frame_"+str(frame_idx)+".png"
                cv2.imwrite(image_path, frame)
                frame_idx += 1

        self.project_status = "Dataset ready"
        self.save_project_status()

        return True
    
    def get_auto_annotation_progress(self):
        finished_frames = 0
        total_frames = 0
        for videoID in self.videos:
            video = Video(projectID=self.projectID, videoID=videoID)
            annotation_status, last_annotated_frame = video.get_annotation_status()
            frame_count = video.get_frame_count()
            if annotation_status == "auto annotation in progress":
                total_frames = last_annotated_frame + 1
            elif annotation_status == "completed":
                finished_frames += frame_count
            total_frames += frame_count
        overall_progress = finished_frames / total_frames if total_frames > 0 else 0

        return overall_progress
    
    def get_uploaded_videos(self):
        videos_info = []
        for videoID in self.videos:
            video = Video(projectID=self.projectID, videoID=videoID)
            video_info = video.get_video_info()
            videos_info.append(video_info)
        return videos_info
    
    def get_class_ids(self):
        ### db ###
        sql = "SELECT `class_id` FROM `class` ORDER BY `class_id` ASC"
        with db_connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            class_ids = [row[0] for row in rows]
        db_connection.commit()
        return class_ids
    
    # Save project status to database
    def save_project_status(self):
        ### db ###
        sql = "SELECT `class_id` FROM `class` ORDER BY `class_id` ASC"
        with db_connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            class_ids = [row[0] for row in rows]
        db.connection.commit()
        success = True  # data saved successfully
        return success
    
    # Save video IDs to database
    def save_videos(self):
        ### db ###
        insert_sql = """
                    INSERT INTO `video` (project_id, video_path, video_name, annotation_status, last_annotated_frame, total_frames)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
        for v in getattr(self, 'videos_to_save', []):
            cur=self.conn.cursor()
            cur.execute(insert_sql, (
                v.get('project_id'),
                v.get('video_path'),
                v.get('video_name'),
                v.get('annotation_status'),
                v.get('last_annotated_frame', 0),
            ))
        self.conn.commit()
        success = True  # data saved successfully
        return success
    
    # Save video count to database
    def save_video_count(self):
        ### db ###
        success = True  # data saved successfully
        return success
    
    # Save project name to database
    def save_project_name(self):
        ### db ###
        connection=None
        success=False
        with connection.cursor() as cursor:
            sql="INSERT INTO my_project_name('project_name')VALUES(%s)"
            cursor.execute(sql,('project_name,'))
        connection.commit()
        success = True  # data saved successfully
        return success
    
    # Save project type to database
    def save_project_type(self):
        ### db ###
        connection=None
        success=False
        with connection.cursor() as cursor:
            sql="INSERT INTO {self.project_type} (project_type) VALUES(%s)"
            cursor.execute(sql,('project_type,'))
        connection.commit()
        success = True  # data saved successfully
        return success
    
    # Save owner ID to database
    def save_owner(self):
        ### db ###
        username=None
        password=None
        sql = """
        INSERT INTO `user` (`username`, `password`)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE `password` = VALUES(`password`)
    """
        with db_connection.cursor() as cursor:
            cursor.execute(sql, (username, password))
        db_connection.commit()
        success = True  # data saved successfully
        return success
    
    def save_training_progress(self, training_progress: int):
        ### db ###
        sql = """
        INSERT INTO `training_progress_table` (`training_progress`)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE `training_progress` = VALUES(`training_progress`)
        """
        with db_connection.cursor() as cursor:
            cursor.execute(sql, (training_progress))
        db_connection.commit()
        success = True if data_saved_successfully else False
        return success
    
    def save_auto_annotation_progress(self, auto_annotation_progress: int):
        ### db ###
        sql = """
        INSERT INTO `auto_annotation_progress_table` (`progress_id`, `auto_annotation_progress`)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE `auto_annotation_progress` = VALUES(`auto_annotation_progress`)
    """
        with db_connection.cursor() as cursor:
            cursor.execute(sql, (progress_id, auto_annotation_progress))
        db_connection.commit()
        success = True  # data saved successfully
        return success
    
    @staticmethod
    def copy_files(image_list, img_dest, lbl_dest, labels_dir):
        for img_path in image_list:
            base_name = img_path.stem
            label_path = labels_dir / (base_name + ".txt")
            copy2(img_path, img_dest)
            if label_path.exists():
                copy2(label_path, lbl_dest)
    
    def train(self):
        self.project_status = "Training in progress"
        self.save_project_status()

        dataset_dir = Path(self.get_project_path()) / "dataset"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        train_img_dir = dataset_dir / "train" / "images"
        train_lbl_dir = dataset_dir / "train" / "labels"
        val_img_dir = dataset_dir / "val" / "images"
        val_lbl_dir = dataset_dir / "val" / "labels"
        output_dir = Path(self.get_project_path()) / "output"
        output_dir.mkdir(exist_ok=True)

        # Clean and create split folders
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            if d.exists():
                rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # Gather all images and shuffle
        all_images = list(images_dir.glob("*.jpg"))
        random.shuffle(all_images)
        split_idx = int(0.8 * len(all_images))

        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        self.copy_files(train_images, train_img_dir, train_lbl_dir, labels_dir)
        self.copy_files(val_images, val_img_dir, val_lbl_dir, labels_dir)

        # Define class list
        classes = [key for key in self.get_class_ids()]

        # Create data.yaml for dataset with full COCO classes
        data_yaml = {
            "train": str(train_img_dir.resolve()),
            "val": str(val_img_dir.resolve()),
            "nc": len(classes),
            "names": classes
        }

        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        # Load pretrained YOLOv11m model
        model = YOLO("yolo11m.pt")

        # Total epochs
        total_epochs = 5 # CHANGE THIS IF YOU NEED TO SEE FASTER TRAINING

        # Train epoch-by-epoch to track progress
        for epoch in range(total_epochs):
            model.train(
                data=str(yaml_path),
                epochs=1,
                imgsz=640,
                batch=4,
                #device="cpu",
                save=True,
                exist_ok=True
            )
            progress = int((epoch + 1) / total_epochs * 100) # YOUR TRAINING VARIABLE
            print(f"Training progress: {progress:.2f}%")
            self.save_training_progress(progress)

        # Copy best.pt to output folder
        best_weights_src = Path("runs/detect/train/weights/best.pt")
        best_weights_dst = output_dir / "best.pt"

        if best_weights_src.exists():
            copy2(best_weights_src, best_weights_dst)
            print(f"Best model weights saved to: {best_weights_dst}")
        else:
            print("Warning: best.pt not found")

        print("Training complete.")

        self.project_status = "Training completed"
        self.save_project_status()

    def get_model_path(self):
        model_path = Path(self.get_project_path()) / "output" / "best.pt"
        return model_path.resolve()

    def get_model_performance(self):
        ### YOLO ###
        return performance
    
class Video(Project):
    def __init__(self, projectID: str, videoID=None, initialize=False):
        super().__init__(projectID)
        self.videoID = videoID
        if not initialize:
            self.video_path = self.get_video_path()
        if initialize:
            self.initialize()
        self.cap = cv2.VideoCapture(self.video_path)

    def initialize(self, ext):
        self.annotation_status, self.last_annotated_frame = "yet to start", None
        self.save_annotation_status()
        self.save_last_annotated_frame()
        self.initialize_bbox_data()
        self.videoID = self.get_videoID()
        self.initialize_video_path(ext)
        self.save_video_path()
        self.video_count += 1
        self.videoID = self.get_videoID()
        self.videos.append(self.videoID)
        self.save_video_count()
        self.save_videos()

        return self.videoID, self.video_path
    
    def get_video_info(self):
        info = {
            "name": self.get_video_name,
            "file": self.get_video,
            "path": self.get_video_path
        }
        return info
    
    async def get_video(self, file_path: str):
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found.")
        ext = os.path.splitext(file_path)[-1].lower()
        media_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska'
        }
        # Read the file contents for response
        async with aiofiles.open(file_path, "rb") as f:
            contents = await f.read()
        return Response(contents, media_type=media_types.get(ext, "application/octet-stream"),
                        headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"})
    
    def get_videoID(self):
        ### db ###
        # get videoID when video is first uploaded
        
        return videoID

    def get_video_name(self):
        ### db ###
        query="SELECT video_name FROM video where ID=%s"
        cursor.execute(query,(self.video_name))
        row=cursor.fetchone()
        return video_name
    
    def update_video_name(self, new_name: str):
        ### db ###
        
        self.video_name = new_name
        
    def get_video_path(self):
        ### db ###
        return video_path
    
    def initialize_video_path(self, ext):
        project_path = self.get_project_path()
        videoID = self.get_videoID()
        self.video_path = f"{project_path}/{videoID}.{ext}"
        self.save_video_path()

        return self.video_path
    
    def get_frame_count(self):
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count
    
    def get_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    def get_resolution(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_bbox_data(self)->Dict[int,Tuple[float,float,float]]:
        # Output format: {frame_num: (x, y, w, h), ...}
        ### db ###
        bbox_data=()
        with self._connect as conn:
            cursor=conn.cursor()
            cursor.execute("SELECT frame_num,x,y,w,h FROM bounding_boxes ORDER BY frame_num")
            for frmae_num,x,y,w,h in cursor:
                bbox_data[frame_num]=(x,y,w,h)
        return bbox_data
    
    def get_annotation_status(self):
        ### db ###
        ### status can be "yet to start", "manual annotation in progress", "auto annotation in progress", "completed" ###
        ### default is "yet to start" ###
        ### last_annotated_frame is the last frame number that has been annotated (0-indexed) ###
        ### default is None, not 0 ###
        annotation_status='yet to start'
        last_annotated_frame=None
        connection=self._get_connection()
        with connection.cursor() as cursor:
            sql="SELECT annotation_status,last_annotated_frame FROM video_annotation_status LIMIT1"
            cursor.execeute(sql)
        result=cursor.fetchone()
        if result:
            if annotation_status in result and result['annotation_status']:
                annotation_status=result['annotation_status']
            if last_annotated_frame in result:
                last_annotated_frame=result['last_annotated_frame']
            if last_annotated_frame is not None and last_annotated_frame==0:
                pass
        connection.close()
        return annotation_status, last_annotated_frame
    
    ###### Selecting Frame for Manual Annotation ######
    # For testing purpose, annotate every second
    def get_next_frame_to_annotate(self):
        self.frame_count = self.get_frame_count()
        self.fps = self.get_fps()
        if self.annotation_status == "yet to start":
            return self.get_frame(0)
        elif self.annotation_status == "completed":
            return None
        elif isinstance(self.last_annotated_frame, int):
            next_frame = self.last_annotated_frame + self.fps
            if next_frame < self.frame_count:
                return self.get_frame(next_frame)
            else:
                # no more frames to annotate
                self.annotation_status = "manual annotation completed"
                return None
        else:
            return None
    
    def get_frame(self, frame_num: int):
        if frame_num < 0 or frame_num >= self.frame_count:
            raise ValueError("Frame number out of range")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            frame_encoded = base64.b64encode(frame_bytes).decode('utf-8')
        else:
            raise ValueError("Could not read frame")
        return frame_encoded
    
    def annotate(self, frame_num: int, coordinates: list):
        try:
            self.annotation_status = "manual annotation in progress"
            if frame_num < 0 or frame_num >= self.frame_count:
                raise ValueError("Frame number out of range")
            width, height = self.resolution
            x, y, w, h = coordinates[1:5]
            x_center, y_center = x + w/2, y + h/2
            x_normalized, y_normalized, w_normalized, h_normalized = x_center/width, y_center/height, w/width, h/height
            self.bbox_data[frame_num].append(f"{coordinates[0]} {x_normalized} {y_normalized} {w_normalized} {h_normalized}")
            self.last_annotated_frame = frame_num

            # Save data
            self.save_last_annotated_frame()
            self.save_bbox_data(frame_num)
            return True
        
        except Exception as e:
            logger.error(f"Error in annotate: {str(e)}")
            return f"Error in annotate: {str(e)}"

    @staticmethod
    def _calculate_iou(bbox1, bbox2):
        # Unpack the boxes
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate the coordinates of the corners of the boxes
        bbox1_x2 = x1 + w1
        bbox1_y2 = y1 + h1
        bbox2_x2 = x2 + w2
        bbox2_y2 = y2 + h2

        # Calculate the coordinates of the intersection rectangle
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(bbox1_x2, bbox2_x2)
        inter_y2 = min(bbox1_y2, bbox2_y2)

        # Calculate the area of the intersection rectangle
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate the area of both bounding boxes
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Calculate the IoU
        iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
        return iou

    def auto_annotate(self):
        self.annotation_status = "auto annotation in progress"
        
        df = pd.read_csv(self.user_bounded_data_path)

        # Find all annotated frames
        annotated_frames = [index for index, row in df.iterrows() if isinstance(df.iloc[index]["bbox"], str)]

        # Loop through each segment between annotated frames
        for i in range(len(annotated_frames) - 1):
            starting_frame_num = annotated_frames[i]
            ending_frame_num = annotated_frames[i + 1]
            print(f"Auto-annotating frames from {starting_frame_num} to {ending_frame_num}...")

            # Convert the bbox string to a tuple of integers
            starting_frame_bbox = tuple(df.iloc[starting_frame_num]["bbox"])

            # Perform KCF tracking on the first 60 frames
            kcf_tracker = KCF(video_path=self.video_path)
            tracking_results = kcf_tracker.predict_frames(starting_frame_bbox=starting_frame_bbox, starting_frame_num=starting_frame_num, ending_frame_num=ending_frame_num)

            # Find the correct segment for each unbounded frame
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame_num+1)
            for index in range(starting_frame_num+1, ending_frame_num):
                if isinstance(df.iloc[index]["bbox"], float):
                    print(f"Processing frame {index+1}...")
                    ret, frame = cap.read()
                    if not ret:
                        print(f"无法读取第 {index+1} 帧，视频可能结束")
                        break
                    sam_segmenter = SAM(image=frame)
                    bboxes = sam_segmenter.segment()
                    
                    # Find the best matching box using IoU
                    max_iou = -1
                    best_bbox = None
                    target_bbox = tracking_results[index-starting_frame_num]

                    for bbox in bboxes:
                        iou = self._calculate_iou(target_bbox, bbox)
                        if iou > max_iou:
                            max_iou = iou
                            best_bbox = bbox

                    df.iloc[index]['bbox'] = best_bbox
                    print(f"Best matching bbox for frame {index+1}: {best_bbox}")

                    # Save the number of frame in progress
                    self.last_annotated_frame = index
                    self.save_last_annotated_frame()

        cap.release()

        self.annotation_status = "completed"
        self.save_annotation_status()

        return True

    # Save video path to database
    def save_video_path(self):
        ### db ###
        success=False
        video_path=None
        record_id=None
        connection = self._get_connection()
        with connection.cursor() as cursor:
            sql = "UPDATE video SET video_path = %s WHERE id = %s"
            cursor.execute(sql, (video_path, record_id))
        connection.commit()
        success = cursor.rowcount > 0
        connection.close()
        success = True  # data saved successfully
        return success
    
    # Save video name to database
    def save_video_name(self):
        ### db ###
        success=False
        video_name=None
        connection=self._get_connection()
        with connection.cursor() as cursor:
            sql="UPDATE video SET video_name = %s WHERE id=%s"
            cursor.execute(sql,(video_name))
        connection.commit()
        success=cursor.rowcount>0
        connection.close()
        success = True  # data saved successfully
        return success
    
    # Save annotation status to database
    def save_annotation_status(self):
        ### db ### 
        success=False
        annotation_status='not yet started'
        connection=self._get_connection()
        with connection.cursor() as cursor:
            sql="UPDATE video SET annotation_status = %s WHERE id = %s"
            cursor.execute(sql,(annotation_status))
        connection.commit()
        success=cursor.rowcount>0
        connection.close()
        success = True  # data saved successfully
