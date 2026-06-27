import os
import torch
from ultralytics import YOLO
import supervision as sv

class FootballDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        
        # Check if the custom model exists. If not, fallback to COCO model
        if os.path.exists(model_path):
            print(f"Loading custom football detector from {model_path}...")
            self.model = YOLO(model_path)
            self.is_custom = True
        else:
            fallback_model = "yolov8s.pt" if device == "cuda" else "yolov8n.pt"
            print(f"WARNING: Custom model {model_path} not found! Falling back to baseline {fallback_model}...")
            self.model = YOLO(fallback_model)
            self.is_custom = False

        print("Model classes:", self.model.names)
        
        # Create helper mappings
        self.name_to_id = {name: idx for idx, name in self.model.names.items()}
        
        if self.is_custom:
            self.player_id = self.name_to_id.get("player")
            self.goalkeeper_id = self.name_to_id.get("goalkeeper")
            self.referee_id = self.name_to_id.get("referee")
            self.ball_id = self.name_to_id.get("ball")
        else:
            # Fallback mappings for COCO model
            self.player_id = self.name_to_id.get("person", 0)
            self.goalkeeper_id = None
            self.referee_id = None
            self.ball_id = self.name_to_id.get("sports ball") # COCO sports ball is class 32
            
        # Class IDs that should be tracked (players, goalkeepers, referees)
        self.tracked_class_ids = [
            class_id for class_id in [self.player_id, self.goalkeeper_id, self.referee_id]
            if class_id is not None
        ]

    def detect(self, frame, conf: float = 0.15, imgsz: int = 1280):
        """
        Run YOLOv8 inference on a frame and return sv.Detections
        """
        result = self.model(
            frame,
            device=self.device,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(result)
        return detections
