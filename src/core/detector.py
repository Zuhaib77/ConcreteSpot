from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from data.models import Detection, DamageType, Severity, BoundingBox


class ConcreteDetector:
    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[YOLO] = None
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.class_map = {
            0: DamageType.CRACK,
            1: DamageType.SPALLING,
            2: DamageType.CORROSION,
            3: DamageType.EXPOSED_REBAR
        }
    
    def load_model(self, model_path: Optional[Path] = None):
        if model_path:
            self.model_path = model_path
        
        if self.model_path and self.model_path.exists():
            self.model = YOLO(str(self.model_path))
        else:
            self.model = YOLO("yolov8n.pt")
        
        self.model.to(self.device)
    
    def detect(self, image: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.load_model()
        
        results = self.model(image, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                
                if cls in self.class_map:
                    damage_type = self.class_map[cls]
                else:
                    damage_type = DamageType.CRACK
                
                detection = Detection(
                    id=None,
                    damage_type=damage_type,
                    severity=Severity.MODERATE,
                    confidence=float(conf),
                    bbox=BoundingBox(
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1
                    )
                )
                detections.append(detection)
        
        return detections
    
    def is_loaded(self) -> bool:
        return self.model is not None
