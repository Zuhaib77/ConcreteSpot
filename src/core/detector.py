from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from data.models import Detection, DamageType, Severity, BoundingBox


class ConcreteDetector:
    def __init__(self, model_path: Optional[Path] = None, use_ensemble: bool = False):
        self.model: Optional[YOLO] = None
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cam = None  # CAM visualizer (lazy load)
        self.use_ensemble = use_ensemble
        self.ensemble_detector = None  # Lazy load
        
        self.class_map = {
            0: DamageType.CRACK,
            1: DamageType.SPALLING,
            2: DamageType.CORROSION,
            3: DamageType.EXPOSED_REBAR,
            4: DamageType.EFFLORESCENCE
        }
        
        self.class_colors = {
            DamageType.CRACK: (255, 50, 50),
            DamageType.SPALLING: (50, 255, 50),
            DamageType.CORROSION: (255, 165, 0),
            DamageType.EXPOSED_REBAR: (50, 50, 255),
            DamageType.EFFLORESCENCE: (255, 255, 50),
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
        # Use ensemble if enabled
        if self.use_ensemble:
            if self.ensemble_detector is None:
                from core.ensemble_detector import EnsembleDetector
                self.ensemble_detector = EnsembleDetector()
                self.ensemble_detector.load_models()
            return self.ensemble_detector.detect(image)
        
        # Single model inference
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
    
    def detect_with_cam(self, image: np.ndarray, return_overlay: bool = True) -> tuple[list[Detection], np.ndarray]:
        """
        Run detection with CAM heatmap visualization.
        
        Args:
            image: Input image (BGR format from OpenCV)
            return_overlay: If True, return overlay with heatmap + boxes
            
        Returns:
            Tuple of (detections, visualization_image)
        """
        if self.model is None:
            self.load_model()
        
        # Get detections
        detections = self.detect(image)
        
        # Initialize CAM if needed
        if self.cam is None:
            try:
                from core.gradcam import YOLOv8CAM
                self.cam = YOLOv8CAM(str(self.model_path))
            except Exception as e:
                print(f"CAM not available: {e}")
                return detections, image
        
        # Generate CAM heatmap
        # Save temp image for CAM processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, image)
            result = self.cam.generate_cam(f.name)
        
        if return_overlay and 'overlay' in result:
            overlay = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
            return detections, overlay
        else:
            return detections, image
    
    def draw_detections(self, image: np.ndarray, detections: list[Detection], 
                       show_heatmap: bool = False) -> np.ndarray:
        """Draw detection boxes on image."""
        output = image.copy()
        
        for det in detections:
            color = self.class_colors.get(det.damage_type, (255, 255, 255))
            x1, y1 = det.bbox.x, det.bbox.y
            x2, y2 = x1 + det.bbox.width, y1 + det.bbox.height
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{det.damage_type.value}: {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output
    
    def is_loaded(self) -> bool:
        return self.model is not None

