from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image

from .detector import ConcreteDetector
from .classifier import SeverityClassifier
from data.models import Analysis, Detection, SourceType, Severity


class InferencePipeline:
    def __init__(
        self,
        detector_model_path: Optional[Path] = None,
        classifier_model_path: Optional[Path] = None
    ):
        self.detector = ConcreteDetector(detector_model_path)
        self.classifier = SeverityClassifier(classifier_model_path)
        self._initialized = False
    
    def initialize(self):
        if not self._initialized:
            self.detector.load_model()
            self.classifier.load_model()
            self._initialized = True
    
    def process_image(
        self,
        image_path: Path,
        stored_path: Optional[Path] = None
    ) -> Analysis:
        self.initialize()
        
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = self.detector.detect(image_rgb)
        
        for detection in detections:
            x = detection.bbox.x
            y = detection.bbox.y
            w = detection.bbox.width
            h = detection.bbox.height
            
            x = max(0, x)
            y = max(0, y)
            x2 = min(image_rgb.shape[1], x + w)
            y2 = min(image_rgb.shape[0], y + h)
            
            if x2 > x and y2 > y:
                crop = image_rgb[y:y2, x:x2]
                detection.severity = self.classifier.classify(crop)
        
        cracks_count = sum(1 for d in detections if d.damage_type.value == "crack")
        spalling_count = sum(1 for d in detections if d.damage_type.value == "spalling")
        
        analysis = Analysis(
            id=None,
            timestamp=datetime.now(),
            source_type=SourceType.IMAGE,
            source_path=str(image_path),
            image_path=str(stored_path or image_path),
            total_detections=len(detections),
            cracks_count=cracks_count,
            spalling_count=spalling_count,
            detections=detections
        )
        
        return analysis
    
    def process_batch(
        self,
        image_paths: list[Path],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[Analysis]:
        self.initialize()
        
        analyses = []
        total = len(image_paths)
        
        for idx, image_path in enumerate(image_paths):
            analysis = self.process_image(image_path)
            analysis.source_type = SourceType.BATCH
            analyses.append(analysis)
            
            if progress_callback:
                progress_callback(idx + 1, total)
        
        return analyses
    
    def process_video(
        self,
        video_path: Path,
        frame_interval: int = 30,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[Analysis]:
        self.initialize()
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        analyses = []
        frame_count = 0
        processed_count = 0
        expected_frames = total_frames // frame_interval
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                detections = self.detector.detect(frame_rgb)
                
                for detection in detections:
                    x = detection.bbox.x
                    y = detection.bbox.y
                    w = detection.bbox.width
                    h = detection.bbox.height
                    
                    x = max(0, x)
                    y = max(0, y)
                    x2 = min(frame_rgb.shape[1], x + w)
                    y2 = min(frame_rgb.shape[0], y + h)
                    
                    if x2 > x and y2 > y:
                        crop = frame_rgb[y:y2, x:x2]
                        detection.severity = self.classifier.classify(crop)
                
                cracks_count = sum(1 for d in detections if d.damage_type.value == "crack")
                spalling_count = sum(1 for d in detections if d.damage_type.value == "spalling")
                
                analysis = Analysis(
                    id=None,
                    timestamp=datetime.now(),
                    source_type=SourceType.VIDEO,
                    source_path=str(video_path),
                    image_path=f"frame_{frame_count}",
                    total_detections=len(detections),
                    cracks_count=cracks_count,
                    spalling_count=spalling_count,
                    detections=detections
                )
                analyses.append(analysis)
                processed_count += 1
                
                if progress_callback:
                    progress_callback(processed_count, expected_frames)
            
            frame_count += 1
        
        cap.release()
        return analyses
    
    def draw_detections(self, image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        result = image.copy()
        
        severity_colors = {
            Severity.MINOR: (0, 255, 0),
            Severity.MODERATE: (0, 165, 255),
            Severity.SEVERE: (0, 0, 255)
        }
        
        for detection in detections:
            x = detection.bbox.x
            y = detection.bbox.y
            w = detection.bbox.width
            h = detection.bbox.height
            
            color = severity_colors.get(detection.severity, (255, 255, 0))
            
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            label = f"{detection.damage_type.value} ({detection.severity.value})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(
                result,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 10, y),
                color,
                -1
            )
            
            cv2.putText(
                result,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result
