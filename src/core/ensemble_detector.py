"""
Ensemble Detector - Combines 5 specialist models for high-accuracy detection.

Models:
- Crack: 99.5% mAP50
- Efflorescence: 99.5% mAP50
- Spalling: 98.9% mAP50
- Exposed Rebar: 91.5% mAP50
- Corrosion: 74.7% mAP50

Combined Average: 92.8% mAP50
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from ultralytics import YOLO

from data.models import Detection, DamageType, Severity, BoundingBox


class EnsembleDetector:
    """
    Ensemble detector using 5 specialist models for each damage type.
    Each specialist outputs only its target class, results are merged with NMS.
    """
    
    # Default paths for specialist models
    SPECIALIST_PATHS = {
        DamageType.CRACK: "models/specialists/crack/best.pt",
        DamageType.EFFLORESCENCE: "models/specialists/efflorescence/best.pt",
        DamageType.SPALLING: "models/specialists/spalling/spalling_98pct.pt",
        DamageType.CORROSION: "models/specialists/corrosion/best.pt",
        DamageType.EXPOSED_REBAR: "models/specialists/exposed_rebar/best.pt",
    }
    
    # Class-specific confidence thresholds (HIGHER = less false positives)
    CONFIDENCE_THRESHOLDS = {
        DamageType.CRACK: 0.50,           # High accuracy model - stricter threshold
        DamageType.EFFLORESCENCE: 0.50,   # High accuracy model - stricter threshold  
        DamageType.SPALLING: 0.45,        # Good accuracy - moderate threshold
        DamageType.CORROSION: 0.40,       # Lower accuracy - slightly lower threshold
        DamageType.EXPOSED_REBAR: 0.45,   # Good accuracy - moderate threshold
    }
    
    # Colors for visualization
    CLASS_COLORS = {
        DamageType.CRACK: (255, 50, 50),        # Red
        DamageType.EFFLORESCENCE: (255, 255, 50), # Yellow
        DamageType.SPALLING: (50, 255, 50),      # Green
        DamageType.CORROSION: (255, 165, 0),     # Orange
        DamageType.EXPOSED_REBAR: (50, 50, 255), # Blue
    }
    
    def __init__(self, base_path: Optional[Path] = None, device: str = None):
        """
        Initialize ensemble detector.
        
        Args:
            base_path: Base path for model files. If None, uses current directory.
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.base_path = Path(base_path) if base_path else Path(".")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[DamageType, YOLO] = {}
        self.loaded = False
    
    def load_models(self) -> bool:
        """
        Load all specialist models.
        
        Returns:
            True if all models loaded successfully.
        """
        print("Loading ensemble specialist models...")
        loaded_count = 0
        
        for damage_type, rel_path in self.SPECIALIST_PATHS.items():
            model_path = self.base_path / rel_path
            
            if model_path.exists():
                try:
                    model = YOLO(str(model_path))
                    model.to(self.device)
                    self.models[damage_type] = model
                    print(f"  ✓ {damage_type.value}: {model_path.name}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  ✗ {damage_type.value}: Failed to load - {e}")
            else:
                print(f"  ✗ {damage_type.value}: Model not found at {model_path}")
        
        self.loaded = loaded_count > 0
        print(f"Loaded {loaded_count}/{len(self.SPECIALIST_PATHS)} specialist models")
        return loaded_count == len(self.SPECIALIST_PATHS)
    
    def detect(self, image: np.ndarray, nms_iou: float = 0.5) -> list[Detection]:
        """
        Run ensemble detection on image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            nms_iou: IoU threshold for NMS merging
            
        Returns:
            List of Detection objects
        """
        if not self.loaded:
            self.load_models()
        
        all_detections = []
        
        # Run each specialist model
        for damage_type, model in self.models.items():
            threshold = self.CONFIDENCE_THRESHOLDS.get(damage_type, 0.25)
            
            results = model(image, verbose=False, conf=threshold)
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    detection = Detection(
                        id=None,
                        damage_type=damage_type,
                        severity=self._estimate_severity(conf),
                        confidence=float(conf),
                        bbox=BoundingBox(
                            x=x1, y=y1,
                            width=x2 - x1,
                            height=y2 - y1
                        )
                    )
                    all_detections.append(detection)
        
        # Apply NMS across all detections
        merged = self._apply_nms(all_detections, nms_iou)
        
        return merged
    
    def _estimate_severity(self, confidence: float) -> Severity:
        """Estimate damage severity based on confidence."""
        if confidence >= 0.8:
            return Severity.SEVERE
        elif confidence >= 0.5:
            return Severity.MODERATE
        else:
            return Severity.MINOR
    
    def _apply_nms(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        """
        Apply Non-Maximum Suppression to merge overlapping detections.
        
        Same-class detections with IoU > threshold are merged.
        Different-class detections are kept even if overlapping.
        """
        if not detections:
            return []
        
        # Group by damage type
        by_type: dict[DamageType, list[Detection]] = {}
        for det in detections:
            if det.damage_type not in by_type:
                by_type[det.damage_type] = []
            by_type[det.damage_type].append(det)
        
        result = []
        
        for damage_type, dets in by_type.items():
            # Sort by confidence (descending)
            dets.sort(key=lambda d: d.confidence, reverse=True)
            
            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)
                
                # Remove overlapping boxes
                remaining = []
                for det in dets:
                    iou = self._compute_iou(best.bbox, det.bbox)
                    if iou < iou_threshold:
                        remaining.append(det)
                dets = remaining
            
            result.extend(keep)
        
        return result
    
    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, image: np.ndarray, 
                        detections: list[Detection]) -> np.ndarray:
        """Draw detection boxes on image."""
        import cv2
        output = image.copy()
        
        for det in detections:
            color = self.CLASS_COLORS.get(det.damage_type, (255, 255, 255))
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
        """Check if models are loaded."""
        return self.loaded and len(self.models) > 0
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "loaded": self.loaded,
            "device": self.device,
            "models": {dt.value: True for dt in self.models.keys()},
            "total_models": len(self.models),
            "expected_models": len(self.SPECIALIST_PATHS),
        }
