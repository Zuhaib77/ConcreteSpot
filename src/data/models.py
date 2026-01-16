from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class DamageType(Enum):
    CRACK = "crack"
    SPALLING = "spalling"
    CORROSION = "corrosion"
    EXPOSED_REBAR = "exposed_rebar"


class Severity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class SourceType(Enum):
    IMAGE = "image"
    BATCH = "batch"
    VIDEO = "video"


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Detection:
    id: Optional[int]
    damage_type: DamageType
    severity: Severity
    confidence: float
    bbox: BoundingBox
    
    def to_dict(self):
        return {
            "id": self.id,
            "damage_type": self.damage_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "bbox_x": self.bbox.x,
            "bbox_y": self.bbox.y,
            "bbox_w": self.bbox.width,
            "bbox_h": self.bbox.height
        }


@dataclass
class Analysis:
    id: Optional[int]
    timestamp: datetime
    source_type: SourceType
    source_path: str
    image_path: str
    total_detections: int
    cracks_count: int
    spalling_count: int
    detections: list[Detection]
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type.value,
            "source_path": self.source_path,
            "image_path": self.image_path,
            "total_detections": self.total_detections,
            "cracks_count": self.cracks_count,
            "spalling_count": self.spalling_count
        }
