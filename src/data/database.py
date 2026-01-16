import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import Analysis, Detection, DamageType, Severity, SourceType, BoundingBox


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.connection.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                image_path TEXT NOT NULL,
                total_detections INTEGER DEFAULT 0,
                cracks_count INTEGER DEFAULT 0,
                spalling_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                damage_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL,
                bbox_h INTEGER NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
        """)
        
        self.connection.commit()
    
    def save_analysis(self, analysis: Analysis) -> int:
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO analyses (timestamp, source_type, source_path, image_path, 
                                  total_detections, cracks_count, spalling_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis.timestamp.isoformat(),
            analysis.source_type.value,
            analysis.source_path,
            analysis.image_path,
            analysis.total_detections,
            analysis.cracks_count,
            analysis.spalling_count
        ))
        
        analysis_id = cursor.lastrowid
        
        for detection in analysis.detections:
            cursor.execute("""
                INSERT INTO detections (analysis_id, damage_type, severity, confidence,
                                       bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                detection.damage_type.value,
                detection.severity.value,
                detection.confidence,
                detection.bbox.x,
                detection.bbox.y,
                detection.bbox.width,
                detection.bbox.height
            ))
        
        self.connection.commit()
        return analysis_id
    
    def get_all_analyses(self) -> list[Analysis]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM analyses ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        analyses = []
        for row in rows:
            detections = self._get_detections_for_analysis(row["id"])
            analysis = Analysis(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                source_type=SourceType(row["source_type"]),
                source_path=row["source_path"],
                image_path=row["image_path"],
                total_detections=row["total_detections"],
                cracks_count=row["cracks_count"],
                spalling_count=row["spalling_count"],
                detections=detections
            )
            analyses.append(analysis)
        
        return analyses
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Analysis]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        detections = self._get_detections_for_analysis(analysis_id)
        return Analysis(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            source_type=SourceType(row["source_type"]),
            source_path=row["source_path"],
            image_path=row["image_path"],
            total_detections=row["total_detections"],
            cracks_count=row["cracks_count"],
            spalling_count=row["spalling_count"],
            detections=detections
        )
    
    def _get_detections_for_analysis(self, analysis_id: int) -> list[Detection]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM detections WHERE analysis_id = ?", (analysis_id,))
        rows = cursor.fetchall()
        
        detections = []
        for row in rows:
            detection = Detection(
                id=row["id"],
                damage_type=DamageType(row["damage_type"]),
                severity=Severity(row["severity"]),
                confidence=row["confidence"],
                bbox=BoundingBox(
                    x=row["bbox_x"],
                    y=row["bbox_y"],
                    width=row["bbox_w"],
                    height=row["bbox_h"]
                )
            )
            detections.append(detection)
        
        return detections
    
    def delete_analysis(self, analysis_id: int):
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM detections WHERE analysis_id = ?", (analysis_id,))
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        self.connection.commit()
    
    def close(self):
        if self.connection:
            self.connection.close()
