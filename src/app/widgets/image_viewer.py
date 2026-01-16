from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QImage

import numpy as np
import cv2

from data.models import Detection, Severity


class ImageViewer(QWidget):
    image_clicked = Signal(int, int)
    
    def __init__(self):
        super().__init__()
        
        self.original_pixmap = None
        self.current_pixmap = None
        self.detections = []
        self.scale_factor = 1.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #2d2d2d;")
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
    
    def set_image(self, image_path: str = None, image_array: np.ndarray = None):
        if image_path:
            self.original_pixmap = QPixmap(image_path)
        elif image_array is not None:
            if len(image_array.shape) == 3:
                height, width, channel = image_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    image_array.data, width, height,
                    bytes_per_line, QImage.Format_RGB888
                )
            else:
                height, width = image_array.shape
                bytes_per_line = width
                q_image = QImage(
                    image_array.data, width, height,
                    bytes_per_line, QImage.Format_Grayscale8
                )
            self.original_pixmap = QPixmap.fromImage(q_image)
        
        self.detections = []
        self._update_display()
    
    def set_detections(self, detections: list[Detection]):
        self.detections = detections
        self._update_display()
    
    def _update_display(self):
        if self.original_pixmap is None:
            return
        
        pixmap = self.original_pixmap.copy()
        
        if self.detections:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # TODO: Uncomment when severity classifier is trained
            # severity_colors = {
            #     Severity.MINOR: QColor(0, 200, 0),
            #     Severity.MODERATE: QColor(255, 165, 0),
            #     Severity.SEVERE: QColor(255, 0, 0)
            # }
            
            # Use damage type colors instead
            damage_colors = {
                'crack': QColor(255, 100, 100),       # Red
                'spalling': QColor(100, 100, 255),    # Blue
                'corrosion': QColor(255, 165, 0),     # Orange
                'exposed_rebar': QColor(200, 0, 200)  # Purple
            }
            
            for detection in self.detections:
                # color = severity_colors.get(detection.severity, QColor(255, 255, 0))
                color = damage_colors.get(detection.damage_type.value, QColor(0, 200, 0))
                
                pen = QPen(color, 3)
                painter.setPen(pen)
                
                painter.drawRect(
                    detection.bbox.x,
                    detection.bbox.y,
                    detection.bbox.width,
                    detection.bbox.height
                )
                
                # label = f"{detection.damage_type.value} ({detection.severity.value})"
                label = f"{detection.damage_type.value} ({detection.confidence:.0%})"
                
                font = painter.font()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                
                label_rect = painter.fontMetrics().boundingRect(label)
                label_x = detection.bbox.x
                label_y = detection.bbox.y - 5
                
                painter.fillRect(
                    label_x - 2,
                    label_y - label_rect.height() - 2,
                    label_rect.width() + 8,
                    label_rect.height() + 4,
                    color
                )
                
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    label_x + 2,
                    label_y - 4,
                    label
                )
            
            painter.end()
        
        scaled_size = self.scroll_area.size() - QSize(20, 20)
        self.current_pixmap = pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(self.current_pixmap)
    
    def clear(self):
        self.original_pixmap = None
        self.current_pixmap = None
        self.detections = []
        self.image_label.clear()
        self.image_label.setStyleSheet("background-color: #2d2d2d;")
    
    def get_annotated_image(self) -> np.ndarray:
        if self.original_pixmap is None:
            return None
        
        image = self.original_pixmap.toImage()
        width = image.width()
        height = image.height()
        
        ptr = image.bits()
        arr = np.array(ptr).reshape(height, width, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
