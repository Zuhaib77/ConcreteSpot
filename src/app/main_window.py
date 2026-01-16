from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

from .widgets.image_viewer import ImageViewer
from .widgets.single_image_tab import SingleImageTab
from .widgets.batch_tab import BatchTab
from .widgets.video_tab import VideoTab
from .widgets.history_tab import HistoryTab

from core.pipeline import InferencePipeline
from data.database import Database
from data.storage import Storage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ConcreteSpot v1.0.0 - Concrete Damage Classification")
        self.setMinimumSize(1024, 768)
        
        self._setup_paths()
        self._init_components()
        self._setup_ui()
        self._setup_statusbar()
    
    def _setup_paths(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.app_dir / "data"
        self.models_dir = self.app_dir / "models"
        
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def _init_components(self):
        self.database = Database(self.data_dir / "database.db")
        self.storage = Storage(self.data_dir)
        
        detector_path = self.models_dir / "yolov8_concrete.pt"
        classifier_path = self.models_dir / "inceptionv3_severity.pt"
        
        self.pipeline = InferencePipeline(
            detector_model_path=detector_path if detector_path.exists() else None,
            classifier_model_path=classifier_path if classifier_path.exists() else None
        )
    
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        self.tab_widget = QTabWidget()
        
        self.single_image_tab = SingleImageTab(
            self.pipeline, self.database, self.storage
        )
        self.batch_tab = BatchTab(
            self.pipeline, self.database, self.storage
        )
        self.video_tab = VideoTab(
            self.pipeline, self.database, self.storage
        )
        self.history_tab = HistoryTab(
            self.database, self.storage
        )
        
        self.tab_widget.addTab(self.single_image_tab, "Single Image")
        self.tab_widget.addTab(self.batch_tab, "Batch Processing")
        self.tab_widget.addTab(self.video_tab, "Video Analysis")
        self.tab_widget.addTab(self.history_tab, "History")
        
        main_layout.addWidget(self.tab_widget)
    
    def _setup_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.model_label = QLabel()
        self._update_model_status()
        self.status_bar.addPermanentWidget(self.model_label)
        
        self.gpu_label = QLabel()
        self._update_gpu_status()
        self.status_bar.addPermanentWidget(self.gpu_label)
    
    def _update_model_status(self):
        detector_path = self.models_dir / "yolov8_concrete.pt"
        classifier_path = self.models_dir / "inceptionv3_severity.pt"
        
        if detector_path.exists() and classifier_path.exists():
            self.model_label.setText("Models: Custom")
            self.model_label.setStyleSheet("color: green;")
        elif detector_path.exists() or classifier_path.exists():
            self.model_label.setText("Models: Partial")
            self.model_label.setStyleSheet("color: orange;")
        else:
            self.model_label.setText("Models: Demo Mode")
            self.model_label.setStyleSheet("color: #888;")
    
    def _update_gpu_status(self):
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.gpu_label.setText(f"GPU: {gpu_name}")
            self.gpu_label.setStyleSheet("color: green;")
        else:
            self.gpu_label.setText("GPU: Not Available (CPU Mode)")
            self.gpu_label.setStyleSheet("color: orange;")
    
    def set_status(self, message: str):
        self.status_label.setText(message)
    
    def closeEvent(self, event):
        self.database.close()
        event.accept()

