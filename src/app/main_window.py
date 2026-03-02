from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QStatusBar, QLabel, QMessageBox,
    QComboBox, QGroupBox
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
    # Available models configuration
    AVAILABLE_MODELS = {
        "V1 - 68K Balanced (47% mAP)": "models/v1_68k_balanced_100ep/best.pt",
        "V2 - Curated Quality": "models/versions/v2_curated/best.pt",
        "V3 - Pure Quality (66% mAP)": "models/versions/v3_pure_quality/best.pt",
        "V4 - Organized (63% mAP)": "models/versions/v4_organized/best.pt",
        "V5 - dacl10k": "runs/detect/runs/detect/models/v5_dacl10k/train/weights/best.pt",
        "V6 - Semi-Auto": "runs/detect/runs/detect/models/v6_semiauto/train/weights/best.pt",
        "Ensemble (92.8% mAP)": "ENSEMBLE",
        "CODEBRIM Trained (v2 Fresh)": "models/codebrim_v1.pt",
        "YOLOv8 Concrete (Default)": "models/yolov8_concrete.pt",
    }
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ConcreteSpot v1.0.0 - Concrete Damage Classification")
        self.setMinimumSize(1024, 768)
        
        self._setup_paths()
        self._init_components()
        self._setup_statusbar()
        self._setup_ui()
    
    def _setup_paths(self):
        self.app_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.app_dir / "data"
        self.models_dir = self.app_dir / "models"
        
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def _init_components(self):
        self.database = Database(self.data_dir / "database.db")
        self.storage = Storage(self.data_dir)
        
        # Start with V3 as default (best single model)
        detector_path = self.app_dir / "models/versions/v3_pure_quality/best.pt"
        if not detector_path.exists():
            detector_path = self.models_dir / "yolov8_concrete.pt"
        
        classifier_path = self.models_dir / "inceptionv3_severity.pt"
        
        self.pipeline = InferencePipeline(
            detector_model_path=detector_path if detector_path.exists() else None,
            classifier_model_path=classifier_path if classifier_path.exists() else None
        )
        self.current_model_name = "V3 - Pure Quality (66% mAP)"
    
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Model selector at the top
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout(model_group)
        
        model_label = QLabel("Detection Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        
        # Add available models
        for name, path in self.AVAILABLE_MODELS.items():
            full_path = self.app_dir / path if path != "ENSEMBLE" else None
            if path == "ENSEMBLE" or (full_path and full_path.exists()):
                self.model_combo.addItem(name, path)
        
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        # Set current model
        index = self.model_combo.findText(self.current_model_name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        
        main_layout.addWidget(model_group)
        
        # Tab widget
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
    
    def _on_model_changed(self, model_name: str):
        """Handle model selection change."""
        path = self.model_combo.currentData()
        
        if path == "ENSEMBLE":
            # Enable ensemble mode
            self.pipeline.detector.use_ensemble = True
            self.pipeline.detector.ensemble_detector = None  # Force reload
            self.set_status(f"Switched to Ensemble Mode (5 specialists)")
            self.model_label.setText(f"Model: Ensemble")
        else:
            # Single model mode
            self.pipeline.detector.use_ensemble = False
            full_path = self.app_dir / path
            if full_path.exists():
                self.pipeline.detector.model_path = full_path
                self.pipeline.detector.model = None  # Force reload
                self.set_status(f"Switched to {model_name}")
                self.model_label.setText(f"Model: {model_name}")
            else:
                QMessageBox.warning(self, "Model Not Found", f"Model file not found:\n{full_path}")
        
        self.current_model_name = model_name
    
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
        self.model_label.setText(f"Model: {self.current_model_name}")
        self.model_label.setStyleSheet("color: green;")
    
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
