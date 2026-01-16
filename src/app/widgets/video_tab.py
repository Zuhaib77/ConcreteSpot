from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QSlider,
    QProgressBar, QSplitter, QMessageBox, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal

from .image_viewer import ImageViewer
from core.pipeline import InferencePipeline
from data.database import Database
from data.storage import Storage
from data.models import Analysis

import cv2


class VideoWorker(QThread):
    progress = Signal(int, int)
    frame_ready = Signal(object, object)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, pipeline: InferencePipeline, video_path: Path, interval: int):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = video_path
        self.interval = interval
    
    def run(self):
        try:
            analyses = self.pipeline.process_video(
                self.video_path,
                frame_interval=self.interval,
                progress_callback=lambda c, t: self.progress.emit(c, t)
            )
            self.finished.emit(analyses)
        except Exception as e:
            self.error.emit(str(e))


class VideoTab(QWidget):
    def __init__(
        self,
        pipeline: InferencePipeline,
        database: Database,
        storage: Storage
    ):
        super().__init__()
        
        self.pipeline = pipeline
        self.database = database
        self.storage = storage
        
        self.video_path = None
        self.analyses = []
        self.worker = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        controls_group = QGroupBox("Video Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.open_btn = QPushButton("Open Video")
        self.open_btn.clicked.connect(self._on_open_video)
        
        controls_layout.addWidget(self.open_btn)
        controls_layout.addWidget(QLabel("Frame Interval:"))
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 300)
        self.interval_spin.setValue(30)
        self.interval_spin.setToolTip("Analyze every Nth frame")
        controls_layout.addWidget(self.interval_spin)
        
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._on_process)
        controls_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        controls_layout.addWidget(self.save_btn)
        
        controls_layout.addStretch()
        layout.addWidget(controls_group)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_group)
        
        splitter = QSplitter(Qt.Horizontal)
        
        viewer_group = QGroupBox("Frame Preview")
        viewer_layout = QVBoxLayout(viewer_group)
        
        self.image_viewer = ImageViewer()
        viewer_layout.addWidget(self.image_viewer)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        viewer_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Frame: -")
        viewer_layout.addWidget(self.frame_label)
        
        results_group = QGroupBox("Frame Results")
        results_layout = QVBoxLayout(results_group)
        
        self.video_info_label = QLabel("No video loaded")
        self.results_label = QLabel("")
        
        results_layout.addWidget(self.video_info_label)
        results_layout.addWidget(self.results_label)
        results_layout.addStretch()
        
        splitter.addWidget(viewer_group)
        splitter.addWidget(results_group)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter, 1)
    
    def _on_open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        
        if file_path:
            self.video_path = Path(file_path)
            
            cap = cv2.VideoCapture(str(self.video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            self.video_info_label.setText(
                f"Video: {self.video_path.name}\n"
                f"Resolution: {width}x{height}\n"
                f"FPS: {fps}\n"
                f"Frames: {total_frames}\n"
                f"Duration: {duration:.1f}s"
            )
            
            self.process_btn.setEnabled(True)
            self.analyses = []
    
    def _on_process(self):
        if not self.video_path:
            return
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing video...")
        
        self.worker = VideoWorker(
            self.pipeline,
            self.video_path,
            self.interval_spin.value()
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_process_complete)
        self.worker.error.connect(self._on_process_error)
        self.worker.start()
    
    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Processing frame {current}/{total}")
    
    def _on_process_complete(self, analyses):
        self.analyses = analyses
        self.progress_label.setText(f"Complete - {len(analyses)} frames analyzed")
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, len(analyses) - 1)
        self.frame_slider.setValue(0)
        self._on_frame_changed(0)
    
    def _on_process_error(self, error_msg):
        self.progress_label.setText("Error occurred")
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Video processing failed: {error_msg}")
    
    def _on_frame_changed(self, idx):
        if idx < 0 or idx >= len(self.analyses):
            return
        
        analysis = self.analyses[idx]
        
        cap = cv2.VideoCapture(str(self.video_path))
        frame_num = idx * self.interval_spin.value()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image_viewer.set_image(image_array=frame_rgb)
            self.image_viewer.set_detections(analysis.detections)
        
        self.frame_label.setText(f"Frame: {frame_num}")
        self.results_label.setText(
            f"Detections: {analysis.total_detections}\n"
            f"Cracks: {analysis.cracks_count}\n"
            f"Spalling: {analysis.spalling_count}"
        )
    
    def _on_save(self):
        if not self.analyses:
            return
        
        saved = 0
        for analysis in self.analyses:
            self.database.save_analysis(analysis)
            saved += 1
        
        QMessageBox.information(
            self, "Saved",
            f"Saved {saved} frame analyses to database"
        )
