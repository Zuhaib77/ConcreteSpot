from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QListWidget,
    QListWidgetItem, QProgressBar, QSplitter, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from .image_viewer import ImageViewer
from core.pipeline import InferencePipeline
from data.database import Database
from data.storage import Storage
from data.models import Analysis


class BatchWorker(QThread):
    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, pipeline: InferencePipeline, image_paths: list):
        super().__init__()
        self.pipeline = pipeline
        self.image_paths = image_paths
    
    def run(self):
        try:
            analyses = self.pipeline.process_batch(
                self.image_paths,
                progress_callback=lambda current, total: self.progress.emit(current, total)
            )
            self.finished.emit(analyses)
        except Exception as e:
            self.error.emit(str(e))


class BatchTab(QWidget):
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
        
        self.image_paths = []
        self.analyses = []
        self.worker = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        controls_group = QGroupBox("Batch Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self._on_select_folder)
        
        self.select_files_btn = QPushButton("Select Files")
        self.select_files_btn.clicked.connect(self._on_select_files)
        
        self.process_btn = QPushButton("Process All")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._on_process)
        
        self.save_all_btn = QPushButton("Save All")
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.clicked.connect(self._on_save_all)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        
        controls_layout.addWidget(self.select_folder_btn)
        controls_layout.addWidget(self.select_files_btn)
        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(self.save_all_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        
        self.progress_label = QLabel("Ready")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        splitter = QSplitter(Qt.Horizontal)
        
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)
        
        self.files_list = QListWidget()
        self.files_list.currentRowChanged.connect(self._on_file_selected)
        
        files_layout.addWidget(self.files_list)
        
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        preview_group = QGroupBox("Preview")
        preview_inner = QVBoxLayout(preview_group)
        
        self.image_viewer = ImageViewer()
        preview_inner.addWidget(self.image_viewer)
        
        preview_layout.addWidget(preview_group)
        
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.result_label = QLabel("Select an image to view results")
        results_layout.addWidget(self.result_label)
        
        preview_layout.addWidget(results_group)
        
        splitter.addWidget(files_group)
        splitter.addWidget(preview_widget)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter, 1)
    
    def _on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Images"
        )
        
        if folder:
            folder_path = Path(folder)
            self.image_paths = []
            
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                self.image_paths.extend(folder_path.glob(ext))
            
            self._update_files_list()
    
    def _on_select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if files:
            self.image_paths = [Path(f) for f in files]
            self._update_files_list()
    
    def _update_files_list(self):
        self.files_list.clear()
        self.analyses = []
        
        for path in self.image_paths:
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            self.files_list.addItem(item)
        
        self.process_btn.setEnabled(len(self.image_paths) > 0)
        self.progress_bar.setMaximum(len(self.image_paths))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"{len(self.image_paths)} images selected")
    
    def _on_process(self):
        if not self.image_paths:
            return
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing...")
        
        self.worker = BatchWorker(self.pipeline, self.image_paths)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_batch_complete)
        self.worker.error.connect(self._on_batch_error)
        self.worker.start()
    
    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Processing {current}/{total}")
    
    def _on_batch_complete(self, analyses):
        self.analyses = analyses
        self.progress_label.setText(f"Complete - {len(analyses)} images processed")
        self.process_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)
        
        for i, analysis in enumerate(analyses):
            item = self.files_list.item(i)
            item.setText(f"{item.text()} ({analysis.total_detections} detections)")
    
    def _on_batch_error(self, error_msg):
        self.progress_label.setText("Error occurred")
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Batch processing failed: {error_msg}")
    
    def _on_file_selected(self, row):
        if row < 0 or row >= len(self.image_paths):
            return
        
        image_path = self.image_paths[row]
        self.image_viewer.set_image(str(image_path))
        
        if row < len(self.analyses):
            analysis = self.analyses[row]
            self.image_viewer.set_detections(analysis.detections)
            self.result_label.setText(
                f"Detections: {analysis.total_detections}\n"
                f"Cracks: {analysis.cracks_count}\n"
                f"Spalling: {analysis.spalling_count}"
            )
        else:
            self.result_label.setText("Not yet analyzed")
    
    def _on_save_all(self):
        if not self.analyses:
            return
        
        saved_count = 0
        for i, analysis in enumerate(self.analyses):
            stored_path = self.storage.save_image(self.image_paths[i])
            analysis.image_path = str(stored_path)
            self.database.save_analysis(analysis)
            saved_count += 1
        
        QMessageBox.information(
            self, "Saved",
            f"Saved {saved_count} analyses to database"
        )
    
    def _on_clear(self):
        self.image_paths = []
        self.analyses = []
        self.files_list.clear()
        self.image_viewer.clear()
        self.result_label.setText("Select an image to view results")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self.process_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
