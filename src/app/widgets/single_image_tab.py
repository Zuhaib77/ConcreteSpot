from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from .image_viewer import ImageViewer
from core.pipeline import InferencePipeline
from data.database import Database
from data.storage import Storage
from data.models import Analysis
from reports.pdf_generator import PDFGenerator


class AnalysisWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, pipeline: InferencePipeline, image_path: Path):
        super().__init__()
        self.pipeline = pipeline
        self.image_path = image_path
    
    def run(self):
        try:
            analysis = self.pipeline.process_image(self.image_path)
            self.finished.emit(analysis)
        except Exception as e:
            self.error.emit(str(e))


class SingleImageTab(QWidget):
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
        
        self.current_image_path = None
        self.current_analysis = None
        self.worker = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self._on_upload)
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        
        self.report_btn = QPushButton("Generate PDF")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self._on_generate_report)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        
        controls_layout.addWidget(self.upload_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.report_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()
        
        left_layout.addWidget(controls_group)
        
        self.image_viewer = ImageViewer()
        left_layout.addWidget(self.image_viewer, 1)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.status_label = QLabel("No image loaded")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        
        self.total_label = QLabel("Total Detections: -")
        self.cracks_label = QLabel("Cracks: -")
        self.spalling_label = QLabel("Spalling: -")
        
        summary_layout.addWidget(self.status_label)
        summary_layout.addWidget(self.total_label)
        summary_layout.addWidget(self.cracks_label)
        summary_layout.addWidget(self.spalling_label)
        
        right_layout.addWidget(summary_group)
        
        detections_group = QGroupBox("Detections")
        detections_layout = QVBoxLayout(detections_group)
        
        self.detections_table = QTableWidget()
        self.detections_table.setColumnCount(3)  # TODO: Change to 4 when severity is ready
        self.detections_table.setHorizontalHeaderLabels([
            "Type", "Confidence", "Location"  # TODO: Add "Severity" when ready
        ])
        self.detections_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detections_table.setAlternatingRowColors(True)
        
        detections_layout.addWidget(self.detections_table)
        right_layout.addWidget(detections_group, 1)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
    
    def _on_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.image_viewer.set_image(file_path)
            self.analyze_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {self.current_image_path.name}")
            self._reset_results()
    
    def _on_analyze(self):
        if not self.current_image_path:
            return
        
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analyzing...")
        
        self.worker = AnalysisWorker(self.pipeline, self.current_image_path)
        self.worker.finished.connect(self._on_analysis_complete)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()
    
    def _on_analysis_complete(self, analysis: Analysis):
        self.current_analysis = analysis
        
        self.image_viewer.set_detections(analysis.detections)
        
        self.status_label.setText("Analysis Complete")
        self.total_label.setText(f"Total Detections: {analysis.total_detections}")
        self.cracks_label.setText(f"Cracks: {analysis.cracks_count}")
        self.spalling_label.setText(f"Spalling: {analysis.spalling_count}")
        
        self.detections_table.setRowCount(len(analysis.detections))
        for row, det in enumerate(analysis.detections):
            self.detections_table.setItem(row, 0, QTableWidgetItem(det.damage_type.value))
            # TODO: Uncomment when severity classifier is trained
            # self.detections_table.setItem(row, 1, QTableWidgetItem(det.severity.value))
            self.detections_table.setItem(row, 1, QTableWidgetItem(f"{det.confidence:.2%}"))
            self.detections_table.setItem(
                row, 2,
                QTableWidgetItem(f"({det.bbox.x}, {det.bbox.y})")
            )
        
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.report_btn.setEnabled(True)
    
    def _on_analysis_error(self, error_msg: str):
        self.status_label.setText("Analysis Failed")
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
    
    def _on_save(self):
        if not self.current_analysis:
            return
        
        stored_path = self.storage.save_image(self.current_image_path)
        self.current_analysis.image_path = str(stored_path)
        
        analysis_id = self.database.save_analysis(self.current_analysis)
        self.current_analysis.id = analysis_id
        
        QMessageBox.information(
            self, "Saved",
            f"Results saved with ID: {analysis_id}"
        )
    
    def _on_generate_report(self):
        if not self.current_analysis:
            return
        
        if not self.current_analysis.id:
            self._on_save()
        
        report_path = self.storage.get_report_path(self.current_analysis.id)
        
        generator = PDFGenerator()
        generator.generate(self.current_analysis, report_path, self.current_image_path)
        
        QMessageBox.information(
            self, "Report Generated",
            f"PDF saved to: {report_path}"
        )
    
    def _on_clear(self):
        self.current_image_path = None
        self.current_analysis = None
        self.image_viewer.clear()
        self._reset_results()
        self.analyze_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.report_btn.setEnabled(False)
    
    def _reset_results(self):
        self.total_label.setText("Total Detections: -")
        self.cracks_label.setText("Cracks: -")
        self.spalling_label.setText("Spalling: -")
        self.detections_table.setRowCount(0)
