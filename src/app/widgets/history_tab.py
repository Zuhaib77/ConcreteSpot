from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox
)
from PySide6.QtCore import Qt

from .image_viewer import ImageViewer
from data.database import Database
from data.storage import Storage
from data.models import Analysis
from reports.pdf_generator import PDFGenerator


class HistoryTab(QWidget):
    def __init__(self, database: Database, storage: Storage):
        super().__init__()
        
        self.database = database
        self.storage = storage
        
        self.analyses = []
        self.selected_analysis = None
        
        self._setup_ui()
        self._load_history()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        controls_group = QGroupBox("History Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_history)
        
        self.report_btn = QPushButton("Generate Report")
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self._on_generate_report)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._on_delete)
        
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.report_btn)
        controls_layout.addWidget(self.delete_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        splitter = QSplitter(Qt.Horizontal)
        
        table_group = QGroupBox("Analysis History")
        table_layout = QVBoxLayout(table_group)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "ID", "Date", "Source", "Detections", "Cracks", "Spalling"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.itemSelectionChanged.connect(self._on_selection_changed)
        
        table_layout.addWidget(self.history_table)
        
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.image_viewer = ImageViewer()
        preview_layout.addWidget(self.image_viewer)
        
        splitter.addWidget(table_group)
        splitter.addWidget(preview_group)
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter, 1)
    
    def _load_history(self):
        self.analyses = self.database.get_all_analyses()
        
        self.history_table.setRowCount(len(self.analyses))
        
        for row, analysis in enumerate(self.analyses):
            self.history_table.setItem(row, 0, QTableWidgetItem(str(analysis.id)))
            self.history_table.setItem(
                row, 1,
                QTableWidgetItem(analysis.timestamp.strftime("%Y-%m-%d %H:%M"))
            )
            self.history_table.setItem(row, 2, QTableWidgetItem(analysis.source_type.value))
            self.history_table.setItem(row, 3, QTableWidgetItem(str(analysis.total_detections)))
            self.history_table.setItem(row, 4, QTableWidgetItem(str(analysis.cracks_count)))
            self.history_table.setItem(row, 5, QTableWidgetItem(str(analysis.spalling_count)))
    
    def _on_selection_changed(self):
        rows = self.history_table.selectedItems()
        if not rows:
            self.selected_analysis = None
            self.report_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.image_viewer.clear()
            return
        
        row = self.history_table.currentRow()
        if row < 0 or row >= len(self.analyses):
            return
        
        self.selected_analysis = self.analyses[row]
        self.report_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        
        image_path = Path(self.selected_analysis.image_path)
        if image_path.exists():
            self.image_viewer.set_image(str(image_path))
            self.image_viewer.set_detections(self.selected_analysis.detections)
    
    def _on_generate_report(self):
        if not self.selected_analysis:
            return
        
        report_path = self.storage.get_report_path(self.selected_analysis.id)
        image_path = Path(self.selected_analysis.image_path)
        
        generator = PDFGenerator()
        generator.generate(self.selected_analysis, report_path, image_path)
        
        QMessageBox.information(
            self, "Report Generated",
            f"PDF saved to: {report_path}"
        )
    
    def _on_delete(self):
        if not self.selected_analysis:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete analysis #{self.selected_analysis.id}?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.database.delete_analysis(self.selected_analysis.id)
            self._load_history()
            self.image_viewer.clear()
    
    def showEvent(self, event):
        super().showEvent(event)
        self._load_history()
