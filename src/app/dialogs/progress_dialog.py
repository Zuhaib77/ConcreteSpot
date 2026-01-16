from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
)
from PySide6.QtCore import Qt


class ProgressDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Processing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
    
    def set_progress(self, value: int, maximum: int = 100):
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
    
    def set_status(self, text: str):
        self.status_label.setText(text)
