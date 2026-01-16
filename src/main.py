import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from app.main_window import MainWindow
from styles.win7 import get_win7_stylesheet


def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("ConcreteSpot")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ConcreteSpot")
    
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    app.setStyleSheet(get_win7_stylesheet())
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "reports").mkdir(exist_ok=True)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
