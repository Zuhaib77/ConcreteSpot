def get_win7_stylesheet():
    return """
    QMainWindow {
        background-color: #f0f0f0;
    }
    
    QWidget {
        background-color: #f0f0f0;
        color: #000000;
        font-family: "Segoe UI", sans-serif;
        font-size: 9pt;
    }
    
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:0.5 #e8e8e8, stop:1 #d4d4d4);
        border: 1px solid #8e8f8f;
        border-radius: 3px;
        padding: 5px 15px;
        min-height: 20px;
        color: #000000;
    }
    
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #e8f4fc, stop:0.5 #c7e2f6, stop:1 #ade2fc);
        border: 1px solid #3c7fb1;
    }
    
    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #c7e2f6, stop:0.5 #a9d9f5, stop:1 #89cef3);
        border: 1px solid #2c628b;
    }
    
    QPushButton:disabled {
        background: #f4f4f4;
        border: 1px solid #adb2b5;
        color: #838383;
    }
    
    QTabWidget::pane {
        border: 1px solid #8e8f8f;
        background-color: #ffffff;
        border-radius: 3px;
    }
    
    QTabBar::tab {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e0e0e0);
        border: 1px solid #8e8f8f;
        border-bottom: none;
        border-top-left-radius: 3px;
        border-top-right-radius: 3px;
        padding: 6px 16px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background: #ffffff;
        border-bottom: 1px solid #ffffff;
    }
    
    QTabBar::tab:hover:!selected {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #e8f4fc, stop:1 #c7e2f6);
    }
    
    QLineEdit, QTextEdit, QPlainTextEdit {
        background-color: #ffffff;
        border: 1px solid #abadb3;
        border-radius: 2px;
        padding: 4px;
        selection-background-color: #3399ff;
        selection-color: #ffffff;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 1px solid #569de5;
    }
    
    QComboBox {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e8e8e8);
        border: 1px solid #abadb3;
        border-radius: 2px;
        padding: 4px 8px;
        min-height: 20px;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        width: 10px;
        height: 10px;
    }
    
    QProgressBar {
        border: 1px solid #8e8f8f;
        border-radius: 2px;
        background-color: #e6e6e6;
        text-align: center;
        height: 20px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #37b54a, stop:0.5 #2d9a3e, stop:1 #258033);
        border-radius: 1px;
    }
    
    QGroupBox {
        border: 1px solid #d4d4d4;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 8px;
        font-weight: bold;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        background-color: #f0f0f0;
    }
    
    QListWidget, QTableWidget, QTreeWidget {
        background-color: #ffffff;
        border: 1px solid #abadb3;
        alternate-background-color: #f5f5f5;
    }
    
    QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {
        background-color: #3399ff;
        color: #ffffff;
    }
    
    QListWidget::item:hover, QTableWidget::item:hover, QTreeWidget::item:hover {
        background-color: #e5f3ff;
    }
    
    QScrollBar:vertical {
        background: #f0f0f0;
        width: 17px;
        border: 1px solid #d4d4d4;
    }
    
    QScrollBar::handle:vertical {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #e8e8e8, stop:0.5 #d4d4d4, stop:1 #c0c0c0);
        min-height: 30px;
        border-radius: 2px;
        margin: 1px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #c7e2f6, stop:0.5 #a9d9f5, stop:1 #89cef3);
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 17px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e0e0e0);
        border: 1px solid #d4d4d4;
    }
    
    QScrollBar:horizontal {
        background: #f0f0f0;
        height: 17px;
        border: 1px solid #d4d4d4;
    }
    
    QScrollBar::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #e8e8e8, stop:0.5 #d4d4d4, stop:1 #c0c0c0);
        min-width: 30px;
        border-radius: 2px;
        margin: 1px;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 17px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e0e0e0);
        border: 1px solid #d4d4d4;
    }
    
    QMenuBar {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e0e0e0);
        border-bottom: 1px solid #d4d4d4;
    }
    
    QMenuBar::item {
        padding: 4px 10px;
        background: transparent;
    }
    
    QMenuBar::item:selected {
        background: #3399ff;
        color: #ffffff;
    }
    
    QMenu {
        background-color: #ffffff;
        border: 1px solid #8e8f8f;
    }
    
    QMenu::item {
        padding: 6px 30px 6px 20px;
    }
    
    QMenu::item:selected {
        background-color: #3399ff;
        color: #ffffff;
    }
    
    QStatusBar {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #f5f5f5, stop:1 #e0e0e0);
        border-top: 1px solid #d4d4d4;
    }
    
    QLabel {
        background: transparent;
    }
    
    QToolTip {
        background-color: #ffffe1;
        border: 1px solid #000000;
        color: #000000;
        padding: 2px;
    }
    """
