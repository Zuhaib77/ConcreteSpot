# ConcreteSpot 🏗️

**AI-Powered Concrete Damage Classification System**

A desktop application for detecting and classifying concrete structural damage using deep learning. Built for infrastructure monitoring with offline capability.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

- **4 Damage Types Detection**: Crack, Spalling, Corrosion, Exposed Rebar
- **High Accuracy**: 97.52% mAP@50 on test dataset
- **Offline Operation**: Fully functional without internet
- **Desktop GUI**: Windows 7-style interface using PySide6
- **Multiple Input Modes**: Single image, batch processing, video analysis
- **PDF Reports**: Generate detailed analysis reports
- **History Tracking**: SQLite database for storing results

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **mAP@50** | 97.52% |
| **mAP@50-95** | 81.56% |
| **Precision** | 88.6% |
| **Recall** | 80.9% |

### Per-Class Performance

| Class | AP@50 |
|-------|-------|
| Crack | 96.1% |
| Spalling | 99.5% |
| Corrosion | 80.5% |
| Exposed Rebar | 98.0% |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Zuhaib77/ConcreteSpot.git
cd ConcreteSpot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application (model included in repo)
cd src && python main.py
```

### Dataset Download (For Training)

📁 **[Download Dataset from Google Drive](https://drive.google.com/drive/u/0/folders/1NMA89N6kRFN7ZuZRKDPuJ2VL7paVIfQK)**

Extract to `dataset/` folder:
```
dataset/
├── images/
│   ├── train/  (3334 images)
│   ├── val/    (375 images)
│   └── test/   (238 images)
├── labels/
└── data.yaml
```

---

## 📁 Project Structure

```
ConcreteSpot/
├── src/
│   ├── main.py                 # Application entry point
│   ├── app/
│   │   ├── main_window.py      # Main window UI
│   │   ├── widgets/            # UI components
│   │   └── dialogs/            # Dialog windows
│   ├── core/
│   │   ├── detector.py         # YOLOv8 damage detector
│   │   ├── classifier.py       # Severity classifier
│   │   └── pipeline.py         # Inference pipeline
│   ├── data/
│   │   ├── models.py           # Data models
│   │   ├── database.py         # SQLite manager
│   │   └── storage.py          # File storage
│   ├── reports/
│   │   └── pdf_generator.py    # PDF report generation
│   └── styles/
│       └── win7.py             # Windows 7 stylesheet
├── training/
│   ├── train_detector.py       # YOLOv8 training script
│   ├── train_progressive.py    # Progressive epoch training
│   ├── train_classifier.py     # Severity classifier training
│   ├── evaluate.py             # Model evaluation
│   ├── convert_hrcds.py        # LabelMe to YOLO converter
│   └── convert_voc.py          # Pascal VOC to YOLO converter
├── docs/
│   ├── USER_MANUAL.md          # User guide
│   └── DEVELOPMENT_JOURNEY.md  # Development notes
├── models/                     # Trained models (gitignored)
├── dataset/                    # Training data (gitignored)
├── requirements.txt
├── concretespot.spec           # PyInstaller config
├── build_linux.sh              # Linux build script
├── build_windows.bat           # Windows build script
└── LICENSE
```

---

## 🔧 Training Your Own Model

### Dataset Preparation

1. Organize images in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

2. Label format (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
# Classes: 0=crack, 1=spalling, 2=corrosion, 3=exposed_rebar
```

### Train

```bash
# Basic training
python training/train_detector.py --dataset /path/to/dataset --epochs 100

# Progressive training with graphs
python training/train_progressive.py --epochs 50 100 150 --augmentation balanced
```

### Evaluate

```bash
python training/evaluate.py --model models/yolov8_concrete.pt --data dataset/data.yaml
```

---

## 📦 Building Executable

### Linux
```bash
./build_linux.sh
# Output: dist/ConcreteSpot/ConcreteSpot
```

### Windows
```batch
build_windows.bat
# Use Inno Setup with installer.iss for installer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| GUI Framework | PySide6 (Qt6) |
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, Pillow |
| Database | SQLite |
| PDF Generation | ReportLab |
| Packaging | PyInstaller |

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 👤 Author

**Zuhaib** - [GitHub](https://github.com/Zuhaib77)

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- HRCDS and Multi-Feature Concrete Damage datasets

---

## 📝 Development Log

### January 16-17, 2026 - Progressive Training Experiments

**Environment Setup (Windows)**
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch: 2.9.1+cu126
- Python: 3.13.7
- Ultralytics: 8.4.4

**Dataset (Corrected)**
- Training: 3334 images
- Validation: 375 images
- Test: 238 images
- Classes: crack, spalling, corrosion, exposed_rebar

**Training Runs**

| Run | Epochs | Model Path | mAP@50 | mAP@50-95 |
|-----|--------|------------|--------|-----------|
| Baseline | 100 | `models/yolov8_concrete.pt` | 68.18% | 49.11% |
| Exp-1 | 150 | `runs/detect/yolov8n_150ep_balanced2/weights/best.pt` | 72.03% | 52.02% |
| **Exp-2** | **200** | `runs/detect/yolov8n_200ep_balanced3/weights/best.pt` | **72.61%** | **53.49%** |

**Best Model: 200 epochs (mAP@50: 72.61%)**

| Metric | Score |
|--------|-------|
| mAP@50 | 72.61% |
| mAP@50-95 | 53.49% |
| Precision | 69.95% |
| Recall | 72.17% |

> See [docs/TRAINING_COMPARISON.md](docs/TRAINING_COMPARISON.md) for detailed comparison.

**Next Steps**
- [x] Complete 150 epoch training on full dataset
- [x] Complete 200 epoch training on full dataset
- [x] Evaluate on test set
- [ ] Explore severity-labeled datasets for classifier training
- [ ] YOLO version comparison (v6, v7, v8)

