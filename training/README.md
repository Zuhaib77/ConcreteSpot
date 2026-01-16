# Model Training Guide

## Quick Start

### 1. Train YOLOv8 Detector

```bash
cd /home/zuhaib/Projects_files/DRDO/Concrete_Damage_Classification
source venv/bin/activate

python training/train_detector.py \
    --dataset /path/to/your/detection_dataset \
    --epochs 100 \
    --batch 16 \
    --model n
```

### 2. Train InceptionV3 Classifier

```bash
python training/train_classifier.py \
    --dataset /path/to/your/classification_dataset \
    --epochs 30 \
    --batch 32
```

## Dataset Requirements

### For YOLOv8 (Detection)

```
detection_dataset/
├── images/
│   ├── train/     # 80% of your 2500 images
│   └── val/       # 20% of your 2500 images
├── labels/
│   ├── train/     # One .txt per image (YOLO format)
│   └── val/
└── data.yaml      # Auto-created by script
```

**YOLO Label Format** (one .txt file per image):
```
0 0.45 0.32 0.12 0.08
1 0.78 0.65 0.15 0.10
```
- Class 0 = crack, Class 1 = spalling
- Values are: x_center, y_center, width, height (normalized 0-1)

### For InceptionV3 (Classification)

```
classification_dataset/
├── train/
│   ├── minor/      # Minor damage images
│   ├── moderate/   # Moderate damage images
│   └── severe/     # Severe damage images
└── val/
    ├── minor/
    ├── moderate/
    └── severe/
```

## Annotation Tools

- **LabelImg**: Simple bounding box annotation
  ```bash
  pip install labelImg
  labelImg
  ```

- **Roboflow**: Online annotation + dataset management
  https://roboflow.com

- **Label Studio**: Full-featured, self-hosted
  https://labelstud.io

## After Training

Copy the trained models to the `models/` folder:
```bash
cp models/yolov8_concrete.pt ../models/
cp models/inceptionv3_severity.pt ../models/
```

The app will automatically use them on next launch!
