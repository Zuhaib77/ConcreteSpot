#!/bin/bash
# GCP A100 Training Setup Script for ConcreteSpot
# Run this AFTER SSH into your GCP VM

set -e  # Exit on error

echo "=============================================="
echo "ConcreteSpot GCP A100 Training Setup"
echo "=============================================="

# 1. Verify GPU
echo ""
echo "[1/5] Verifying GPU..."
python3 -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}'); print(f'✓ CUDA: {torch.version.cuda}')"

# 2. Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install --quiet ultralytics gdown tqdm

# 3. Create directory structure
echo ""
echo "[3/5] Creating directories..."
mkdir -p ~/concretespot/dataset
cd ~/concretespot

# 4. Download dataset (user needs to set GDRIVE_FILE_ID)
echo ""
echo "[4/5] Dataset download..."
if [ -z "$GDRIVE_FILE_ID" ]; then
    echo "⚠ GDRIVE_FILE_ID not set. Please run:"
    echo "   export GDRIVE_FILE_ID=your_file_id"
    echo "   ./setup_gcp.sh"
    echo ""
    echo "Or manually download with:"
    echo "   gdown https://drive.google.com/uc?id=YOUR_ID -O balanced_95plus.zip"
else
    gdown https://drive.google.com/uc?id=$GDRIVE_FILE_ID -O balanced_95plus.zip
    unzip -q balanced_95plus.zip -d dataset/
    echo "✓ Dataset extracted"
fi

# 5. Create training script
echo ""
echo "[5/5] Creating training script..."
cat << 'TRAINEOF' > train.py
"""
ConcreteSpot YOLOv8s Training on GCP A100
Optimized for 40GB VRAM
"""
from ultralytics import YOLO
import torch

print("=" * 50)
print("ConcreteSpot Training - GCP A100")
print("=" * 50)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")
print("=" * 50)

# Load model
model = YOLO("yolov8s.pt")

# Train with A100-optimized settings
results = model.train(
    data="dataset/balanced_95plus/data.yaml",
    epochs=150,
    imgsz=640,
    batch=64,           # A100 handles large batches
    workers=8,
    device=0,
    patience=30,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=5,
    cls=1.5,            # Increased class loss weight
    project="runs",
    name="yolov8s_a100",
    exist_ok=True,
    save=True,
    save_period=25,
    plots=True,
    verbose=True
)

print("\n" + "=" * 50)
print("Training Complete!")
print(f"Best model: runs/yolov8s_a100/weights/best.pt")
print("=" * 50)
TRAINEOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To start training:"
echo "  1. screen -S training"
echo "  2. python train.py"
echo "  3. Ctrl+A, then D to detach"
echo ""
echo "To monitor:"
echo "  nvidia-smi -l 1"
echo "  tail -f runs/yolov8s_a100/results.csv"
echo ""
