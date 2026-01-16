#!/usr/bin/env python3
"""
YOLOv8 Training Script for Concrete Damage Detection
=====================================================
Trains YOLOv8 to detect: crack, spalling

Dataset Structure Required:
    dataset/
    ├── images/
    │   ├── train/    (your training images)
    │   └── val/      (your validation images)
    ├── labels/
    │   ├── train/    (YOLO format .txt files)
    │   └── val/
    └── data.yaml

YOLO Label Format (each .txt file):
    <class_id> <x_center> <y_center> <width> <height>
    
    class_id: 0 = crack, 1 = spalling
    coordinates: normalized 0-1 relative to image size
    
Example label file (image1.txt):
    0 0.45 0.32 0.12 0.08
    1 0.78 0.65 0.15 0.10
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


def create_data_yaml(dataset_path: Path):
    data_yaml_content = f"""
path: {dataset_path.absolute()}
train: images/train
val: images/val

names:
  0: crack
  1: spalling

nc: 2
"""
    yaml_path = dataset_path / "data.yaml"
    yaml_path.write_text(data_yaml_content.strip())
    print(f"Created {yaml_path}")
    return yaml_path


def train_yolov8(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    model_size: str = "n"
):
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist!")
        print("\nExpected structure:")
        print("  dataset/images/train/  - training images")
        print("  dataset/images/val/    - validation images")
        print("  dataset/labels/train/  - training labels (.txt)")
        print("  dataset/labels/val/    - validation labels (.txt)")
        return
    
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        data_yaml = create_data_yaml(dataset_path)
    
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading pretrained {model_name}...")
    model = YOLO(model_name)
    
    print(f"\nStarting training:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {image_size}")
    print(f"  - Dataset: {dataset_path}")
    print("-" * 50)
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name="concrete_detector",
        patience=20,
        save=True,
        plots=True,
        amp=True,
        workers=4,
    )
    
    best_model = Path("runs/detect/concrete_detector/weights/best.pt")
    output_path = Path("models/yolov8_concrete.pt")
    output_path.parent.mkdir(exist_ok=True)
    
    if best_model.exists():
        import shutil
        shutil.copy(best_model, output_path)
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {output_path}")
        print(f"\n   Copy this file to your ConcreteSpot/models/ folder")
    else:
        print("\n⚠️ Training completed but best.pt not found")
        print(f"   Check runs/detect/concrete_detector/weights/")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for concrete damage detection")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size: n(nano), s(small), m(medium), l(large), x(extra)")
    
    args = parser.parse_args()
    
    train_yolov8(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        model_size=args.model
    )
