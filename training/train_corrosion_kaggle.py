"""
Train Corrosion Specialist Model with Kaggle Dataset
Target: 95%+ mAP50
Dataset: 1164 train, 134 val (converted from segmentation)
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION SPECIALIST TRAINING (Kaggle Dataset)")
    print("Target: 95%+ mAP50")
    print("Model: YOLOv8s")
    print("Dataset: 1164 train, 134 val")
    print("Epochs: 100")
    print("=" * 60)
    
    # Use YOLOv8s for best results
    model = YOLO("yolov8s.pt")
    
    model.train(
        data="datasets/specialists/corrosion_kaggle_yolo/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=50,
        
        # Enhanced augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        scale=0.7,
        erasing=0.3,
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="train_kaggle_v8s",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/corrosion/train_kaggle_v8s/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
