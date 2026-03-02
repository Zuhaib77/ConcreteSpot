"""
Train Spalling Specialist Model
Replicating the 98.6% mAP50 achieved in yolov8n_200ep_balanced3

Original Settings (from runs/archive/yolov8n_200ep_balanced3/args.yaml):
- Model: YOLOv8n (not YOLOv8s!)
- Epochs: 200
- Batch: 16
- Image Size: 640
- Dataset: 3334 train, 375 val images
- Patience: 0 (train all epochs)
- Augmentation: mosaic=0.5, degrees=10, erasing=0.2

Per-class results at 200 epochs:
- Spalling: 98.6% mAP50
- Exposed Rebar: 79.2%
- Corrosion: 78.7%
- Crack: 34.0%
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("SPALLING SPECIALIST TRAINING")
    print("Replicating 98.6% mAP50 from original training")
    print("=" * 60)
    print()
    print("Settings:")
    print("  Model: YOLOv8n (same as original)")
    print("  Epochs: 200")
    print("  Batch: 16")
    print("  Dataset: datasets/specialists/spalling")
    print("=" * 60)
    
    # Use YOLOv8n (same as original training!)
    model = YOLO("yolov8n.pt")
    
    # Train with EXACT settings from yolov8n_200ep_balanced3
    model.train(
        data="datasets/specialists/spalling/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=0,  # Train all epochs (no early stopping)
        
        # Augmentation settings from original
        mosaic=0.5,
        degrees=10.0,
        erasing=0.2,
        
        # Output
        project="runs/detect/models/specialists/spalling",
        name="train",
        exist_ok=True,
        
        # Other settings
        workers=4,
        device='0',
        amp=True,
        deterministic=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/spalling/train/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
