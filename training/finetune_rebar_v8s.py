"""
Fine-tune Exposed Rebar Specialist with YOLOv8s
Target: 95%+ mAP50 (current: 87.5% with YOLOv8n)

Changes from previous training:
- YOLOv8s instead of YOLOv8n (larger model)
- 150 epochs (more training)
- Enhanced augmentation
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("EXPOSED REBAR FINE-TUNING WITH YOLOv8s")
    print("Target: 95%+ mAP50")
    print("Model: YOLOv8s (upgraded from YOLOv8n)")
    print("Epochs: 150")
    print("=" * 60)
    
    # Use YOLOv8s (larger model)
    model = YOLO("yolov8s.pt")
    
    model.train(
        data="datasets/specialists/exposed_rebar/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=50,  # Early stopping
        
        # Enhanced augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        scale=0.7,
        erasing=0.3,
        
        # Output
        project="runs/detect/models/specialists/exposed_rebar",
        name="train_v8s",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/exposed_rebar/train_v8s/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
