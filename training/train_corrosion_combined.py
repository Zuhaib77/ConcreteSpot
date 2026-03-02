"""
Train Corrosion Specialist with Combined Dataset
1796 train images (OBB + Kaggle combined)
Target: 85%+ mAP50
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION TRAINING (Combined Dataset)")
    print("Train: 1796 images (OBB + Kaggle)")
    print("Valid: 188 images")
    print("Target: 85%+ mAP50")
    print("=" * 60)
    
    # Use YOLOv8s
    model = YOLO("yolov8s.pt")
    
    model.train(
        data="datasets/specialists/corrosion_combined/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=50,
        
        # Good augmentation
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.15,
        degrees=15.0,
        scale=0.6,
        shear=5.0,
        flipud=0.2,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="train_combined_v8s",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/corrosion/train_combined_v8s/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
