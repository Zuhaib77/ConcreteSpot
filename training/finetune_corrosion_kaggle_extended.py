"""
Fine-tune Corrosion Kaggle Model - Extended Training
Starting from 74.7% mAP50
Target: 80%+ mAP50
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION FINE-TUNING (Extended)")
    print("Starting from: 74.7% mAP50 (Kaggle best)")
    print("Target: 80%+ mAP50")
    print("Epochs: 150")
    print("=" * 60)
    
    # Start from best Kaggle weights
    model = YOLO("runs/detect/runs/detect/models/specialists/corrosion/train_kaggle_v8s/weights/best.pt")
    
    model.train(
        data="datasets/specialists/corrosion_kaggle_yolo/data.yaml",
        epochs=150,
        batch=16,
        imgsz=640,
        patience=75,
        
        # Heavy augmentation for rust detection
        mosaic=1.0,
        mixup=0.25,
        copy_paste=0.15,
        degrees=20.0,
        scale=0.7,
        shear=8.0,
        perspective=0.0005,
        flipud=0.3,
        fliplr=0.5,
        
        # Heavy color augmentation (critical for rust)
        hsv_h=0.025,
        hsv_s=0.75,
        hsv_v=0.45,
        
        erasing=0.35,
        
        # Lower LR for fine-tuning
        lr0=0.0005,
        lrf=0.01,
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="finetune_kaggle_150ep",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("Weights: runs/detect/models/specialists/corrosion/finetune_kaggle_150ep/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
