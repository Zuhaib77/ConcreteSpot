"""
Continue Corrosion Kaggle Training - 50 more epochs
Train from best.pt with 50 additional epochs
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION KAGGLE - 50 MORE EPOCHS")
    print("Starting from: 74.7% mAP50 best weights")
    print("Additional epochs: 50")
    print("=" * 60)
    
    # Load best weights and train more epochs
    model = YOLO("runs/detect/runs/detect/models/specialists/corrosion/train_kaggle_v8s/weights/best.pt")
    
    model.train(
        data="datasets/specialists/corrosion_kaggle_yolo/data.yaml",
        epochs=50,
        batch=16,
        imgsz=640,
        patience=50,
        
        # Same augmentation as original
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        scale=0.7,
        erasing=0.3,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Lower LR since starting from trained weights
        lr0=0.001,
        lrf=0.01,
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="kaggle_more_epochs",
        exist_ok=True,
        
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
