"""
Fine-tune Corrosion Specialist Model
Starting from 73.9% mAP50, targeting 85%+
Heavy augmentation for rust/corrosion color variations
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION FINE-TUNING (Heavy Augmentation)")
    print("Starting from: 73.9% mAP50")
    print("Target: 85%+ mAP50")
    print("=" * 60)
    
    # Start from best weights
    model = YOLO("runs/detect/runs/detect/models/specialists/corrosion/train_kaggle_v8s/weights/best.pt")
    
    model.train(
        data="datasets/specialists/corrosion_kaggle_yolo/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=50,
        
        # Heavy augmentation for rust/corrosion variations
        mosaic=1.0,
        mixup=0.3,          # Increased
        copy_paste=0.2,     # Increased
        degrees=20.0,       # More rotation
        scale=0.8,          # More scale variation
        shear=10.0,         # Add shear
        perspective=0.001,  # Slight perspective
        flipud=0.3,         # Vertical flip (rust can be any orientation)
        fliplr=0.5,         # Horizontal flip
        
        # Color augmentation (critical for rust detection)
        hsv_h=0.03,         # Hue variation (rust colors)
        hsv_s=0.8,          # Saturation variation
        hsv_v=0.5,          # Value/brightness variation
        
        erasing=0.4,        # Random erasing
        
        # Lower LR for fine-tuning
        lr0=0.001,          # Lower initial LR
        lrf=0.01,           # Final LR factor
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="finetune_v8s_heavy_aug",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
    )
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("Weights: runs/detect/models/specialists/corrosion/finetune_v8s_heavy_aug/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
