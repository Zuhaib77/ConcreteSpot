"""
Train Corrosion Specialist Model
Using CONCORNET2023 dataset - high quality corrosion detection dataset
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("CORROSION SPECIALIST TRAINING")
    print("Dataset: CONCORNET2023 (632 train, 79 val)")
    print("Model: YOLOv8n")
    print("Epochs: 100")
    print("=" * 60)
    
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="datasets/specialists/corrosion_new/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=0,
        
        # Augmentation settings
        mosaic=0.5,
        degrees=10.0,
        erasing=0.2,
        
        # Output
        project="runs/detect/models/specialists/corrosion",
        name="train",
        exist_ok=True,
        
        # Other
        workers=4,
        device='0',
        amp=True,
        deterministic=True,
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/corrosion/train/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
