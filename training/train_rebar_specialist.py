"""
Train Exposed Rebar Specialist Model
Using converted rebar dataset (VOC -> YOLO)
"""
from ultralytics import YOLO
import os

def main():
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("EXPOSED REBAR SPECIALIST TRAINING")
    print("Dataset: Rebar Dataset (converted from VOC format)")
    print("Model: YOLOv8n")
    print("Epochs: 100")
    print("=" * 60)
    
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="datasets/specialists/exposed_rebar/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        patience=0,
        
        # Augmentation settings
        mosaic=0.5,
        degrees=10.0,
        erasing=0.2,
        
        # Output
        project="runs/detect/models/specialists/exposed_rebar",
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
    print("Weights: runs/detect/models/specialists/exposed_rebar/train/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
