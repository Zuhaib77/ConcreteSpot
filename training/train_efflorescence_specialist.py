"""
Train Efflorescence Specialist Model
Same settings as crack specialist: 100 epochs, batch 16, YOLOv8s
"""
from ultralytics import YOLO
import os

def main():
    # Change to project directory
    os.chdir(r"C:\Users\spect\ConcreteSpotDetection")
    
    print("=" * 60)
    print("EFFLORESCENCE SPECIALIST TRAINING")
    print("Dataset: dataset/specialists/efflorescence")
    print("Epochs: 100")
    print("=" * 60)
    
    # Load base model
    model = YOLO("yolov8s.pt")
    
    # Train
    model.train(
        data="dataset/specialists/efflorescence/data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        project="runs/detect/models/specialists/efflorescence",
        name="train",
        patience=100,
        save_period=25,
        exist_ok=True
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Weights: runs/detect/models/specialists/efflorescence/train/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
