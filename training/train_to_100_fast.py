"""
Resume training from epoch 78 to 100 (FAST version with workers=4)
"""
from ultralytics import YOLO
import shutil
from pathlib import Path
from multiprocessing import freeze_support

def main():
    print("="*60)
    print("RESUMING TRAINING: Epoch 78 → 100 (FAST)")
    print("="*60)

    # Load from last checkpoint
    model = YOLO("runs/detect/runs/train/yolov8s_balanced_5class/weights/last.pt")

    # Train to epoch 100 with workers=4 for speed
    results = model.train(
        data="dataset/dataset/balanced_95plus/data.yaml",
        epochs=100,
        resume=True,
        project="runs/detect/runs/train",
        name="yolov8s_balanced_5class",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=4  # Fast parallel data loading
    )

    # Create checkpoint directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights
    weights_dir = Path("runs/detect/runs/train/yolov8s_balanced_5class/weights")
    shutil.copy(weights_dir / "best.pt", checkpoint_dir / "baseline_100ep_best.pt")
    shutil.copy(weights_dir / "last.pt", checkpoint_dir / "baseline_100ep_last.pt")
    
    print()
    print("="*60)
    print("TRAINING COMPLETE - 100 EPOCHS")
    print("Checkpoints saved to models/checkpoints/")
    print("="*60)

if __name__ == "__main__":
    freeze_support()
    main()
