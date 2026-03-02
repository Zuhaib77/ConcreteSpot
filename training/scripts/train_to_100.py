"""
Train baseline model from epoch 68 to 100
Creates checkpoints for gradual extension (200, 300, 400 epochs)
"""
from ultralytics import YOLO
import shutil
from pathlib import Path

def main():
    print("="*60)
    print("RESUMING TRAINING: Epoch 68 → 100")
    print("="*60)

    # Load from last checkpoint
    model = YOLO("runs/detect/runs/train/yolov8s_balanced_5class/weights/last.pt")

    # Train to epoch 100
    results = model.train(
        data="dataset/dataset/balanced_95plus/data.yaml",
        epochs=100,
        resume=True,
        project="runs/train",
        name="yolov8s_baseline",
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        workers=0  # Fix for Windows multiprocessing
    )

    # Create checkpoint directory
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights for gradual extension
    weights_dir = Path("runs/train/yolov8s_baseline/weights")
    if weights_dir.exists():
        shutil.copy(weights_dir / "best.pt", checkpoint_dir / "baseline_100ep_best.pt")
        shutil.copy(weights_dir / "last.pt", checkpoint_dir / "baseline_100ep_last.pt")
        print()
        print("="*60)
        print("CHECKPOINTS SAVED")
        print("="*60)
        print(f"  {checkpoint_dir / 'baseline_100ep_best.pt'}")
        print(f"  {checkpoint_dir / 'baseline_100ep_last.pt'}")
        print()
        print("To extend to 200 epochs:")
        print('  model = YOLO("models/checkpoints/baseline_100ep_last.pt")')
        print("  model.train(epochs=200, resume=True)")

    print()
    print("Training complete!")

if __name__ == "__main__":
    main()
