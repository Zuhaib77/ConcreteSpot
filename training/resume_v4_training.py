"""
Resume V4 Training
Continues training from the last checkpoint
"""
from ultralytics import YOLO

# Load from last checkpoint
model = YOLO('runs/detect/models/v4_organized/train/weights/last.pt')

# Resume training
model.train(
    resume=True,  # Critical: resume from checkpoint
    # The following are already saved in the checkpoint:
    # data, epochs, batch, imgsz, project, name
)

print("\n" + "="*60)
print("V4 TRAINING RESUMED")
print("="*60)
print("Checkpoint: runs/detect/models/v4_organized/train/weights/last.pt")
print("Output: models/v4_organized/train/")
print("="*60)
