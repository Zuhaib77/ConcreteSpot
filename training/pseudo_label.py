"""
Pseudo-Labeling Script
Runs trained model on unlabeled images to generate pseudo-labels.
Only keeps high-confidence predictions.
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Paths
BASE_DIR = Path("dataset/dataset")
UNLABELED_DIR = BASE_DIR / "kaggle_crack_unlabeled/Concrete Crack Images for Classification/Positive"
TARGET_TRAIN_IMAGES = BASE_DIR / "images/train"
TARGET_TRAIN_LABELS = BASE_DIR / "labels/train"

# Model and confidence threshold
MODEL_PATH = "models/yolov8_concrete.pt"
CONF_THRESHOLD = 0.7  # Only keep high-confidence predictions
MAX_IMAGES = 10000  # Limit to avoid overwhelming the dataset

def pseudo_label():
    print("=" * 60)
    print("Pseudo-Labeling for Semi-Supervised Learning")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Get images
    images = list(UNLABELED_DIR.glob("*.jpg")) + list(UNLABELED_DIR.glob("*.png"))
    print(f"Found {len(images)} images in Positive folder")
    
    if len(images) > MAX_IMAGES:
        import random
        random.seed(42)
        images = random.sample(images, MAX_IMAGES)
        print(f"Sampling {MAX_IMAGES} images")
    
    added = 0
    skipped = 0
    
    for img_path in tqdm(images, desc="Pseudo-labeling"):
        # Run inference
        results = model.predict(img_path, verbose=False, conf=CONF_THRESHOLD)
        
        if len(results[0].boxes) == 0:
            skipped += 1
            continue
        
        # Get image dimensions
        img_h, img_w = results[0].orig_shape
        
        # Build YOLO label lines
        lines = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get normalized bbox
            x_c, y_c, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        if not lines:
            skipped += 1
            continue
        
        # Copy image with unique prefix
        new_img_name = f"pseudo_{img_path.name}"
        new_lbl_name = f"pseudo_{img_path.stem}.txt"
        
        target_img = TARGET_TRAIN_IMAGES / new_img_name
        target_lbl = TARGET_TRAIN_LABELS / new_lbl_name
        
        if not target_img.exists():
            shutil.copy(img_path, target_img)
        
        with open(target_lbl, 'w') as f:
            f.write('\n'.join(lines))
        
        added += 1
    
    print("\n" + "=" * 60)
    print(f"Pseudo-labeled: {added}")
    print(f"Skipped (low confidence): {skipped}")
    print("=" * 60)
    
    # Count final dataset
    final_images = len(list(TARGET_TRAIN_IMAGES.glob("*")))
    final_labels = len(list(TARGET_TRAIN_LABELS.glob("*.txt")))
    print(f"\nFinal dataset:")
    print(f"  Images: {final_images}")
    print(f"  Labels: {final_labels}")

if __name__ == "__main__":
    pseudo_label()
