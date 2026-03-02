"""
dacl10k Supervisely to YOLO Converter
Converts polygon annotations to bounding boxes in YOLO format.
Only extracts relevant damage classes.
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

# Base paths
BASE_DIR = Path("dataset/dataset")
DACL_DIR = BASE_DIR / "dacl10k_supervisely"
TARGET_TRAIN_IMAGES = BASE_DIR / "images/train"
TARGET_TRAIN_LABELS = BASE_DIR / "labels/train"

# Class mapping: dacl10k classTitle -> our class index
CLASS_MAPPING = {
    "crack": 0,
    "alligator crack": 0,  # map to crack
    "spalling": 1,
    "rust": 2,  # map to corrosion
    "washouts/concrete corrosion": 2,  # map to corrosion
    "exposed rebars": 3,
}

def polygon_to_bbox(points: list, img_width: int, img_height: int) -> tuple:
    """Convert polygon points to YOLO bbox (x_center, y_center, width, height) normalized."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Compute center and size
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Clamp to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def convert_annotation(ann_path: Path) -> list:
    """Convert a single Supervisely annotation to YOLO format."""
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_width = data['size']['width']
    img_height = data['size']['height']
    
    lines = []
    for obj in data.get('objects', []):
        class_title = obj.get('classTitle', '').lower()
        
        if class_title not in CLASS_MAPPING:
            continue  # Skip irrelevant classes
        
        class_id = CLASS_MAPPING[class_title]
        
        # Get polygon points
        points = obj.get('points', {}).get('exterior', [])
        if len(points) < 3:
            continue  # Invalid polygon
        
        # Convert to bbox
        x_c, y_c, w, h = polygon_to_bbox(points, img_width, img_height)
        
        # Skip tiny boxes (noise)
        if w < 0.01 or h < 0.01:
            continue
        
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    return lines

def convert_split(split: str = "train"):
    """Convert all annotations in a split."""
    ann_dir = DACL_DIR / split / "ann"
    img_dir = DACL_DIR / split / "img"
    
    if not ann_dir.exists():
        print(f"Warning: {ann_dir} does not exist")
        return 0
    
    ann_files = list(ann_dir.glob("*.json"))
    converted = 0
    
    for ann_path in tqdm(ann_files, desc=f"Converting dacl10k/{split}"):
        # Convert annotation
        lines = convert_annotation(ann_path)
        
        if not lines:
            continue  # No relevant objects
        
        # Get image filename (remove .json suffix)
        img_name = ann_path.stem  # e.g., dacl10k_v2_train_0000.jpg
        
        # Find actual image
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        
        # Create unique filename
        new_img_name = f"dacl10k_{img_name}"
        new_lbl_name = f"dacl10k_{Path(img_name).stem}.txt"
        
        # Copy image
        target_img = TARGET_TRAIN_IMAGES / new_img_name
        if not target_img.exists():
            import shutil
            shutil.copy(img_path, target_img)
        
        # Write label
        target_lbl = TARGET_TRAIN_LABELS / new_lbl_name
        with open(target_lbl, 'w') as f:
            f.write('\n'.join(lines))
        
        converted += 1
    
    return converted

def main():
    print("=" * 60)
    print("dacl10k Supervisely → YOLO Converter")
    print("=" * 60)
    
    # Convert train and val splits
    total = 0
    for split in ["train", "val"]:
        count = convert_split(split)
        print(f"  Converted {count} images from {split}/")
        total += count
    
    print("\n" + "=" * 60)
    print(f"Total converted: {total}")
    print("=" * 60)
    
    # Count final dataset
    final_images = len(list(TARGET_TRAIN_IMAGES.glob("*")))
    final_labels = len(list(TARGET_TRAIN_LABELS.glob("*.txt")))
    print(f"\nFinal dataset:")
    print(f"  Images: {final_images}")
    print(f"  Labels: {final_labels}")

if __name__ == "__main__":
    main()
