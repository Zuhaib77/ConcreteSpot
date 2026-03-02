"""
Dataset Merger Script
Merges multiple YOLO-format datasets into a unified training set.
Handles class remapping to unified classes:
    0: crack, 1: spalling, 2: corrosion, 3: exposed_rebar
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Base paths
BASE_DIR = Path("dataset/dataset")
TARGET_TRAIN_IMAGES = BASE_DIR / "images/train"
TARGET_TRAIN_LABELS = BASE_DIR / "labels/train"

# Class mapping for each dataset
# Format: {original_class_id: new_class_id}
CLASS_MAPPINGS = {
    "roboflow_crack_detection": {0: 0},  # '0' -> crack
    "roboflow_cracks": {0: 0},  # 'cracks' -> crack
    "roboflow_concrete_damage": {0: 0, 1: 1},  # crack, spall -> crack, spalling
}

def remap_labels(label_path: Path, class_mapping: dict) -> list:
    """Read label file and remap class indices."""
    if not label_path.exists():
        return []
    
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class = int(parts[0])
                if old_class in class_mapping:
                    parts[0] = str(class_mapping[old_class])
                    lines.append(' '.join(parts))
    return lines

def merge_dataset(dataset_name: str, class_mapping: dict, split: str = "train"):
    """Merge a single dataset into the main training set."""
    dataset_path = BASE_DIR / dataset_name
    
    # Check for Roboflow structure (train/images, train/labels)
    img_src = dataset_path / split / "images"
    lbl_src = dataset_path / split / "labels"
    
    if not img_src.exists():
        print(f"Warning: {img_src} does not exist, skipping...")
        return 0
    
    images = list(img_src.glob("*"))
    merged = 0
    
    for img_path in tqdm(images, desc=f"Merging {dataset_name}/{split}"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        # Create unique filename with dataset prefix
        new_img_name = f"{dataset_name}_{img_path.name}"
        new_lbl_name = f"{dataset_name}_{img_path.stem}.txt"
        
        # Copy image
        target_img = TARGET_TRAIN_IMAGES / new_img_name
        if not target_img.exists():
            shutil.copy(img_path, target_img)
        
        # Process and copy label
        label_path = lbl_src / f"{img_path.stem}.txt"
        remapped_lines = remap_labels(label_path, class_mapping)
        
        if remapped_lines:
            target_lbl = TARGET_TRAIN_LABELS / new_lbl_name
            with open(target_lbl, 'w') as f:
                f.write('\n'.join(remapped_lines))
            merged += 1
    
    return merged

def main():
    print("=" * 60)
    print("YOLO Dataset Merger")
    print("=" * 60)
    
    total_merged = 0
    
    for dataset_name, class_mapping in CLASS_MAPPINGS.items():
        print(f"\nProcessing: {dataset_name}")
        
        # Merge train split
        count = merge_dataset(dataset_name, class_mapping, "train")
        total_merged += count
        print(f"  Merged {count} images from train/")
        
        # Also merge valid split as additional training data
        count = merge_dataset(dataset_name, class_mapping, "valid")
        total_merged += count
        if count > 0:
            print(f"  Merged {count} images from valid/")
    
    print("\n" + "=" * 60)
    print(f"Total images merged: {total_merged}")
    print("=" * 60)
    
    # Count final dataset
    final_images = len(list(TARGET_TRAIN_IMAGES.glob("*")))
    final_labels = len(list(TARGET_TRAIN_LABELS.glob("*.txt")))
    print(f"\nFinal dataset:")
    print(f"  Images: {final_images}")
    print(f"  Labels: {final_labels}")

if __name__ == "__main__":
    main()
