"""
Merge New Crack Datasets for Specialist Model Training
Combines: concrete_crack_v2, roads_bridges, a_crack_bridge (class 0 only)
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path("dataset/dataset")
CRACK_NEW = BASE_DIR / "crack_new"
OUTPUT_DIR = BASE_DIR / "crack_specialist"

# Create output directories
(OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "valid" / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "valid" / "labels").mkdir(parents=True, exist_ok=True)


def merge_yolo_dataset(src_dir: Path, split: str, prefix: str):
    """Merge a YOLO dataset split into the crack_specialist output."""
    img_src = src_dir / split / "images"
    lbl_src = src_dir / split / "labels"
    
    if not img_src.exists():
        print(f"  Skipping {src_dir.name}/{split} - not found")
        return 0
    
    images = list(img_src.glob("*.*"))
    count = 0
    
    for img_path in tqdm(images, desc=f"  {prefix}/{split}"):
        # Find corresponding label
        label_name = img_path.stem + ".txt"
        label_path = lbl_src / label_name
        
        if not label_path.exists():
            continue
        
        # New names with prefix to avoid collisions
        new_name = f"{prefix}_{img_path.name}"
        new_label = f"{prefix}_{label_name}"
        
        # Determine output split (train/valid)
        out_split = "train" if split in ["train", "training"] else "valid"
        
        # Copy image
        shutil.copy2(img_path, OUTPUT_DIR / out_split / "images" / new_name)
        
        # Copy label (all class 0 for crack)
        shutil.copy2(label_path, OUTPUT_DIR / out_split / "labels" / new_label)
        
        count += 1
    
    return count


def merge_bridge_dataset_filtered(src_dir: Path, prefix: str):
    """Merge bridge dataset, filtering only class 0 (cracks)."""
    img_src = src_dir / "images"
    lbl_src = src_dir / "labels"
    
    if not img_src.exists():
        print(f"  Skipping bridge dataset - not found")
        return 0
    
    images = list(img_src.glob("*.*"))
    count = 0
    
    for img_path in tqdm(images, desc=f"  {prefix}"):
        label_name = img_path.stem + ".txt"
        label_path = lbl_src / label_name
        
        if not label_path.exists():
            continue
        
        # Read and filter only class 0 labels
        with open(label_path) as f:
            lines = f.readlines()
        
        filtered_lines = [line for line in lines if line.strip().startswith("0 ")]
        
        if not filtered_lines:
            continue  # Skip images with no crack labels
        
        # New names
        new_name = f"{prefix}_{img_path.name}"
        new_label = f"{prefix}_{label_name}"
        
        # Copy image
        shutil.copy2(img_path, OUTPUT_DIR / "train" / "images" / new_name)
        
        # Write filtered labels
        with open(OUTPUT_DIR / "train" / "labels" / new_label, "w") as f:
            f.writelines(filtered_lines)
        
        count += 1
    
    return count


def create_data_yaml():
    """Create data.yaml for the crack specialist dataset."""
    yaml_content = f"""# Crack Specialist Dataset
path: {OUTPUT_DIR.absolute()}
train: train/images
val: valid/images

nc: 1
names: ['crack']
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"\nCreated: {OUTPUT_DIR / 'data.yaml'}")


def main():
    print("=" * 60)
    print("Crack Specialist Dataset Merger")
    print("=" * 60)
    
    total = 0
    
    # 1. Merge concrete_crack_v2 (clean, class 0)
    print("\n1. Merging concrete_crack_v2...")
    total += merge_yolo_dataset(CRACK_NEW / "concrete_crack_v2", "train", "cc")
    total += merge_yolo_dataset(CRACK_NEW / "concrete_crack_v2", "valid", "cc")
    
    # 2. Merge roads_bridges (clean, class 0)
    print("\n2. Merging roads_bridges...")
    total += merge_yolo_dataset(CRACK_NEW / "roads_bridges", "train", "rb")
    total += merge_yolo_dataset(CRACK_NEW / "roads_bridges", "valid", "rb")
    
    # 3. Merge bridge dataset (filtered to class 0 only)
    print("\n3. Merging a_crack_bridge (class 0 only)...")
    bridge_path = CRACK_NEW / "a_crack_dataset" / "Bridge defect detect for YOLO" / "Bridge defect detect for YOLO" / "datasets"
    total += merge_bridge_dataset_filtered(bridge_path, "bridge")
    
    # Create data.yaml
    create_data_yaml()
    
    # Count final
    train_count = len(list((OUTPUT_DIR / "train" / "images").glob("*.*")))
    valid_count = len(list((OUTPUT_DIR / "valid" / "images").glob("*.*")))
    
    print("\n" + "=" * 60)
    print(f"COMPLETE: {train_count} train + {valid_count} valid = {train_count + valid_count} total")
    print("=" * 60)


if __name__ == "__main__":
    main()
