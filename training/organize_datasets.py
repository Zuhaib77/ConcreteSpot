"""
Dataset Organizer - Separates datasets by damage type
Creates specialist folders for: crack, corrosion, spalling, exposed_rebar
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import Counter

BASE_DIR = Path("dataset/dataset")

# Output directories for each damage type
SPECIALIST_DIRS = {
    "crack": BASE_DIR / "crack_specialist",
    "corrosion": BASE_DIR / "corrosion_specialist", 
    "spalling": BASE_DIR / "spalling_specialist",
    "exposed_rebar": BASE_DIR / "rebar_specialist",
}

# Create all directories
for name, path in SPECIALIST_DIRS.items():
    (path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (path / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (path / "valid" / "labels").mkdir(parents=True, exist_ok=True)


def copy_yolo_dataset(src_dir: Path, target_type: str, prefix: str, class_filter: int = None):
    """
    Copy a YOLO dataset to the appropriate specialist folder.
    
    Args:
        src_dir: Source dataset directory (containing train/valid folders)
        target_type: One of 'crack', 'corrosion', 'spalling', 'exposed_rebar'
        prefix: Prefix to add to filenames to avoid collisions
        class_filter: If set, only keep labels with this class ID (remapped to 0)
    """
    output_dir = SPECIALIST_DIRS[target_type]
    total = 0
    
    for split in ["train", "valid"]:
        img_src = src_dir / split / "images"
        lbl_src = src_dir / split / "labels"
        
        if not img_src.exists():
            continue
        
        images = list(img_src.glob("*.*"))
        
        for img_path in tqdm(images, desc=f"  {prefix}/{split}"):
            label_name = img_path.stem + ".txt"
            label_path = lbl_src / label_name
            
            if not label_path.exists():
                continue
            
            # Read labels
            with open(label_path) as f:
                lines = f.readlines()
            
            # Filter if needed
            if class_filter is not None:
                filtered = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == class_filter:
                        # Remap to class 0
                        parts[0] = "0"
                        filtered.append(" ".join(parts) + "\n")
                if not filtered:
                    continue
                lines = filtered
            else:
                # Remap all classes to 0 (single-class specialist)
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"
                        new_lines.append(" ".join(parts) + "\n")
                lines = new_lines
            
            if not lines:
                continue
            
            # New names with prefix
            new_img = f"{prefix}_{img_path.name}"
            new_lbl = f"{prefix}_{label_name}"
            
            # Copy image
            shutil.copy2(img_path, output_dir / split / "images" / new_img)
            
            # Write label
            with open(output_dir / split / "labels" / new_lbl, "w") as f:
                f.writelines(lines)
            
            total += 1
    
    return total


def create_data_yaml(target_type: str, class_name: str):
    """Create data.yaml for a specialist dataset."""
    output_dir = SPECIALIST_DIRS[target_type]
    yaml_content = f"""# {class_name.title()} Specialist Dataset
path: {output_dir.absolute()}
train: train/images
val: valid/images

nc: 1
names: ['{class_name}']
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)


def count_dataset(path: Path) -> dict:
    """Count images in a dataset."""
    train = len(list((path / "train" / "images").glob("*.*"))) if (path / "train" / "images").exists() else 0
    valid = len(list((path / "valid" / "images").glob("*.*"))) if (path / "valid" / "images").exists() else 0
    return {"train": train, "valid": valid, "total": train + valid}


def main():
    print("=" * 60)
    print("Dataset Organizer - Separate by Damage Type")
    print("=" * 60)
    
    # =========================================================
    # CORROSION SPECIALIST
    # =========================================================
    print("\n" + "=" * 40)
    print("CORROSION SPECIALIST")
    print("=" * 40)
    
    # Corrosion YOLOv8 dataset (clean, single class)
    corr_src = BASE_DIR / "corrosion_new" / "corrosion_yolov8"
    if corr_src.exists():
        print("\n1. Adding Corrosion YOLOv8 dataset...")
        copy_yolo_dataset(corr_src, "corrosion", "corr_v8")
    
    create_data_yaml("corrosion", "corrosion")
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, path in SPECIALIST_DIRS.items():
        counts = count_dataset(path)
        if counts["total"] > 0:
            print(f"  {name:15} : {counts['train']:6} train + {counts['valid']:5} valid = {counts['total']:6} total")


if __name__ == "__main__":
    main()
