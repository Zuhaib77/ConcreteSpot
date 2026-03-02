"""
Create Best Ensemble Dataset
Combines the best data per class based on highest accuracy achieved:
- crack: V3 pure_quality (SDNET2018) - 99.5%
- efflorescence: V4 organized_v4 - 99.1%
- spalling: V4 organized_v4 - 57.7% (best available)
- corrosion: V4 organized_v4 - 42.1% (best available)  
- exposed_rebar: V4 organized_v4 - 33.5% (V1 had 69.6% but different format)
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Output directory
OUTPUT = Path("dataset/dataset/best_ensemble")

# Source mapping per class
CLASS_SOURCES = {
    # crack: V3 pure_quality (SDNET2018 data)
    "crack": {
        "source": Path("dataset/dataset/pure_quality_v3"),
        "prefix": "sdnet_",  # SDNET files start with this
        "class_id": 0,
        "note": "99.5% mAP from V3"
    },
    # efflorescence: V4 organized_v4
    "efflorescence": {
        "source": Path("dataset/dataset/organized_v4"),
        "prefix": "efflorescence",
        "class_id": 4,
        "note": "99.1% mAP from V4"
    },
    # spalling: V4 organized_v4 (best available)
    "spalling": {
        "source": Path("dataset/dataset/organized_v4"),
        "prefix": "spalling",
        "class_id": 1,
        "note": "57.7% mAP from V4 (original had 90%)"
    },
    # corrosion: V4 organized_v4 (needs improvement)
    "corrosion": {
        "source": Path("dataset/dataset/organized_v4"),
        "prefix": "corrosion",
        "class_id": 2,
        "note": "42.1% mAP from V4 (needs improvement)"
    },
    # exposed_rebar: V4 organized_v4
    "exposed_rebar": {
        "source": Path("dataset/dataset/organized_v4"),
        "prefix": "exposed_rebar",
        "class_id": 3,
        "note": "33.5% mAP from V4 (V1 had 69.6%)"
    },
}

def setup_dirs():
    for split in ["train", "valid"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT}")

def copy_class_data(class_name, config):
    """Copy data for a specific class from its best source"""
    source = config["source"]
    prefix = config["prefix"]
    class_id = config["class_id"]
    
    print(f"\n  Processing {class_name} from {source.name}...")
    print(f"    Note: {config['note']}")
    
    counts = {"train": 0, "valid": 0}
    
    for split in ["train", "valid"]:
        src_img_dir = source / split / "images"
        src_lbl_dir = source / split / "labels"
        
        if not src_img_dir.exists():
            print(f"    Warning: {src_img_dir} not found")
            continue
        
        # Find all images for this class
        for img_path in src_img_dir.iterdir():
            if not img_path.is_file():
                continue
                
            # Check if this image belongs to this class
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            
            # Read label and check if it contains this class
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            # Filter lines for this class only
            class_lines = [l for l in lines if l.strip().startswith(str(class_id) + " ")]
            
            if not class_lines:
                continue
            
            # Copy image
            dst_img = OUTPUT / split / "images" / f"best_{class_name}_{img_path.name}"
            shutil.copy(img_path, dst_img)
            
            # Copy label (only lines for this class, but keep original class ID)
            dst_lbl = OUTPUT / split / "labels" / f"best_{class_name}_{img_path.stem}.txt"
            with open(dst_lbl, 'w') as f:
                f.writelines(class_lines)
            
            counts[split] += 1
    
    print(f"    Copied: {counts['train']} train, {counts['valid']} valid")
    return counts

def create_data_yaml():
    yaml_content = f"""# Best Ensemble Dataset
# Combines best data per class from highest-performing training runs
path: {OUTPUT.absolute()}
train: train/images
val: valid/images

names:
  0: crack
  1: spalling
  2: corrosion
  3: exposed_rebar
  4: efflorescence

nc: 5

# Source per class:
# crack: V3 pure_quality (SDNET2018) - 99.5%
# efflorescence: V4 organized_v4 - 99.1%
# spalling: V4 organized_v4 - 57.7% (original 90%)
# corrosion: V4 organized_v4 - 42.1% (needs improvement)
# exposed_rebar: V4 organized_v4 - 33.5% (V1 69.6%)
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated: {OUTPUT / 'data.yaml'}")

def main():
    print("="*60)
    print("BEST ENSEMBLE DATASET CREATOR")
    print("="*60)
    
    setup_dirs()
    
    total_counts = {"train": 0, "valid": 0}
    
    for class_name, config in CLASS_SOURCES.items():
        counts = copy_class_data(class_name, config)
        total_counts["train"] += counts["train"]
        total_counts["valid"] += counts["valid"]
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total train images: {total_counts['train']}")
    print(f"  Total valid images: {total_counts['valid']}")
    print(f"  Total: {total_counts['train'] + total_counts['valid']}")
    
    create_data_yaml()
    print("\nBest ensemble dataset ready!")

if __name__ == "__main__":
    main()
