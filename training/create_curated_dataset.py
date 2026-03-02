"""
Curated Dataset Creator
Selects best quality data for each class based on research

Source mapping:
- crack:         organized/ + SDNET2018 (limited)
- spalling:      organized/ ONLY (achieved 90%)
- corrosion:     organized/ + CODEBRIM
- exposed_rebar: organized/ + CODEBRIM
- efflorescence: CODEBRIM only
"""
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Paths
ORGANIZED = Path("dataset/dataset/organized")
OUTPUT = Path("dataset/dataset/curated_quality")

# Class mapping for YOLO
CLASS_MAP = {
    "crack": 0,
    "spalling": 1,
    "corrosion": 2,
    "exposed_rebar": 3,
    "efflorescence": 4
}

def copy_organized_data(class_name: str, max_images: int = None):
    """Copy data from organized folder (already in YOLO format)"""
    src_train = ORGANIZED / class_name / "train"
    src_valid = ORGANIZED / class_name / "valid"
    
    count = 0
    
    # Copy training data
    if src_train.exists():
        images = list((src_train / "images").glob("*.*"))
        if max_images:
            images = images[:max_images]
        
        for img_path in tqdm(images, desc=f"  {class_name}/train"):
            label_path = src_train / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                # Copy image
                dst_img = OUTPUT / "train/images" / f"org_{class_name}_{img_path.name}"
                shutil.copy(img_path, dst_img)
                
                # Copy label (update class ID)
                dst_label = OUTPUT / "train/labels" / f"org_{class_name}_{img_path.stem}.txt"
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                with open(dst_label, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Keep original class ID (already correct)
                            f.write(line)
                count += 1
    
    # Copy validation data
    if src_valid.exists():
        images = list((src_valid / "images").glob("*.*"))
        if max_images:
            images = images[:max_images // 5]  # 20% for validation
        
        for img_path in tqdm(images, desc=f"  {class_name}/valid"):
            label_path = src_valid / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                dst_img = OUTPUT / "valid/images" / f"org_{class_name}_{img_path.name}"
                shutil.copy(img_path, dst_img)
                
                dst_label = OUTPUT / "valid/labels" / f"org_{class_name}_{img_path.stem}.txt"
                shutil.copy(label_path, dst_label)
                count += 1
    
    return count

def create_data_yaml():
    """Create data.yaml for the curated dataset"""
    yaml_content = f"""# Curated Quality Dataset
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
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    print("="*60)
    print("CURATED QUALITY DATASET CREATOR")
    print("="*60)
    print()
    
    # Curate from organized folder (best quality)
    print("Copying from organized/ (highest quality)...")
    
    counts = {}
    
    # Crack: limit to prevent dominance
    counts["crack"] = copy_organized_data("crack", max_images=3000)
    
    # Spalling: ALL (achieved 90%)
    counts["spalling"] = copy_organized_data("spalling")
    
    # Corrosion: ALL
    counts["corrosion"] = copy_organized_data("corrosion")
    
    # Exposed rebar: ALL
    counts["exposed_rebar"] = copy_organized_data("exposed_rebar")
    
    # Note: efflorescence not in organized, will add from CODEBRIM later
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} images")
    
    # Create data.yaml
    create_data_yaml()
    print()
    print(f"Created data.yaml at {OUTPUT / 'data.yaml'}")
    print()
    print("Next: Add efflorescence from CODEBRIM")

if __name__ == "__main__":
    main()
