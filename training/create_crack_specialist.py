"""
Create Crack Specialist Dataset
- Train: pure_quality_v3 (SDNET2018 - 99.5% accuracy)
- Valid: organized_v4 (test generalization)
"""
import os
import shutil
from pathlib import Path

OUTPUT = Path("dataset/specialists/crack_v2")

def extract_crack_data(source, split, output_split):
    """Extract crack data from source to output"""
    src_img = source / split / "images"
    src_lbl = source / split / "labels"
    
    dst_img = OUTPUT / output_split / "images"
    dst_lbl = OUTPUT / output_split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in src_img.iterdir():
        if not img_path.is_file():
            continue
        
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        # Class 0 = crack
        crack_lines = [l for l in lines if l.strip().startswith("0 ")]
        
        if not crack_lines:
            continue
        
        # Copy image
        shutil.copy(img_path, dst_img / img_path.name)
        
        # Write label (keep class 0)
        with open(dst_lbl / (img_path.stem + ".txt"), 'w') as f:
            f.writelines(crack_lines)
        
        count += 1
    
    return count

def main():
    print("="*60)
    print("CRACK SPECIALIST DATASET")
    print("Train: pure_quality_v3 (99.5% accuracy source)")
    print("Valid: organized_v4 (test generalization)")
    print("="*60)
    
    # Train from V3 (both train and valid)
    v3 = Path("dataset/dataset/pure_quality_v3")
    train_count = 0
    train_count += extract_crack_data(v3, "train", "train")
    train_count += extract_crack_data(v3, "valid", "train")  # Use V3 valid as extra train
    print(f"\nTrain images (from V3): {train_count}")
    
    # Valid from organized_v4
    v4 = Path("dataset/dataset/organized_v4")
    valid_count = extract_crack_data(v4, "valid", "valid")
    print(f"Valid images (from V4): {valid_count}")
    
    # Create data.yaml
    yaml = f"""# Crack Specialist Dataset
# Train: pure_quality_v3 (SDNET2018)
# Valid: organized_v4
path: {OUTPUT.absolute()}
train: train/images
val: valid/images

names:
  0: crack

nc: 1
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml)
    
    print(f"\nDataset ready: {OUTPUT}")
    print(f"data.yaml created")

if __name__ == "__main__":
    main()
