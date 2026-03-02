"""
V4 Dataset Creator - Organized with Fixed Class IDs
Copies organized/ data with correct class mapping
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
ORGANIZED = Path("dataset/dataset/organized")
CODEBRIM = Path("dataset/dataset/new_data_for_95plus/codebrim/multirotulo")
OUTPUT = Path("dataset/dataset/organized_v4")

# Class mapping (folder name -> class ID)
CLASS_MAP = {
    "crack": 0,
    "spalling": 1,
    "corrosion": 2,
    "exposed_rebar": 3,
}

def setup_dirs():
    for split in ["train", "valid"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

def fix_class_id(label_path, target_class_id):
    """Read label and fix class ID to target"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = str(target_class_id)
                fixed_lines.append(' '.join(parts) + '\n')
        
        return fixed_lines
    except:
        return []

def process_organized():
    """Copy organized/ with fixed class IDs"""
    print("="*60)
    print("V4: Processing organized/ dataset")
    print("="*60)
    
    counts = {}
    
    for class_name, class_id in CLASS_MAP.items():
        class_dir = ORGANIZED / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_name} not found")
            continue
        
        counts[class_name] = 0
        
        for split in ["train", "valid"]:
            img_dir = class_dir / split / "images"
            lbl_dir = class_dir / split / "labels"
            
            if not img_dir.exists():
                continue
            
            images = list(img_dir.glob("*.*"))
            
            for img_path in tqdm(images, desc=f"  {class_name}/{split}"):
                # Find corresponding label
                label_name = img_path.stem + ".txt"
                label_path = lbl_dir / label_name
                
                if not label_path.exists():
                    continue
                
                # Fix class ID in label
                fixed_lines = fix_class_id(label_path, class_id)
                if not fixed_lines:
                    continue
                
                # Copy image
                dst_img = OUTPUT / split / "images" / f"org_{class_name}_{img_path.name}"
                shutil.copy(img_path, dst_img)
                
                # Write fixed label
                dst_lbl = OUTPUT / split / "labels" / f"org_{class_name}_{img_path.stem}.txt"
                with open(dst_lbl, 'w') as f:
                    f.writelines(fixed_lines)
                
                counts[class_name] += 1
    
    return counts

def add_efflorescence():
    """Add efflorescence from CODEBRIM"""
    print("\n  Adding efflorescence from CODEBRIM...")
    count = 0
    
    for split_name in ["train", "valid"]:
        labels_dir = CODEBRIM / split_name / "Labels"
        images_dir = CODEBRIM / split_name / "Images"
        
        if not labels_dir.exists():
            continue
        
        for label_file in tqdm(list(labels_dir.glob("*.txt")), desc=f"  efflorescence/{split_name}"):
            with open(label_file) as f:
                content = f.read()
            
            if "Efflorescence" not in content:
                continue
            
            # Find image
            img_name = label_file.stem
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                candidate = images_dir / (img_name + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            
            if not img_path:
                continue
            
            # Map to train/valid (80/20)
            out_split = "train" if split_name == "train" else "valid"
            
            # Copy image
            dst_img = OUTPUT / out_split / "images" / f"codebrim_effl_{img_path.name}"
            shutil.copy(img_path, dst_img)
            
            # Create label with class ID 4
            dst_lbl = OUTPUT / out_split / "labels" / f"codebrim_effl_{img_path.stem}.txt"
            with open(dst_lbl, 'w') as f:
                f.write("4 0.5 0.5 0.8 0.8\n")
            
            count += 1
    
    return count

def create_data_yaml():
    yaml_content = f"""# V4 Organized Dataset with Fixed Class IDs
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
    setup_dirs()
    counts = process_organized()
    counts["efflorescence"] = add_efflorescence()
    
    print("\n" + "="*60)
    print("V4 SUMMARY")
    print("="*60)
    total = 0
    for cls, cnt in counts.items():
        print(f"  {cls:20}: {cnt:5}")
        total += cnt
    print(f"  {'TOTAL':20}: {total:5}")
    
    create_data_yaml()
    print(f"\nCreated: {OUTPUT / 'data.yaml'}")

if __name__ == "__main__":
    main()
