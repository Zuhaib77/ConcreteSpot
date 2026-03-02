"""
Pure Quality Dataset Creator V3
- Crack: SDNET2018 (pure, high quality, 99% benchmark)
- Spalling, Corrosion, Exposed Rebar: CODEBRIM (explicit classes)
- Efflorescence: CODEBRIM (already 91.4%)
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
SDNET = Path("dataset/dataset/new_data_for_95plus/SDNET2018")
CODEBRIM = Path("dataset/dataset/new_data_for_95plus/codebrim/multirotulo")
OUTPUT = Path("dataset/dataset/pure_quality_v3")

# Class IDs
CLASS_IDS = {
    "crack": 0,
    "spalling": 1,
    "corrosion": 2,
    "exposed_rebar": 3,
    "efflorescence": 4
}

# CODEBRIM class mapping (from their Labels)
CODEBRIM_CLASSES = {
    "Crack": "crack",
    "Spallation": "spalling",
    "CorrosionStain": "corrosion",
    "ExposedBars": "exposed_rebar",
    "Efflorescence": "efflorescence"
}

def setup_dirs():
    """Create output directories"""
    for split in ["train", "valid"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

def process_sdnet_cracks(max_per_source=1000):
    """Convert SDNET2018 cracks to YOLO format"""
    print("\n" + "="*60)
    print("Processing SDNET2018 Cracks (Highest Quality)")
    print("="*60)
    
    count = 0
    all_images = []
    
    # Collect all cracked images from all sources
    for source in ["Decks", "Pavements", "Walls"]:
        cracked_dir = SDNET / source / "Cracked"
        if cracked_dir.exists():
            images = list(cracked_dir.glob("*.jpg"))
            print(f"  {source}: {len(images)} cracked images")
            all_images.extend([(img, source) for img in images[:max_per_source]])
    
    # Shuffle and split 80/20
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    valid_imgs = all_images[split_idx:]
    
    for split, images in [("train", train_imgs), ("valid", valid_imgs)]:
        for img_path, source in tqdm(images, desc=f"  SDNET {split}"):
            # Copy image
            dst_name = f"sdnet_{source.lower()}_{img_path.stem}"
            dst_img = OUTPUT / split / "images" / f"{dst_name}.jpg"
            shutil.copy(img_path, dst_img)
            
            # Create YOLO label (full image as crack - SDNET is patch-based)
            dst_label = OUTPUT / split / "labels" / f"{dst_name}.txt"
            with open(dst_label, 'w') as f:
                f.write(f"{CLASS_IDS['crack']} 0.5 0.5 0.9 0.9\n")
            count += 1
    
    print(f"  Total SDNET cracks: {count}")
    return count

def process_codebrim(target_class, max_images=1500):
    """Extract specific class from CODEBRIM"""
    print(f"\n  Processing CODEBRIM: {target_class}")
    
    count = 0
    all_pairs = []
    
    for split_name in ["train", "valid"]:
        labels_dir = CODEBRIM / split_name / "Labels"
        images_dir = CODEBRIM / split_name / "Images"
        
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file) as f:
                content = f.read()
            
            # Check if this class exists in the label
            codebrim_name = [k for k, v in CODEBRIM_CLASSES.items() if v == target_class]
            if not codebrim_name:
                continue
            codebrim_name = codebrim_name[0]
            
            if codebrim_name in content:
                # Find corresponding image
                img_name = label_file.stem
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                    img_path = images_dir / (img_name + ext)
                    if img_path.exists():
                        all_pairs.append((img_path, label_file, split_name))
                        break
    
    # Limit and shuffle
    random.shuffle(all_pairs)
    all_pairs = all_pairs[:max_images]
    
    # Split 80/20 if not already split
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    valid_pairs = all_pairs[split_idx:]
    
    for split, pairs in [("train", train_pairs), ("valid", valid_pairs)]:
        for img_path, label_path, _ in pairs:
            # Copy image
            dst_name = f"codebrim_{target_class}_{img_path.stem}"
            dst_img = OUTPUT / split / "images" / f"{dst_name}{img_path.suffix}"
            shutil.copy(img_path, dst_img)
            
            # Create YOLO label
            dst_label = OUTPUT / split / "labels" / f"{dst_name}.txt"
            with open(dst_label, 'w') as f:
                f.write(f"{CLASS_IDS[target_class]} 0.5 0.5 0.8 0.8\n")
            count += 1
    
    print(f"    Added: {count} images")
    return count

def create_data_yaml():
    """Create data.yaml"""
    yaml_content = f"""# Pure Quality Dataset V3
# Sources: SDNET2018 (cracks), CODEBRIM (all others)
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
    print("PURE QUALITY DATASET V3 CREATOR")
    print("="*60)
    
    setup_dirs()
    counts = {}
    
    # 1. SDNET2018 for cracks (limit to 2500 for balance)
    counts["crack"] = process_sdnet_cracks(max_per_source=1000)
    
    # 2. CODEBRIM for other classes
    print("\n" + "="*60)
    print("Processing CODEBRIM Classes")
    print("="*60)
    
    counts["spalling"] = process_codebrim("spalling", max_images=1500)
    counts["corrosion"] = process_codebrim("corrosion", max_images=1500)
    counts["exposed_rebar"] = process_codebrim("exposed_rebar", max_images=1000)
    counts["efflorescence"] = process_codebrim("efflorescence", max_images=800)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Pure Quality Dataset V3")
    print("="*60)
    total = 0
    for cls, cnt in counts.items():
        print(f"  {cls:20}: {cnt:5} images")
        total += cnt
    print(f"  {'TOTAL':20}: {total:5} images")
    
    # Create data.yaml
    create_data_yaml()
    print(f"\nCreated: {OUTPUT / 'data.yaml'}")
    print("\nDataset ready for training!")

if __name__ == "__main__":
    main()
