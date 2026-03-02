"""
Master Dataset Unifier for ConcreteSpot v2.0
Unifies CODEBRIM, S2DS, SDNET2018, dacl10k into single YOLO dataset

Target Classes:
0: crack
1: spalling
2: corrosion
3: exposed_rebar
4: efflorescence
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Configuration
BASE = Path("dataset/dataset/new_data_for_95plus")
OUTPUT = Path("dataset/dataset/unified_95plus")

# Create output structure
for split in ["train", "valid", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

# Class mapping
CLASS_MAP = {
    "crack": 0, "Crack": 0, "cracks": 0,
    "spalling": 1, "Spallation": 1, "Spalling": 1,
    "corrosion": 2, "Corrosion": 2, "CorrosionStain": 2, "Rust": 2, "rust": 2,
    "exposed_rebar": 3, "ExposedBars": 3, "exposed rebars": 3, "ExposedRebars": 3, "rebar": 3,
    "efflorescence": 4, "Efflorescence": 4,
}

# S2DS color to class mapping (from README)
S2DS_COLORS = {
    (255, 0, 0): 0,      # crack - red
    (0, 255, 0): 1,      # spalling - green
    (0, 0, 255): 2,      # corrosion - blue
    (255, 255, 0): 4,    # efflorescence - yellow
}

counts = {
    "crack": 0, "spalling": 0, "corrosion": 0, 
    "exposed_rebar": 0, "efflorescence": 0
}


def mask_to_bboxes(mask_path, img_w, img_h):
    """Convert semantic mask to YOLO bboxes."""
    try:
        mask = Image.open(mask_path).convert("RGB")
        mask_arr = np.array(mask)
    except:
        return []
    
    bboxes = []
    
    for color, class_id in S2DS_COLORS.items():
        # Find pixels matching this color
        match = np.all(mask_arr == color, axis=2)
        if not match.any():
            continue
        
        # Get bounding box
        rows = np.any(match, axis=1)
        cols = np.any(match, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Convert to YOLO format
        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        
        if width > 0.01 and height > 0.01:
            bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return bboxes


def process_codebrim():
    """Process CODEBRIM dataset (already in YOLO-like format)."""
    print("\n" + "=" * 60)
    print("Processing CODEBRIM")
    print("=" * 60)
    
    codebrim = BASE / "codebrim" / "multirotulo"
    count = 0
    
    for in_split, out_split in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        img_dir = codebrim / in_split / "Images"
        lbl_dir = codebrim / in_split / "Labels"
        
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob("*.*"))
        print(f"  {in_split}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"  {in_split}"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            
            if not lbl_path.exists():
                continue
            
            # Read labels (text format: one class per line)
            with open(lbl_path) as f:
                labels = [line.strip() for line in f if line.strip()]
            
            # Convert to YOLO format
            yolo_lines = []
            for label in labels:
                if label in CLASS_MAP:
                    class_id = CLASS_MAP[label]
                    # For multi-label, use full image bbox
                    yolo_lines.append(f"{class_id} 0.5 0.5 0.9 0.9\n")
                    counts[list(CLASS_MAP.keys())[class_id]] = counts.get(list(CLASS_MAP.keys())[class_id], 0) + 1
            
            if yolo_lines:
                new_name = f"codebrim_{img_path.name}"
                shutil.copy2(img_path, OUTPUT / out_split / "images" / new_name)
                with open(OUTPUT / out_split / "labels" / f"codebrim_{img_path.stem}.txt", "w") as f:
                    f.writelines(yolo_lines)
                count += 1
    
    print(f"  Total: {count} images processed")
    return count


def process_s2ds():
    """Process S2DS dataset (semantic masks)."""
    print("\n" + "=" * 60)
    print("Processing S2DS")
    print("=" * 60)
    
    s2ds = BASE / "s2ds"
    count = 0
    
    for in_split, out_split in [("train", "train"), ("val", "valid"), ("test", "test")]:
        split_dir = s2ds / in_split
        
        if not split_dir.exists():
            continue
        
        # Get images (exclude masks)
        images = [f for f in split_dir.glob("*.png") if "_lab" not in f.stem]
        print(f"  {in_split}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"  {in_split}"):
            mask_path = split_dir / f"{img_path.stem}_lab.png"
            
            if not mask_path.exists():
                continue
            
            # Get image size
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except:
                continue
            
            # Convert mask to bboxes
            bboxes = mask_to_bboxes(mask_path, img_w, img_h)
            
            if bboxes:
                new_name = f"s2ds_{img_path.name}"
                shutil.copy2(img_path, OUTPUT / out_split / "images" / new_name)
                with open(OUTPUT / out_split / "labels" / f"s2ds_{img_path.stem}.txt", "w") as f:
                    f.writelines(bboxes)
                count += 1
    
    print(f"  Total: {count} images processed")
    return count


def process_sdnet():
    """Process SDNET2018 (crack classification with pseudo-labels)."""
    print("\n" + "=" * 60)
    print("Processing SDNET2018 (crack only)")
    print("=" * 60)
    
    sdnet = BASE / "SDNET2018"
    count = 0
    
    for surface in ["Decks", "Pavements", "Walls"]:
        cracked_dir = sdnet / surface / "Cracked"
        
        if not cracked_dir.exists():
            continue
        
        images = list(cracked_dir.glob("*.*"))[:1000]  # Limit per category
        print(f"  {surface}: {len(images)} cracked images (limited to 1000)")
        
        for i, img_path in enumerate(tqdm(images, desc=f"  {surface}")):
            # Use 80/10/10 split based on index
            if i % 10 < 8:
                split = "train"
            elif i % 10 < 9:
                split = "valid"
            else:
                split = "test"
            
            # Create pseudo-label (center object assumption)
            yolo_line = "0 0.5 0.5 0.7 0.7\n"  # crack class
            
            new_name = f"sdnet_{surface}_{img_path.name}"
            shutil.copy2(img_path, OUTPUT / split / "images" / new_name)
            with open(OUTPUT / split / "labels" / f"sdnet_{surface}_{img_path.stem}.txt", "w") as f:
                f.write(yolo_line)
            count += 1
    
    print(f"  Total: {count} images processed")
    return count


def process_dacl10k():
    """Process dacl10k dataset (labelme polygons)."""
    print("\n" + "=" * 60)
    print("Processing dacl10k")
    print("=" * 60)
    
    dacl = BASE / "dacl10k_v2" / "dacl10k_v2_devphase"
    count = 0
    
    for in_split, out_split in [("train", "train"), ("validation", "valid")]:
        ann_dir = dacl / "annotations" / in_split
        img_dir = dacl / "images" / in_split
        
        if not ann_dir.exists():
            continue
        
        annotations = list(ann_dir.glob("*.json"))
        print(f"  {in_split}: {len(annotations)} annotations")
        
        for ann_path in tqdm(annotations, desc=f"  {in_split}"):
            with open(ann_path, encoding='utf-8') as f:
                data = json.load(f)
            
            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)
            
            if img_w == 0 or img_h == 0:
                continue
            
            # Find matching image
            img_name = data.get("imageName", ann_path.stem + ".jpg")
            img_path = img_dir / img_name
            if not img_path.exists():
                img_path = img_dir / (ann_path.stem + ".jpg")
            if not img_path.exists():
                continue
            
            # Convert polygons to bboxes
            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                
                if label not in CLASS_MAP:
                    continue
                
                class_id = CLASS_MAP[label]
                points = shape.get("points", [])
                
                if len(points) < 3:
                    continue
                
                # Convert polygon to bbox
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                # Clamp and validate
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                if width > 0.02 and height > 0.02:
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            if yolo_lines:
                new_name = f"dacl_{img_path.name}"
                shutil.copy2(img_path, OUTPUT / out_split / "images" / new_name)
                with open(OUTPUT / out_split / "labels" / f"dacl_{img_path.stem}.txt", "w") as f:
                    f.writelines(yolo_lines)
                count += 1
    
    print(f"  Total: {count} images processed")
    return count


def add_existing_data():
    """Add existing organized data from previous cleanup."""
    print("\n" + "=" * 60)
    print("Adding Existing Organized Data")
    print("=" * 60)
    
    organized = Path("dataset/dataset/organized")
    count = 0
    
    class_to_id = {
        "crack": 0, "corrosion": 1, "spalling": 2, "exposed_rebar": 3
    }
    
    for damage_type, class_id in class_to_id.items():
        src_path = organized / damage_type
        if not src_path.exists():
            continue
        
        for split in ["train", "valid"]:
            out_split = split
            img_dir = src_path / split / "images"
            lbl_dir = src_path / split / "labels"
            
            if not img_dir.exists():
                continue
            
            images = list(img_dir.glob("*.*"))
            print(f"  {damage_type}/{split}: {len(images)} images")
            
            for img_path in images:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                
                if lbl_path.exists():
                    # Re-map class ID
                    with open(lbl_path) as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            parts[0] = str(class_id)
                            new_lines.append(" ".join(parts) + "\n")
                    
                    if new_lines:
                        new_name = f"org_{damage_type}_{img_path.name}"
                        shutil.copy2(img_path, OUTPUT / out_split / "images" / new_name)
                        with open(OUTPUT / out_split / "labels" / f"org_{damage_type}_{img_path.stem}.txt", "w") as f:
                            f.writelines(new_lines)
                        count += 1
    
    print(f"  Total: {count} images added")
    return count


def create_data_yaml():
    """Create data.yaml for training."""
    yaml_content = f"""# ConcreteSpot v2.0 Unified Dataset
# Target: 95%+ per-class accuracy

path: {OUTPUT.absolute()}
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  0: crack
  1: spalling
  2: corrosion
  3: exposed_rebar
  4: efflorescence
"""
    with open(OUTPUT / "data.yaml", "w") as f:
        f.write(yaml_content)
    print("\n  Created data.yaml")


def print_summary():
    """Print final dataset summary."""
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    
    for split in ["train", "valid", "test"]:
        img_count = len(list((OUTPUT / split / "images").glob("*.*")))
        lbl_count = len(list((OUTPUT / split / "labels").glob("*.txt")))
        print(f"  {split}: {img_count} images, {lbl_count} labels")
    
    # Count classes
    print("\n  Class distribution (train):")
    class_counts = [0, 0, 0, 0, 0]
    for lbl_file in (OUTPUT / "train" / "labels").glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < 5:
                            class_counts[class_id] += 1
                    except:
                        pass
    
    names = ["crack", "spalling", "corrosion", "exposed_rebar", "efflorescence"]
    for i, name in enumerate(names):
        print(f"    {name}: {class_counts[i]:,} annotations")


def main():
    print("=" * 60)
    print("CONCRETESPOT v2.0 DATASET UNIFIER")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT.absolute()}")
    print("\nTarget classes:")
    print("  0: crack")
    print("  1: spalling")
    print("  2: corrosion")
    print("  3: exposed_rebar")
    print("  4: efflorescence")
    
    # Process each dataset
    total = 0
    total += process_codebrim()
    total += process_s2ds()
    total += process_sdnet()
    total += process_dacl10k()
    total += add_existing_data()
    
    # Create config
    create_data_yaml()
    
    # Summary
    print_summary()
    
    print(f"\n  TOTAL: {total} images unified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
