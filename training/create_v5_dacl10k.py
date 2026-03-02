"""
V5 Dataset Creator - dacl10k to YOLO format
Converts dacl10k polygon annotations to YOLO bounding boxes
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
DACL10K = Path("dataset/dataset/new_data_for_95plus/dacl10k_v2/dacl10k_v2_devphase")
OUTPUT = Path("dataset/dataset/dacl10k_v5")

# dacl10k class mapping to our 5 classes
# Based on dacl10k paper at https://arxiv.org/abs/2309.00460
CLASS_MAP = {
    # Damage classes (our targets)
    "Crack": 0,
    "ACrack": 0,              # Alligator crack -> crack
    "Weathering": 1,          # Weathering can cause spalling-like damage
    "Spalling": 1,
    "Corrosion": 2,
    "CorrosionStain": 2,
    "Rust": 2,
    "ExposedRebars": 3,
    "Exposed_Rebars": 3,
    "Rebar": 3,
    "Efflorescence": 4,
    # Classes to SKIP (not concrete damage)
    "Graffiti": None,
    "WConccrete": None,       # Washout concrete (not damage)
    "RottingWood": None,
    "Bearing": None,
    "JointTape": None,
    "Hollowareas": None,
    "Cavity": None,
    "Restformwork": None,
    "Wetspot": None,
    "Rockpocket": None,
    "Drainage": None,
    "PEquipment": None,
    "Protective_Equipment": None,
    "Joint": None,
    "Expansion_Joint": None,
}

def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to YOLO bbox format"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Convert to YOLO format (center x, center y, width, height) normalized
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    
    # Clamp values
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0.01, min(1, width))
    height = max(0.01, min(1, height))
    
    return x_center, y_center, width, height

def setup_dirs():
    for split in ["train", "valid"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

def process_annotations(split_name):
    """Process all annotations for a split"""
    out_split = "train" if split_name == "train" else "valid"
    ann_dir = DACL10K / "annotations" / split_name
    img_dir = DACL10K / "images" / split_name
    
    if not ann_dir.exists():
        print(f"  Warning: {ann_dir} not found")
        return {}
    
    counts = {i: 0 for i in range(5)}
    json_files = list(ann_dir.glob("*.json"))
    
    for json_file in tqdm(json_files, desc=f"  {split_name}"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        
        img_w = data.get("imageWidth", 1920)
        img_h = data.get("imageHeight", 1080)
        img_name = data.get("imageName", json_file.stem + ".jpg")
        
        # Find image file
        img_path = img_dir / img_name
        if not img_path.exists():
            # Try different extensions
            for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                candidate = img_dir / (json_file.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
        
        if not img_path.exists():
            continue
        
        # Process shapes
        yolo_lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            
            # Skip non-damage classes
            class_id = CLASS_MAP.get(label)
            if class_id is None:
                continue
            
            points = shape.get("points", [])
            if len(points) < 3:
                continue
            
            # Convert polygon to bbox
            x, y, w, h = polygon_to_bbox(points, img_w, img_h)
            yolo_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            counts[class_id] += 1
        
        # Only save if we have damage annotations
        if yolo_lines:
            # Copy image
            dst_img = OUTPUT / out_split / "images" / f"dacl_{img_path.name}"
            shutil.copy(img_path, dst_img)
            
            # Save labels
            dst_lbl = OUTPUT / out_split / "labels" / f"dacl_{img_path.stem}.txt"
            with open(dst_lbl, 'w') as f:
                f.writelines(yolo_lines)
    
    return counts

def create_data_yaml():
    yaml_content = f"""# V5 dacl10k Dataset
# Source: WACV 2024 dacl10k benchmark
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
    print("V5: DACL10K TO YOLO CONVERTER")
    print("="*60)
    
    setup_dirs()
    
    total_counts = {i: 0 for i in range(5)}
    
    for split in ["train", "validation"]:
        counts = process_annotations(split)
        for k, v in counts.items():
            total_counts[k] += v
    
    # Summary
    print("\n" + "="*60)
    print("V5 SUMMARY")
    print("="*60)
    class_names = ["crack", "spalling", "corrosion", "exposed_rebar", "efflorescence"]
    total = 0
    for i, name in enumerate(class_names):
        print(f"  {name:20}: {total_counts[i]:5} annotations")
        total += total_counts[i]
    print(f"  {'TOTAL':20}: {total:5} annotations")
    
    create_data_yaml()
    print(f"\nCreated: {OUTPUT / 'data.yaml'}")

if __name__ == "__main__":
    main()
