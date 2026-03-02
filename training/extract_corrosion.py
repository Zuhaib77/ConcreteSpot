"""
Extract Corrosion Data from Existing Datasets
Extracts from: HRCDS (labelme), dacl10k (Supervisely)
Outputs to: corrosion_specialist folder in YOLO format
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

BASE_DIR = Path("dataset/dataset")
OUTPUT_DIR = BASE_DIR / "corrosion_specialist"

# Ensure output dirs exist
(OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "valid" / "images").mkdir(parents=True, exist_ok=True)  
(OUTPUT_DIR / "valid" / "labels").mkdir(parents=True, exist_ok=True)


def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to YOLO bbox format."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # YOLO format: x_center, y_center, width, height (normalized)
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    
    # Clamp values
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def extract_hrcds_corrosion():
    """Extract corrosion from HRCDS (labelme format)."""
    print("\n" + "=" * 50)
    print("Extracting HRCDS Corrosion Data")
    print("=" * 50)
    
    hrcds_base = BASE_DIR / "MDMCS A Benchmark Dataset for Multi-Damage Monitor" / "HRCDS"
    splits = [("train", "train_annotations", "train_image"),
              ("val", "val_annotations", "val_image")]
    
    total = 0
    
    for out_split, ann_dir, img_dir in splits:
        ann_path = hrcds_base / ann_dir
        img_path = hrcds_base / img_dir
        
        if not ann_path.exists():
            continue
        
        annotations = list(ann_path.glob("*.json"))
        print(f"\nProcessing {len(annotations)} annotations from {ann_dir}...")
        
        for ann_file in tqdm(annotations, desc=f"  {ann_dir}"):
            with open(ann_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter corrosion labels only
            corrosion_shapes = [s for s in data.get('shapes', []) 
                               if s.get('label', '').lower() == 'corrosion']
            
            if not corrosion_shapes:
                continue
            
            # Find image
            img_name = ann_file.stem + ".jpg"
            img_file = img_path / img_name
            if not img_file.exists():
                img_name = ann_file.stem + ".png"
                img_file = img_path / img_name
            if not img_file.exists():
                continue
            
            # Get image dimensions
            try:
                with Image.open(img_file) as img:
                    img_w, img_h = img.size
            except:
                continue
            
            # Convert to YOLO format
            yolo_lines = []
            for shape in corrosion_shapes:
                points = shape.get('points', [])
                if len(points) < 3:
                    continue
                
                x_c, y_c, w, h = polygon_to_bbox(points, img_w, img_h)
                if w > 0.01 and h > 0.01:  # Skip tiny boxes
                    yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            if not yolo_lines:
                continue
            
            # Output paths
            out_split_name = "train" if out_split == "train" else "valid"
            new_img = f"hrcds_{img_name}"
            new_lbl = f"hrcds_{ann_file.stem}.txt"
            
            # Copy image
            shutil.copy2(img_file, OUTPUT_DIR / out_split_name / "images" / new_img)
            
            # Write label
            with open(OUTPUT_DIR / out_split_name / "labels" / new_lbl, "w") as f:
                f.writelines(yolo_lines)
            
            total += 1
    
    print(f"\n  Extracted {total} corrosion images from HRCDS")
    return total


def extract_dacl10k_corrosion():
    """Extract rust and corrosion from dacl10k (Supervisely format)."""
    print("\n" + "=" * 50)
    print("Extracting dacl10k Rust/Corrosion Data")
    print("=" * 50)
    
    dacl_base = BASE_DIR / "dacl10k_supervisely"
    
    # Target classes for corrosion
    target_classes = ['rust', 'washouts/concrete corrosion']
    
    splits = [("train", "train"), ("val", "val")]
    total = 0
    
    for out_split, in_split in splits:
        img_path = dacl_base / in_split / "img"
        ann_path = dacl_base / in_split / "ann"
        
        if not ann_path.exists():
            continue
        
        annotations = list(ann_path.glob("*.json"))
        print(f"\nProcessing {len(annotations)} annotations from {in_split}...")
        
        for ann_file in tqdm(annotations, desc=f"  {in_split}"):
            with open(ann_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image size
            img_size = data.get('size', {})
            img_w = img_size.get('width', 0)
            img_h = img_size.get('height', 0)
            
            if img_w == 0 or img_h == 0:
                continue
            
            # Filter corrosion/rust objects
            corrosion_objs = [obj for obj in data.get('objects', [])
                            if obj.get('classTitle', '').lower() in target_classes]
            
            if not corrosion_objs:
                continue
            
            # Find image
            img_name = ann_file.stem.replace('.jpg', '').replace('.png', '')
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = img_path / (img_name + ext)
                if img_file.exists():
                    break
            else:
                # Try exact match
                img_file = img_path / ann_file.stem
                if not img_file.exists():
                    continue
            
            # Convert to YOLO format
            yolo_lines = []
            for obj in corrosion_objs:
                geom_type = obj.get('geometryType', '')
                
                if geom_type == 'polygon':
                    exterior = obj.get('points', {}).get('exterior', [])
                    if len(exterior) < 3:
                        continue
                    x_c, y_c, w, h = polygon_to_bbox(exterior, img_w, img_h)
                elif geom_type == 'rectangle':
                    points = obj.get('points', {})
                    exterior = points.get('exterior', [])
                    if len(exterior) >= 2:
                        x_min, y_min = exterior[0]
                        x_max, y_max = exterior[1]
                        x_c = (x_min + x_max) / 2 / img_w
                        y_c = (y_min + y_max) / 2 / img_h
                        w = (x_max - x_min) / img_w
                        h = (y_max - y_min) / img_h
                    else:
                        continue
                else:
                    continue
                
                if w > 0.01 and h > 0.01:
                    yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            if not yolo_lines:
                continue
            
            # Output
            out_split_name = "train" if out_split == "train" else "valid"
            new_img = f"dacl_{img_file.name}"
            new_lbl = f"dacl_{img_file.stem}.txt"
            
            try:
                shutil.copy2(img_file, OUTPUT_DIR / out_split_name / "images" / new_img)
                with open(OUTPUT_DIR / out_split_name / "labels" / new_lbl, "w") as f:
                    f.writelines(yolo_lines)
                total += 1
            except Exception as e:
                pass
    
    print(f"\n  Extracted {total} corrosion images from dacl10k")
    return total


def update_data_yaml():
    """Update data.yaml for corrosion specialist."""
    yaml_content = f"""# Corrosion Specialist Dataset
path: {OUTPUT_DIR.absolute()}
train: train/images
val: valid/images

nc: 1
names: ['corrosion']
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)


def main():
    print("=" * 60)
    print("Corrosion Data Extractor")
    print("=" * 60)
    
    # Count existing
    existing_train = len(list((OUTPUT_DIR / "train" / "images").glob("*.*")))
    existing_valid = len(list((OUTPUT_DIR / "valid" / "images").glob("*.*")))
    print(f"\nExisting corrosion data: {existing_train} train + {existing_valid} valid")
    
    # Extract from HRCDS
    hrcds_count = extract_hrcds_corrosion()
    
    # Extract from dacl10k
    dacl_count = extract_dacl10k_corrosion()
    
    # Update data.yaml
    update_data_yaml()
    
    # Final count
    final_train = len(list((OUTPUT_DIR / "train" / "images").glob("*.*")))
    final_valid = len(list((OUTPUT_DIR / "valid" / "images").glob("*.*")))
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Before: {existing_train} train + {existing_valid} valid")
    print(f"  Added: HRCDS={hrcds_count}, dacl10k={dacl_count}")
    print(f"  Final: {final_train} train + {final_valid} valid = {final_train + final_valid} total")


if __name__ == "__main__":
    main()
