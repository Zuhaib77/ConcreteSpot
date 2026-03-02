"""
Extract Spalling and Exposed Rebar Data from Existing Datasets
Sources: HRCDS (labelme), dacl10k (Supervisely)
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

BASE = Path("dataset/dataset")
OUTPUT = BASE / "organized"

# Create output directories
for damage_type in ["spalling", "exposed_rebar"]:
    (OUTPUT / damage_type / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / damage_type / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT / damage_type / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / damage_type / "valid" / "labels").mkdir(parents=True, exist_ok=True)


def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to YOLO bbox format."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    x_center = max(0, min(1, (x_min + x_max) / 2 / img_w))
    y_center = max(0, min(1, (y_min + y_max) / 2 / img_h))
    width = max(0, min(1, (x_max - x_min) / img_w))
    height = max(0, min(1, (y_max - y_min) / img_h))
    
    return x_center, y_center, width, height


def extract_hrcds():
    """Extract from HRCDS (labelme format)."""
    print("=" * 60)
    print("Extracting from HRCDS")
    print("=" * 60)
    
    hrcds_base = BASE / "MDMCS A Benchmark Dataset for Multi-Damage Monitor" / "HRCDS"
    
    class_mapping = {
        "spalling": "spalling",
        "exposed rebar": "exposed_rebar"
    }
    
    counts = {"spalling": 0, "exposed_rebar": 0}
    
    for split_name, ann_dir, img_dir in [("train", "train_annotations", "train_image"),
                                          ("valid", "val_annotations", "val_image")]:
        ann_path = hrcds_base / ann_dir
        img_path = hrcds_base / img_dir
        
        if not ann_path.exists():
            continue
        
        for ann_file in tqdm(list(ann_path.glob("*.json")), desc=f"  {ann_dir}"):
            with open(ann_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # Find matching image
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
            
            # Process each damage type
            for label_name, output_type in class_mapping.items():
                shapes = [s for s in data.get('shapes', []) 
                         if s.get('label', '').lower() == label_name]
                
                if not shapes:
                    continue
                
                # Convert to YOLO
                yolo_lines = []
                for shape in shapes:
                    points = shape.get('points', [])
                    if len(points) < 3:
                        continue
                    x_c, y_c, w, h = polygon_to_bbox(points, img_w, img_h)
                    if w > 0.01 and h > 0.01:
                        yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
                if not yolo_lines:
                    continue
                
                # Save
                out_dir = OUTPUT / output_type / split_name
                new_img = f"hrcds_{img_name}"
                new_lbl = f"hrcds_{ann_file.stem}.txt"
                
                shutil.copy2(img_file, out_dir / "images" / new_img)
                with open(out_dir / "labels" / new_lbl, "w") as f:
                    f.writelines(yolo_lines)
                
                counts[output_type] += 1
    
    print(f"  Spalling: {counts['spalling']} images")
    print(f"  Exposed Rebar: {counts['exposed_rebar']} images")
    return counts


def extract_dacl10k():
    """Extract from dacl10k (Supervisely format)."""
    print("\n" + "=" * 60)
    print("Extracting from dacl10k")
    print("=" * 60)
    
    dacl_base = BASE / "dacl10k_supervisely"
    
    class_mapping = {
        "spalling": "spalling",
        "exposed rebars": "exposed_rebar"
    }
    
    counts = {"spalling": 0, "exposed_rebar": 0}
    
    for split_name, in_split in [("train", "train"), ("valid", "val")]:
        img_path = dacl_base / in_split / "img"
        ann_path = dacl_base / in_split / "ann"
        
        if not ann_path.exists():
            continue
        
        for ann_file in tqdm(list(ann_path.glob("*.json")), desc=f"  {in_split}"):
            with open(ann_file, encoding='utf-8') as f:
                data = json.load(f)
            
            img_size = data.get('size', {})
            img_w = img_size.get('width', 0)
            img_h = img_size.get('height', 0)
            
            if img_w == 0 or img_h == 0:
                continue
            
            # Find image
            img_name = ann_file.stem
            img_file = None
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = img_path / (img_name + ext)
                if candidate.exists():
                    img_file = candidate
                    break
            
            if not img_file:
                continue
            
            # Process each damage type
            for class_title, output_type in class_mapping.items():
                objs = [obj for obj in data.get('objects', [])
                       if obj.get('classTitle', '').lower() == class_title]
                
                if not objs:
                    continue
                
                yolo_lines = []
                for obj in objs:
                    geom_type = obj.get('geometryType', '')
                    
                    if geom_type == 'polygon':
                        exterior = obj.get('points', {}).get('exterior', [])
                        if len(exterior) < 3:
                            continue
                        x_c, y_c, w, h = polygon_to_bbox(exterior, img_w, img_h)
                    elif geom_type == 'rectangle':
                        points = obj.get('points', {}).get('exterior', [])
                        if len(points) >= 2:
                            x_min, y_min = points[0]
                            x_max, y_max = points[1]
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
                
                out_dir = OUTPUT / output_type / split_name
                new_img = f"dacl_{img_file.name}"
                new_lbl = f"dacl_{img_file.stem}.txt"
                
                try:
                    shutil.copy2(img_file, out_dir / "images" / new_img)
                    with open(out_dir / "labels" / new_lbl, "w") as f:
                        f.writelines(yolo_lines)
                    counts[output_type] += 1
                except:
                    pass
    
    print(f"  Spalling: {counts['spalling']} images")
    print(f"  Exposed Rebar: {counts['exposed_rebar']} images")
    return counts


def create_data_yamls():
    """Create data.yaml for each organized dataset."""
    for damage_type in ["crack", "corrosion", "spalling", "exposed_rebar"]:
        out_path = OUTPUT / damage_type
        train_count = len(list((out_path / "train" / "images").glob("*.*"))) if (out_path / "train" / "images").exists() else 0
        valid_count = len(list((out_path / "valid" / "images").glob("*.*"))) if (out_path / "valid" / "images").exists() else 0
        
        if train_count > 0:
            yaml_content = f"""# {damage_type.replace('_', ' ').title()} Dataset
path: {out_path.absolute()}
train: train/images
val: valid/images

nc: 1
names: ['{damage_type.replace('_', ' ')}']
"""
            with open(out_path / "data.yaml", "w") as f:
                f.write(yaml_content)
            print(f"  Created: {damage_type}/data.yaml ({train_count} train, {valid_count} valid)")


def main():
    print("=" * 60)
    print("SPALLING & EXPOSED REBAR EXTRACTOR")
    print("=" * 60)
    
    hrcds_counts = extract_hrcds()
    dacl_counts = extract_dacl10k()
    
    print("\n" + "=" * 60)
    print("CREATING DATA YAMLS")
    print("=" * 60)
    create_data_yamls()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_spalling = hrcds_counts['spalling'] + dacl_counts['spalling']
    total_rebar = hrcds_counts['exposed_rebar'] + dacl_counts['exposed_rebar']
    
    print(f"  Spalling: {total_spalling} images extracted")
    print(f"  Exposed Rebar: {total_rebar} images extracted")


if __name__ == "__main__":
    main()
