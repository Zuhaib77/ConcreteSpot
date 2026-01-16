#!/usr/bin/env python3
"""
Convert HRCDS LabelMe format to YOLO format
===========================================
This script converts the HRCDS dataset from LabelMe JSON polygon format
to YOLO bounding box format for training.

Classes:
    0: crack
    1: spalling
    2: corrosion
    3: exposed_rebar
"""

import json
import shutil
from pathlib import Path
from typing import Tuple


CLASS_MAP = {
    "crack": 0,
    "spalling": 1,
    "corrosion": 2,
    "exposed rebar": 3,
    "exposed_rebar": 3,
}


def polygon_to_bbox(points: list) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def convert_annotation(json_path: Path, output_path: Path, img_width: int, img_height: int):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lines = []
    
    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower()
        points = shape.get('points', [])
        
        if label not in CLASS_MAP:
            continue
        
        if len(points) < 3:
            continue
        
        class_id = CLASS_MAP[label]
        x_center, y_center, width, height = polygon_to_bbox(points)
        
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return len(lines)


def get_image_size(json_path: Path) -> Tuple[int, int]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data.get('imageWidth', 640), data.get('imageHeight', 480)


def convert_dataset(
    source_dir: Path,
    output_dir: Path
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    for split in ['train', 'val', 'test']:
        img_src = source_dir / f"{split}_image"
        ann_src = source_dir / f"{split}_annotations"
        
        if not img_src.exists():
            print(f"Skipping {split}: {img_src} not found")
            continue
        
        img_dst = output_dir / "images" / split
        lbl_dst = output_dir / "labels" / split
        
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split}...")
        
        img_files = list(img_src.glob("*.*"))
        converted = 0
        skipped = 0
        
        for img_file in img_files:
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            ann_file = ann_src / f"{img_file.stem}.json"
            
            if not ann_file.exists():
                skipped += 1
                continue
            
            shutil.copy(img_file, img_dst / img_file.name)
            
            img_width, img_height = get_image_size(ann_file)
            
            label_file = lbl_dst / f"{img_file.stem}.txt"
            num_objects = convert_annotation(ann_file, label_file, img_width, img_height)
            
            converted += 1
        
        print(f"  Converted: {converted} images")
        print(f"  Skipped: {skipped} (no annotation)")
    
    data_yaml = output_dir / "data.yaml"
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: crack
  1: spalling
  2: corrosion
  3: exposed_rebar
"""
    data_yaml.write_text(yaml_content)
    print(f"\nCreated {data_yaml}")
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert HRCDS to YOLO format")
    parser.add_argument("--source", type=str, required=True, help="Path to HRCDS folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder for YOLO dataset")
    
    args = parser.parse_args()
    
    convert_dataset(args.source, args.output)
