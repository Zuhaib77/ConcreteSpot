#!/usr/bin/env python3
"""
Convert Pascal VOC XML format to YOLO format and merge with existing dataset
"""

import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
from typing import Tuple
import random


CLASS_MAP = {
    "crack": 0,
    "spalling": 1,
    "corrosion": 2,
    "exposed_rebar": 3,
    "exposed rebar": 3,
    "damage": 0,
    "cra": 0,
    "amage": 0,
}


def parse_voc_annotation(xml_path: Path) -> Tuple[int, int, list]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower().strip()
        if name not in CLASS_MAP:
            continue
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w = max(0, min(1, w))
        h = max(0, min(1, h))
        
        class_id = CLASS_MAP[name]
        objects.append((class_id, x_center, y_center, w, h))
    
    return width, height, objects


def convert_voc_to_yolo(
    source_dir: Path,
    output_dir: Path,
    prefix: str = "voc"
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    img_src = source_dir / "img"
    ann_src = source_dir / "annot"
    
    img_files = list(img_src.glob("*.*"))
    random.shuffle(img_files)
    
    n_total = len(img_files)
    n_train = int(n_total * 0.85)
    n_val = int(n_total * 0.10)
    
    train_files = img_files[:n_train]
    val_files = img_files[n_train:n_train + n_val]
    test_files = img_files[n_train + n_val:]
    
    splits = [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]
    
    for split_name, files in splits:
        img_dst = output_dir / "images" / split_name
        lbl_dst = output_dir / "labels" / split_name
        
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split_name} ({len(files)} images)...")
        converted = 0
        
        for img_file in files:
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            ann_file = ann_src / f"{img_file.stem}.xml"
            if not ann_file.exists():
                continue
            
            try:
                width, height, objects = parse_voc_annotation(ann_file)
            except Exception as e:
                print(f"  Error parsing {ann_file}: {e}")
                continue
            
            if not objects:
                continue
            
            new_name = f"{prefix}_{img_file.name}"
            shutil.copy(img_file, img_dst / new_name)
            
            label_file = lbl_dst / f"{prefix}_{img_file.stem}.txt"
            with open(label_file, 'w') as f:
                for cls_id, x, y, w, h in objects:
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            converted += 1
        
        print(f"  Converted: {converted} images")
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Pascal VOC to YOLO and merge")
    parser.add_argument("--source", type=str, required=True, help="Path to VOC dataset (with img/ and annot/)")
    parser.add_argument("--output", type=str, required=True, help="Output YOLO dataset folder")
    parser.add_argument("--prefix", type=str, default="voc", help="Prefix for merged files")
    
    args = parser.parse_args()
    
    convert_voc_to_yolo(args.source, args.output, args.prefix)
