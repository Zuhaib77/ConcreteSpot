"""
Convert OBB (Oriented Bounding Box) to standard YOLO format
and combine with Kaggle dataset for more training data
"""
import os
import shutil
from pathlib import Path

def obb_to_yolo(obb_line):
    """Convert OBB format (8 coords) to YOLO format (cx, cy, w, h)."""
    parts = obb_line.strip().split()
    if len(parts) < 9:  # class + 8 coords
        return None
    
    cls = parts[0]
    coords = [float(x) for x in parts[1:9]]
    
    # Get bounding rect from 4 corner points
    xs = coords[0::2]  # x1, x2, x3, x4
    ys = coords[1::2]  # y1, y2, y3, y4
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Convert to center, width, height
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    # Use class 0 for corrosion
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def convert_split(src_dir, dst_dir, split_name):
    """Convert OBB labels to YOLO format."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    
    img_dst = dst / split_name / "images"
    lbl_dst = dst / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_file in (src / "images").glob("*.*"):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        # Find label file
        lbl_file = src / "labels" / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue
        
        # Convert labels
        new_labels = []
        with open(lbl_file, 'r') as f:
            for line in f:
                yolo_line = obb_to_yolo(line)
                if yolo_line:
                    new_labels.append(yolo_line)
        
        if not new_labels:
            continue
        
        # Copy image
        shutil.copy(img_file, img_dst / img_file.name)
        
        # Write converted labels
        with open(lbl_dst / (img_file.stem + ".txt"), 'w') as f:
            f.write('\n'.join(new_labels))
        
        count += 1
    
    return count

def main():
    base = Path(r"C:\Users\spect\ConcreteSpotDetection\datasets\specialists")
    obb_dir = base / "corrosion_obb"
    combined_dir = base / "corrosion_combined"
    kaggle_dir = base / "corrosion_kaggle_yolo"
    
    print("=" * 60)
    print("COMBINING CORROSION DATASETS")
    print("=" * 60)
    
    # Step 1: Convert OBB to YOLO
    print("\n[1/3] Converting OBB dataset to YOLO format...")
    train_obb = convert_split(obb_dir / "train", combined_dir, "train")
    val_obb = convert_split(obb_dir / "valid", combined_dir, "valid")
    print(f"  OBB converted: {train_obb} train, {val_obb} valid")
    
    # Step 2: Copy Kaggle dataset
    print("\n[2/3] Adding Kaggle dataset...")
    train_kaggle = 0
    val_kaggle = 0
    
    for img in (kaggle_dir / "train" / "images").glob("*.png"):
        lbl = kaggle_dir / "train" / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(img, combined_dir / "train" / "images" / f"kaggle_{img.name}")
            shutil.copy(lbl, combined_dir / "train" / "labels" / f"kaggle_{img.stem}.txt")
            train_kaggle += 1
    
    for img in (kaggle_dir / "valid" / "images").glob("*.png"):
        lbl = kaggle_dir / "valid" / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(img, combined_dir / "valid" / "images" / f"kaggle_{img.name}")
            shutil.copy(lbl, combined_dir / "valid" / "labels" / f"kaggle_{img.stem}.txt")
            val_kaggle += 1
    
    print(f"  Kaggle added: {train_kaggle} train, {val_kaggle} valid")
    
    # Step 3: Create data.yaml
    print("\n[3/3] Creating data.yaml...")
    yaml_content = f"""# Combined Corrosion Dataset (OBB + Kaggle)
path: {combined_dir}
train: train/images
val: valid/images

names:
  0: corrosion

nc: 1
"""
    with open(combined_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    total_train = train_obb + train_kaggle
    total_val = val_obb + val_kaggle
    
    print("\n" + "=" * 60)
    print("COMBINED DATASET CREATED")
    print(f"  Train: {total_train} images")
    print(f"  Valid: {total_val} images")
    print(f"  Output: {combined_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
