"""
Convert Corrosion Segmentation Dataset to YOLO Format
Extracts bounding boxes from segmentation masks (*_lab.png)
"""
import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def mask_to_yolo_boxes(mask_path, min_area=100):
    """Convert segmentation mask to YOLO bounding boxes."""
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return []
    
    h, w = mask.shape[:2]
    
    # Red channel = corrosion (R > 200, G < 50, B < 50)
    red_mask = (mask[:,:,2] > 200) & (mask[:,:,1] < 50) & (mask[:,:,0] < 50)
    red_mask = red_mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:  # Skip tiny regions
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        cx = (x + bw/2) / w
        cy = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        
        # Class 0 = corrosion
        boxes.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    return boxes

def process_split(src_dir, dst_dir, split_name):
    """Process a split (train/val/test)."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    
    img_dst = dst / split_name / "images"
    lbl_dst = dst / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    
    count = 0
    skipped = 0
    
    # Find all image files (not _lab.png)
    for img_file in src.glob("*.png"):
        if "_lab" in img_file.name:
            continue
        
        # Find corresponding label file
        mask_file = img_file.with_name(img_file.stem + "_lab.png")
        if not mask_file.exists():
            continue
        
        # Convert mask to YOLO boxes
        boxes = mask_to_yolo_boxes(mask_file)
        
        if not boxes:
            skipped += 1
            continue
        
        # Copy image
        shutil.copy(img_file, img_dst / img_file.name)
        
        # Write labels
        txt_file = lbl_dst / img_file.with_suffix('.txt').name
        with open(txt_file, 'w') as f:
            f.write('\n'.join(boxes))
        
        count += 1
    
    return count, skipped

def main():
    base = Path(r"C:\Users\spect\ConcreteSpotDetection\datasets\specialists")
    src_base = base / "corrosion_kaggle" / "spalling_corrosion_patches"
    dst = base / "corrosion_kaggle_yolo"
    
    print("=" * 60)
    print("CORROSION SEGMENTATION → YOLO CONVERSION")
    print("=" * 60)
    
    # Process each split
    print("\n[1/3] Converting train set...")
    train_count, train_skip = process_split(src_base / "train", dst, "train")
    print(f"  Converted: {train_count}, Skipped (no boxes): {train_skip}")
    
    print("\n[2/3] Converting val set...")
    val_count, val_skip = process_split(src_base / "val", dst, "valid")
    print(f"  Converted: {val_count}, Skipped (no boxes): {val_skip}")
    
    print("\n[3/3] Converting test set...")
    test_count, test_skip = process_split(src_base / "test", dst, "test")
    print(f"  Converted: {test_count}, Skipped (no boxes): {test_skip}")
    
    # Create data.yaml
    yaml_content = f"""# Corrosion Specialist Dataset (from Kaggle segmentation)
path: {dst}
train: train/images
val: valid/images

names:
  0: corrosion

nc: 1
"""
    with open(dst / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {train_count} train, {val_count} valid, {test_count} test")
    print(f"Output: {dst}")
    print("data.yaml created!")
    print("=" * 60)

if __name__ == "__main__":
    main()
