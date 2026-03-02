"""
Convert Rebar Dataset from Pascal VOC (XML) to YOLO format
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

def convert_voc_to_yolo(xml_path, img_width, img_height):
    """Convert VOC XML annotation to YOLO format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    labels = []
    for obj in root.findall('object'):
        # All objects are rebar (class 0 for specialist)
        class_id = 0
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return labels

def get_image_size(img_path):
    """Get image dimensions."""
    from PIL import Image
    with Image.open(img_path) as img:
        return img.width, img.height

def process_folder(src_folder, dst_folder, split_name):
    """Process a folder of images and XML files."""
    src = Path(src_folder)
    dst = Path(dst_folder)
    
    img_dst = dst / split_name / "images"
    lbl_dst = dst / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for xml_file in src.glob("*.xml"):
        img_file = xml_file.with_suffix('.jpg')
        if not img_file.exists():
            continue
        
        try:
            width, height = get_image_size(img_file)
            labels = convert_voc_to_yolo(xml_file, width, height)
            
            if labels:
                # Copy image
                shutil.copy(img_file, img_dst / img_file.name)
                
                # Write labels
                txt_file = lbl_dst / xml_file.with_suffix('.txt').name
                with open(txt_file, 'w') as f:
                    f.write('\n'.join(labels))
                
                count += 1
        except Exception as e:
            print(f"  Error: {xml_file.name}: {e}")
    
    return count

def main():
    base = Path(r"C:\Users\spect\ConcreteSpotDetection\datasets\specialists")
    src_dataset = base / "rebar_new" / "Dataset"
    dst_dataset = base / "exposed_rebar"
    
    print("=" * 60)
    print("REBAR VOC → YOLO CONVERSION")
    print("=" * 60)
    
    # Process Training folder
    print("\n[1/3] Converting Training set...")
    train_count = process_folder(src_dataset / "Training", dst_dataset, "train")
    print(f"  Converted: {train_count} images")
    
    # Process Validation folder  
    print("\n[2/3] Converting Validation set...")
    val_count = process_folder(src_dataset / "Validation", dst_dataset, "valid")
    print(f"  Converted: {val_count} images")
    
    # Process Augmentation folder (add to training)
    print("\n[3/3] Converting Augmentation set (adding to train)...")
    aug_count = process_folder(src_dataset / "Augmentation", dst_dataset, "train")
    print(f"  Converted: {aug_count} images")
    
    print("\n" + "=" * 60)
    print(f"TOTAL: {train_count + aug_count} train, {val_count} valid")
    print(f"Output: {dst_dataset}")
    print("=" * 60)

if __name__ == "__main__":
    main()
