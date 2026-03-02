"""
Dataset Balancer for ConcreteSpot v2.0
1. Undersample crack to ~20K
2. Augment exposed_rebar and efflorescence to ~8K each

Uses albumentations for augmentation
"""
import json
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed, using basic augmentation")

# Configuration
INPUT = Path("dataset/dataset/unified_95plus")
OUTPUT = Path("dataset/dataset/balanced_95plus")

# Target counts per class
TARGETS = {
    0: 20000,  # crack: undersample
    1: 20000,  # spalling: keep (27K)
    2: 12000,  # corrosion: keep (11K)
    3: 8000,   # exposed_rebar: augment from 3.6K
    4: 8000,   # efflorescence: augment from 4K
}

CLASS_NAMES = ["crack", "spalling", "corrosion", "exposed_rebar", "efflorescence"]


def get_augmentation_pipeline():
    """Create augmentation pipeline."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            ], p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    return None


def basic_augment(img, boxes):
    """Basic augmentation without albumentations."""
    img_arr = np.array(img)
    augmented = []
    
    # Original
    augmented.append((img, boxes))
    
    # Horizontal flip
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_boxes = []
    for box in boxes:
        cls, x, y, w, h = box
        flipped_boxes.append([cls, 1 - x, y, w, h])
    augmented.append((flipped, flipped_boxes))
    
    # Vertical flip
    vflipped = img.transpose(Image.FLIP_TOP_BOTTOM)
    vflipped_boxes = []
    for box in boxes:
        cls, x, y, w, h = box
        vflipped_boxes.append([cls, x, 1 - y, w, h])
    augmented.append((vflipped, vflipped_boxes))
    
    return augmented


def read_labels(label_path):
    """Read YOLO labels from file."""
    boxes = []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append([int(parts[0])] + [float(x) for x in parts[1:5]])
    return boxes


def write_labels(label_path, boxes):
    """Write YOLO labels to file."""
    with open(label_path, 'w') as f:
        for box in boxes:
            f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def get_primary_class(boxes):
    """Get primary class from list of boxes."""
    if not boxes:
        return None
    class_counts = Counter(box[0] for box in boxes)
    return class_counts.most_common(1)[0][0]


def balance_dataset():
    """Balance dataset by undersampling and augmenting."""
    print("=" * 60)
    print("DATASET BALANCER")
    print("=" * 60)
    
    # Create output directories
    for split in ["train", "valid", "test"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get augmentation pipeline
    aug_pipeline = get_augmentation_pipeline()
    
    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()}")
        print("=" * 60)
        
        img_dir = INPUT / split / "images"
        lbl_dir = INPUT / split / "labels"
        
        if not img_dir.exists():
            continue
        
        # Group images by primary class
        images_by_class = defaultdict(list)
        all_images = list(img_dir.glob("*.*"))
        
        print(f"  Analyzing {len(all_images)} images...")
        for img_path in tqdm(all_images, desc="  Grouping"):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = read_labels(lbl_path)
            primary_class = get_primary_class(boxes)
            if primary_class is not None:
                images_by_class[primary_class].append((img_path, boxes))
        
        # Print current distribution
        print("\n  Current distribution:")
        for cls_id, images in sorted(images_by_class.items()):
            print(f"    {CLASS_NAMES[cls_id]:15} {len(images):>6,} images")
        
        # Process each class
        for cls_id in range(5):
            images = images_by_class.get(cls_id, [])
            target = TARGETS.get(cls_id, len(images))
            
            if split != "train":
                # For valid/test, just copy all
                target = len(images)
            
            class_name = CLASS_NAMES[cls_id]
            print(f"\n  Processing {class_name}: {len(images)} -> {target}")
            
            if len(images) >= target:
                # Undersample
                selected = random.sample(images, target)
                for img_path, boxes in tqdm(selected, desc=f"    Copying {class_name}"):
                    new_name = img_path.name
                    shutil.copy2(img_path, OUTPUT / split / "images" / new_name)
                    write_labels(OUTPUT / split / "labels" / (img_path.stem + ".txt"), boxes)
            else:
                # First copy all originals
                for img_path, boxes in images:
                    shutil.copy2(img_path, OUTPUT / split / "images" / img_path.name)
                    write_labels(OUTPUT / split / "labels" / (img_path.stem + ".txt"), boxes)
                
                # Then augment to reach target
                needed = target - len(images)
                aug_count = 0
                
                print(f"    Need to augment {needed} more images")
                
                pbar = tqdm(total=needed, desc=f"    Augmenting {class_name}")
                while aug_count < needed:
                    for img_path, boxes in random.sample(images, min(len(images), needed - aug_count)):
                        if aug_count >= needed:
                            break
                        
                        try:
                            img = Image.open(img_path).convert("RGB")
                            img_arr = np.array(img)
                        except:
                            continue
                        
                        if aug_pipeline and boxes:
                            try:
                                # Prepare bboxes for albumentations
                                bboxes = [[b[1], b[2], b[3], b[4]] for b in boxes]
                                class_labels = [b[0] for b in boxes]
                                
                                transformed = aug_pipeline(image=img_arr, bboxes=bboxes, class_labels=class_labels)
                                aug_img = Image.fromarray(transformed['image'])
                                aug_boxes = [[cl] + list(bb) for bb, cl in zip(transformed['bboxes'], transformed['class_labels'])]
                                
                                if aug_boxes:
                                    aug_name = f"aug_{aug_count}_{img_path.stem}"
                                    aug_img.save(OUTPUT / split / "images" / f"{aug_name}.jpg")
                                    write_labels(OUTPUT / split / "labels" / f"{aug_name}.txt", aug_boxes)
                                    aug_count += 1
                                    pbar.update(1)
                            except Exception as e:
                                continue
                        else:
                            # Basic augmentation
                            augmented = basic_augment(img, boxes)
                            for i, (aug_img, aug_boxes) in enumerate(augmented[1:]):  # Skip original
                                if aug_count >= needed:
                                    break
                                aug_name = f"aug_{aug_count}_{i}_{img_path.stem}"
                                aug_img.save(OUTPUT / split / "images" / f"{aug_name}.jpg")
                                write_labels(OUTPUT / split / "labels" / f"{aug_name}.txt", aug_boxes)
                                aug_count += 1
                                pbar.update(1)
                
                pbar.close()
    
    # Create data.yaml
    yaml_content = f"""# ConcreteSpot v2.0 Balanced Dataset
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
    """Print final summary."""
    print("\n" + "=" * 60)
    print("FINAL BALANCED DATASET SUMMARY")
    print("=" * 60)
    
    for split in ["train", "valid", "test"]:
        img_count = len(list((OUTPUT / split / "images").glob("*.*")))
        lbl_count = len(list((OUTPUT / split / "labels").glob("*.txt")))
        print(f"  {split}: {img_count:,} images, {lbl_count:,} labels")
    
    # Class distribution for train
    print("\n  Class distribution (train):")
    class_counts = Counter()
    for lbl_file in (OUTPUT / "train" / "labels").glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_counts[int(parts[0])] += 1
                    except:
                        pass
    
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:15} {class_counts.get(i, 0):>6,} annotations")


def main():
    random.seed(42)  # Reproducibility
    balance_dataset()
    print_summary()
    print("\n" + "=" * 60)
    print("BALANCING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
