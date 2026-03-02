"""
Comprehensive Data Cleaning Script v2
More aggressive cleaning to improve crack class performance.
"""
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
TRAIN_IMAGES = Path("dataset/dataset/images/train")
TRAIN_LABELS = Path("dataset/dataset/labels/train")
QUARANTINE_DIR = Path("dataset/dataset/quarantine")

def analyze_dataset_sources():
    """Analyze which datasets contribute to crack annotations."""
    sources = defaultdict(lambda: {"total": 0, "crack_only": 0, "has_crack": 0})
    
    for label_path in TRAIN_LABELS.glob("*.txt"):
        # Determine source from filename prefix
        name = label_path.stem
        if name.startswith("dacl10k_"):
            source = "dacl10k"
        elif name.startswith("roboflow_crack_detection"):
            source = "roboflow_crack_detection"
        elif name.startswith("roboflow_cracks"):
            source = "roboflow_cracks"
        elif name.startswith("roboflow_concrete_damage"):
            source = "roboflow_concrete_damage"
        elif name.startswith("pseudo_"):
            source = "pseudo_labeled"
        else:
            source = "original"
        
        sources[source]["total"] += 1
        
        # Read label
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        classes = set(int(l.split()[0]) for l in lines if l)
        
        if 0 in classes:  # Has crack
            sources[source]["has_crack"] += 1
            if classes == {0}:  # Only crack
                sources[source]["crack_only"] += 1
    
    return sources

def clean_dacl10k_cracks():
    """Remove crack annotations from dacl10k that are likely problematic."""
    print("\n2. Cleaning dacl10k crack annotations...")
    
    removed = 0
    modified = 0
    
    for label_path in tqdm(list(TRAIN_LABELS.glob("dacl10k_*.txt")), desc="Processing dacl10k"):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out crack annotations (class 0) - keep other classes
        new_lines = [l for l in lines if not l.strip().startswith('0 ')]
        
        if len(new_lines) == 0 and len(lines) > 0:
            # Was crack-only, move to quarantine
            img_name = label_path.stem + ".jpg"
            img_path = TRAIN_IMAGES / img_name
            
            if img_path.exists():
                shutil.move(str(img_path), QUARANTINE_DIR / "images" / img_path.name)
            shutil.move(str(label_path), QUARANTINE_DIR / "labels" / label_path.name)
            removed += 1
        elif len(new_lines) < len(lines):
            # Had crack + other classes, just remove crack
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
            modified += 1
    
    return removed, modified

def clean_pseudo_labels():
    """Remove all pseudo-labeled images as they may add noise."""
    print("\n3. Removing pseudo-labeled images...")
    
    removed = 0
    for label_path in tqdm(list(TRAIN_LABELS.glob("pseudo_*.txt")), desc="Removing pseudo"):
        img_name = label_path.stem + ".jpg"
        img_path = TRAIN_IMAGES / img_name
        
        if img_path.exists():
            shutil.move(str(img_path), QUARANTINE_DIR / "images" / img_path.name)
        shutil.move(str(label_path), QUARANTINE_DIR / "labels" / label_path.name)
        removed += 1
    
    return removed

def main():
    print("=" * 60)
    print("Comprehensive Data Cleaning v2")
    print("=" * 60)
    
    # Create quarantine
    (QUARANTINE_DIR / "images").mkdir(parents=True, exist_ok=True)
    (QUARANTINE_DIR / "labels").mkdir(parents=True, exist_ok=True)
    
    # 1. Analyze sources
    print("\n1. Analyzing dataset sources...")
    sources = analyze_dataset_sources()
    
    print("\nDataset breakdown:")
    for src, stats in sorted(sources.items()):
        print(f"  {src}:")
        print(f"    Total: {stats['total']}")
        print(f"    Has crack: {stats['has_crack']} ({100*stats['has_crack']/max(stats['total'],1):.1f}%)")
        print(f"    Crack only: {stats['crack_only']}")
    
    # 2. Clean dacl10k cracks (main problem)
    dacl_removed, dacl_modified = clean_dacl10k_cracks()
    print(f"  Removed: {dacl_removed}, Modified: {dacl_modified}")
    
    # 3. Remove pseudo-labels (added noise)
    pseudo_removed = clean_pseudo_labels()
    print(f"  Removed: {pseudo_removed}")
    
    # Final stats
    final_images = len(list(TRAIN_IMAGES.glob("*")))
    final_labels = len(list(TRAIN_LABELS.glob("*.txt")))
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final dataset:")
    print(f"  Images: {final_images}")
    print(f"  Labels: {final_labels}")

if __name__ == "__main__":
    main()
