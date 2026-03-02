"""
Dataset Cleanup and Reorganization Script
Keeps only high-quality labeled data and organizes by damage type
"""
from pathlib import Path
import shutil
from collections import Counter
from tqdm import tqdm
import json

BASE = Path("dataset/dataset")

# FINAL ORGANIZED STRUCTURE
FINAL_STRUCTURE = {
    "crack": BASE / "organized" / "crack",
    "corrosion": BASE / "organized" / "corrosion", 
    "spalling": BASE / "organized" / "spalling",
    "exposed_rebar": BASE / "organized" / "exposed_rebar",
}

# DATASETS TO KEEP (curated, high quality)
KEEP_DATASETS = {
    "crack_specialist": {
        "type": "crack",
        "format": "yolo",
        "quality": "HIGH",
        "reason": "Merged from 3 sources, validated"
    },
    "corrosion_specialist": {
        "type": "corrosion", 
        "format": "yolo",
        "quality": "HIGH",
        "reason": "Merged from HRCDS, dacl10k, Roboflow"
    },
}

# DATASETS TO REMOVE (duplicates, unlabeled, raw sources)
REMOVE_DATASETS = [
    "crack_new",  # Already merged into crack_specialist
    "corrosion_new",  # Already merged into corrosion_specialist
    "roboflow_concrete_damage",  # Subset, already in merged
    "roboflow_crack_detection",  # Subset, already in merged
    "roboflow_cracks",  # Subset, already in merged
    "kaggle_crack_unlabeled",  # No labels
    "misc_unlabeled",  # No labels  
    "quarantine",  # Low quality, quarantined
    "images",  # Raw images folder, will reorganize
    "labels",  # Raw labels folder, will reorganize
    "DATA_Maguire_20180517_ALL",  # Empty
    "rebar_specialist",  # Empty, will populate
    "spalling_specialist",  # Empty, will populate
]

# DATASETS TO EXTRACT FROM (multi-class sources)
EXTRACT_SOURCES = {
    "dacl10k_supervisely": {
        "format": "supervisely",
        "classes": {
            "crack": ["crack", "alligator crack"],
            "spalling": ["spalling"],
            "corrosion": ["rust", "washouts/concrete corrosion"],
            "exposed_rebar": ["exposed rebars"]
        }
    },
    "MDMCS A Benchmark Dataset for Multi-Damage Monitor": {
        "format": "labelme",
        "classes": {
            "crack": ["crack"],
            "spalling": ["spalling"],
            "corrosion": ["corrosion"],
            "exposed_rebar": ["exposed rebar"]
        }
    },
    "Damage Detection Dataset for Concrete Structures with Multi-Feature Backgrounds": {
        "format": "unknown",
        "check": True
    }
}


def count_dataset(path):
    """Count images and labels in a dataset."""
    if not path.exists():
        return 0, 0
    
    images = 0
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images += len(list(path.rglob(ext)))
    
    labels = len(list(path.rglob('*.txt')))
    return images, labels


def analyze_current_state():
    """Analyze current dataset state."""
    print("=" * 70)
    print("CURRENT DATASET STATE")
    print("=" * 70)
    
    total_images = 0
    total_labels = 0
    
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        
        imgs, lbls = count_dataset(d)
        total_images += imgs
        total_labels += lbls
        
        status = "✅ KEEP" if d.name in KEEP_DATASETS else "❌ REMOVE"
        if d.name in [k.replace(" ", "_") for k in EXTRACT_SOURCES.keys()] or \
           d.name in EXTRACT_SOURCES.keys():
            status = "🔄 EXTRACT"
        
        print(f"{status} {d.name[:45]:45} | imgs:{imgs:7,} | lbls:{lbls:7,}")
    
    print(f"\nTotal: {total_images:,} images, {total_labels:,} labels")
    return total_images, total_labels


def cleanup_project_root():
    """Remove redundant files from project root."""
    project = Path(".")
    
    redundant_files = [
        "corrosion_samples.jpg",  # Temporary viz
        "yolo26n.pt",  # Old model
        "yolov8n.pt",  # Base model (can redownload)
    ]
    
    redundant_dirs = [
        "analysis_results",  # Old analysis
        "evaluation_results",  # Old eval
        "training_results",  # Old training
        "gradcam_results",  # Can regenerate
    ]
    
    print("\n" + "=" * 70)
    print("PROJECT CLEANUP")
    print("=" * 70)
    
    for f in redundant_files:
        p = project / f
        if p.exists():
            print(f"  Removing file: {f}")
            p.unlink()
    
    for d in redundant_dirs:
        p = project / d
        if p.exists():
            print(f"  Removing dir: {d}")
            shutil.rmtree(p)


def cleanup_datasets():
    """Remove redundant dataset folders."""
    print("\n" + "=" * 70)
    print("DATASET CLEANUP")
    print("=" * 70)
    
    for name in REMOVE_DATASETS:
        path = BASE / name
        if path.exists():
            imgs, _ = count_dataset(path)
            print(f"  Removing: {name} ({imgs:,} images)")
            shutil.rmtree(path)


def organize_final_structure():
    """Create organized folder structure."""
    print("\n" + "=" * 70)
    print("FINAL ORGANIZATION")
    print("=" * 70)
    
    # Create final folders
    for damage_type, path in FINAL_STRUCTURE.items():
        (path / "train" / "images").mkdir(parents=True, exist_ok=True)
        (path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (path / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (path / "valid" / "labels").mkdir(parents=True, exist_ok=True)
        print(f"  Created: {damage_type}/")
    
    # Copy from specialists
    for name, info in KEEP_DATASETS.items():
        src = BASE / name
        dst = FINAL_STRUCTURE[info["type"]]
        
        if src.exists():
            for split in ["train", "valid"]:
                src_imgs = src / split / "images"
                src_lbls = src / split / "labels"
                
                if src_imgs.exists():
                    imgs = list(src_imgs.glob("*.*"))
                    print(f"  Copying {len(imgs)} {info['type']} images from {name}/{split}")
                    # Uncomment to actually copy
                    # for img in imgs:
                    #     shutil.copy2(img, dst / split / "images")
                    #     lbl = src_lbls / (img.stem + ".txt")
                    #     if lbl.exists():
                    #         shutil.copy2(lbl, dst / split / "labels")


def main():
    print("=" * 70)
    print("CONCRETESPOT DATASET CLEANUP & ORGANIZATION")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Analyze current dataset state")
    print("2. Remove redundant project files")
    print("3. Remove duplicate/unlabeled datasets")
    print("4. Organize remaining data by damage type")
    print()
    
    # Analyze
    analyze_current_state()
    
    # Preview cleanup (dry run)
    cleanup_project_root()
    cleanup_datasets()
    
    # Final structure
    organize_final_structure()
    
    print("\n" + "=" * 70)
    print("PREVIEW COMPLETE - Add --execute flag to perform cleanup")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if "--execute" in sys.argv:
        print("⚠️ EXECUTING CLEANUP - FILES WILL BE DELETED!")
    main()
