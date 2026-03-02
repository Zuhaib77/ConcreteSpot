"""
V6 Dataset Creator - Semi-Automatic with Grounding DINO + SAM
Uses zero-shot detection for high-quality annotations

NOTE: This requires additional dependencies:
  pip install groundingdino segment-anything torch torchvision

For now, this is a placeholder that uses existing good data 
from V4 (organized) + some augmentation as a baseline.
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
V4_DATA = Path("dataset/dataset/organized_v4")
OUTPUT = Path("dataset/dataset/semiauto_v6")

def setup_dirs():
    for split in ["train", "valid"]:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

def copy_from_v4():
    """Copy best data from V4 as baseline"""
    print("Copying baseline from V4 (organized with fixed IDs)...")
    
    if not V4_DATA.exists():
        print("  V4 not ready yet. Run create_v4_organized.py first.")
        return 0
    
    count = 0
    for split in ["train", "valid"]:
        src_img = V4_DATA / split / "images"
        src_lbl = V4_DATA / split / "labels"
        
        if not src_img.exists():
            continue
        
        for img in tqdm(list(src_img.glob("*.*")), desc=f"  V6 {split}"):
            lbl = src_lbl / (img.stem + ".txt")
            if not lbl.exists():
                continue
            
            # Copy with v6 prefix
            shutil.copy(img, OUTPUT / split / "images" / f"v6_{img.name}")
            shutil.copy(lbl, OUTPUT / split / "labels" / f"v6_{lbl.name}")
            count += 1
    
    return count

def create_data_yaml():
    yaml_content = f"""# V6 Semi-Auto Dataset (DINO+SAM)
# Currently a placeholder using V4 data as baseline
path: {OUTPUT.absolute()}
train: train/images
val: valid/images

names:
  0: crack
  1: spalling
  2: corrosion
  3: exposed_rebar
  4: efflorescence

nc: 5
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    print("="*60)
    print("V6: SEMI-AUTO DATASET (DINO+SAM PLACEHOLDER)")
    print("="*60)
    print("\nNOTE: Full DINO+SAM annotation requires:")
    print("  1. Install: pip install groundingdino segment-anything")
    print("  2. Download SAM weights")
    print("  3. Run inference on raw images")
    print("\nFor now, using V4 organized data as baseline...\n")
    
    setup_dirs()
    count = copy_from_v4()
    
    print(f"\n  Total images: {count}")
    print("\nTo upgrade to full DINO+SAM:")
    print("  1. Uncomment DINO+SAM code below")
    print("  2. Add raw images to dataset/raw/")
    print("  3. Re-run this script")
    
    create_data_yaml()
    print(f"\nCreated: {OUTPUT / 'data.yaml'}")

# =======================================================
# DINO+SAM CODE (COMMENTED OUT - NEEDS DEPENDENCIES)
# =======================================================
"""
import torch
from groundingdino.util.inference import load_model, run_inference
from segment_anything import sam_model_registry, SamPredictor

# Class prompts for Grounding DINO
PROMPTS = {
    "crack": "crack . fracture . fissure",
    "spalling": "spalling . peeling . flaking concrete",
    "corrosion": "rust . corrosion . rust stain",
    "exposed_rebar": "exposed reinforcement bar . rebar . steel bar",
    "efflorescence": "white deposit . efflorescence . salt stain"
}

def run_dino_sam(image_path, model, sam_predictor):
    # Run Grounding DINO for detection
    # Run SAM for segmentation
    # Convert mask to bbox
    pass
"""

if __name__ == "__main__":
    main()
