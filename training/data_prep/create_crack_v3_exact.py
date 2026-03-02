"""
Create Crack Specialist Dataset from pure_quality_v3
- Uses EXACT same train/valid split as V3 training (99.5% accuracy)
- Extracts crack-only annotations (class 0)
"""
import shutil
from pathlib import Path

V3_SOURCE = Path("dataset/dataset/pure_quality_v3")
OUTPUT = Path("dataset/specialists/crack_v3_exact")

def extract_crack_only():
    for split in ["train", "valid"]:
        src_img = V3_SOURCE / split / "images"
        src_lbl = V3_SOURCE / split / "labels"
        
        dst_img = OUTPUT / split / "images"
        dst_lbl = OUTPUT / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_path in src_img.iterdir():
            if not img_path.is_file():
                continue
            
            lbl_path = src_lbl / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            # Class 0 = crack in V3
            crack_lines = [l for l in lines if l.strip().startswith("0 ")]
            
            if not crack_lines:
                continue
            
            # Copy image
            shutil.copy(img_path, dst_img / img_path.name)
            
            # Write label (class 0 stays as class 0 - single class)
            with open(dst_lbl / (img_path.stem + ".txt"), 'w') as f:
                f.writelines(crack_lines)
            
            count += 1
        
        print(f"{split}: {count} images")

    # Create data.yaml
    yaml = f"""# Crack Specialist Dataset (from pure_quality_v3)
# EXACT same data that achieved 99.5% crack accuracy in V3
path: {OUTPUT.absolute()}
train: train/images
val: valid/images

names:
  0: crack

nc: 1
"""
    with open(OUTPUT / "data.yaml", 'w') as f:
        f.write(yaml)
    print(f"\nDataset ready: {OUTPUT}")

if __name__ == "__main__":
    print("Extracting crack-only from pure_quality_v3...")
    extract_crack_only()
