"""
Create specialist datasets - one per class
Extracts single-class data for specialist model training
"""
import os
import shutil
from pathlib import Path

# Source: V3 pure_quality has best crack (99.5%)
# We'll also use organized_v4 data

def create_specialist_dataset(class_name, class_id, sources, output_name):
    """Create a single-class specialist dataset"""
    output = Path(f"dataset/specialists/{output_name}")
    
    for split in ["train", "valid"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    
    counts = {"train": 0, "valid": 0}
    
    for source_path in sources:
        source = Path(source_path)
        if not source.exists():
            print(f"  Warning: {source} not found, skipping")
            continue
            
        for split in ["train", "valid"]:
            img_dir = source / split / "images"
            lbl_dir = source / split / "labels"
            
            if not img_dir.exists():
                continue
            
            for img_path in img_dir.iterdir():
                if not img_path.is_file():
                    continue
                
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                
                # Read labels and filter for this class
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                class_lines = [l for l in lines if l.strip().startswith(f"{class_id} ")]
                
                if not class_lines:
                    continue
                
                # Copy image
                dst_img = output / split / "images" / f"{output_name}_{img_path.name}"
                if not dst_img.exists():
                    shutil.copy(img_path, dst_img)
                
                # Write label with class_id = 0 (single class model)
                dst_lbl = output / split / "labels" / f"{output_name}_{img_path.stem}.txt"
                with open(dst_lbl, 'w') as f:
                    for line in class_lines:
                        parts = line.strip().split()
                        # Change class_id to 0 for single-class model
                        parts[0] = "0"
                        f.write(" ".join(parts) + "\n")
                
                counts[split] += 1
    
    # Create data.yaml
    yaml_content = f"""# {class_name.title()} Specialist Dataset
path: {output.absolute()}
train: train/images
val: valid/images

names:
  0: {class_name}

nc: 1
"""
    with open(output / "data.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"  {class_name}: {counts['train']} train, {counts['valid']} valid")
    return counts

def main():
    print("="*60)
    print("CREATING SPECIALIST DATASETS")
    print("="*60)
    
    # Define class sources (best dataset per class)
    specialists = {
        "crack": {
            "class_id": 0,
            "sources": [
                "dataset/dataset/pure_quality_v3",  # 99.5% mAP
                "dataset/dataset/organized_v4",
            ]
        },
        "efflorescence": {
            "class_id": 4,
            "sources": [
                "dataset/dataset/organized_v4",  # 99.1% mAP
            ]
        },
        "spalling": {
            "class_id": 1,
            "sources": [
                "dataset/dataset/organized_v4",  # 59.1% mAP (best available)
                "dataset/dataset/pure_quality_v3",
            ]
        },
        "corrosion": {
            "class_id": 2,
            "sources": [
                "dataset/dataset/organized_v4",
                "dataset/dataset/pure_quality_v3",  # 51.5% mAP
            ]
        },
        "exposed_rebar": {
            "class_id": 3,
            "sources": [
                "dataset/dataset/organized_v4",
                "dataset/dataset/pure_quality_v3",
            ]
        },
    }
    
    for class_name, config in specialists.items():
        print(f"\nCreating {class_name} specialist dataset...")
        create_specialist_dataset(
            class_name, 
            config["class_id"], 
            config["sources"],
            class_name
        )
    
    print("\n" + "="*60)
    print("SPECIALIST DATASETS READY")
    print("="*60)
    print("\nDatasets created in: dataset/specialists/")

if __name__ == "__main__":
    main()
