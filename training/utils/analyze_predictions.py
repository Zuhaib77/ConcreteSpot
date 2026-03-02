"""
Prediction Analysis Script
Analyzes model predictions on training data to find mislabeled samples.
Focuses on crack class which has low mAP.
"""
import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# Paths
MODEL_PATH = "runs/detect/yolov8n_300ep_merged/weights/best.pt"
TRAIN_IMAGES = Path("dataset/dataset/images/train")
TRAIN_LABELS = Path("dataset/dataset/labels/train")
OUTPUT_DIR = Path("analysis_results")

# Class mapping
CLASSES = {0: "crack", 1: "spalling", 2: "corrosion", 3: "exposed_rebar"}

def parse_yolo_label(label_path):
    """Parse YOLO format label file."""
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    boxes.append({'class': cls_id, 'x': x_c, 'y': y_c, 'w': w, 'h': h})
    return boxes

def iou(box1, box2):
    """Calculate IoU between two boxes in YOLO format."""
    # Convert to corner format
    x1_min = box1['x'] - box1['w']/2
    x1_max = box1['x'] + box1['w']/2
    y1_min = box1['y'] - box1['h']/2
    y1_max = box1['y'] + box1['h']/2
    
    x2_min = box2['x'] - box2['w']/2
    x2_max = box2['x'] + box2['w']/2
    y2_min = box2['y'] - box2['h']/2
    y2_max = box2['y'] + box2['h']/2
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    xi_max = min(x1_max, x2_max)
    yi_min = max(y1_min, y2_min)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    inter_area = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def analyze_predictions():
    print("=" * 60)
    print("Prediction Analysis - Finding Mislabeled Crack Data")
    print("=" * 60)
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "false_positives").mkdir(exist_ok=True)
    (OUTPUT_DIR / "false_negatives").mkdir(exist_ok=True)
    (OUTPUT_DIR / "low_confidence").mkdir(exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Get crack-labeled images
    images = list(TRAIN_IMAGES.glob("*"))
    
    stats = {
        "total_images": 0,
        "crack_labeled": 0,
        "false_positives": [],  # Model predicts crack, no crack label
        "false_negatives": [],  # Has crack label, model doesn't detect
        "low_confidence": [],   # Low confidence crack predictions
        "class_confusion": defaultdict(int),  # Predicted as X, labeled as Y
    }
    
    print(f"Analyzing {len(images)} images...")
    
    for img_path in tqdm(images[:5000], desc="Analyzing"):  # Limit for speed
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        stats["total_images"] += 1
        
        # Get ground truth
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path)
        gt_cracks = [b for b in gt_boxes if b['class'] == 0]
        
        if gt_cracks:
            stats["crack_labeled"] += 1
        
        # Run inference
        results = model.predict(img_path, verbose=False, conf=0.3)
        pred_boxes = []
        for box in results[0].boxes:
            pred_boxes.append({
                'class': int(box.cls[0]),
                'x': float(box.xywhn[0][0]),
                'y': float(box.xywhn[0][1]),
                'w': float(box.xywhn[0][2]),
                'h': float(box.xywhn[0][3]),
                'conf': float(box.conf[0])
            })
        
        pred_cracks = [b for b in pred_boxes if b['class'] == 0]
        
        # Check for false positives (predicted crack, no GT crack)
        if pred_cracks and not gt_cracks:
            stats["false_positives"].append({
                "image": str(img_path),
                "predictions": pred_cracks,
                "gt_classes": [CLASSES[b['class']] for b in gt_boxes]
            })
        
        # Check for false negatives (GT crack, no prediction)
        if gt_cracks and not pred_cracks:
            stats["false_negatives"].append({
                "image": str(img_path),
                "gt_count": len(gt_cracks)
            })
        
        # Check for low confidence predictions
        for pc in pred_cracks:
            if pc['conf'] < 0.5:
                stats["low_confidence"].append({
                    "image": str(img_path),
                    "conf": pc['conf']
                })
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total images analyzed: {stats['total_images']}")
    print(f"Images with crack labels: {stats['crack_labeled']}")
    print(f"False positives (pred crack, no label): {len(stats['false_positives'])}")
    print(f"False negatives (label crack, no pred): {len(stats['false_negatives'])}")
    print(f"Low confidence predictions (<0.5): {len(stats['low_confidence'])}")
    
    # Save problematic images list
    report = {
        "summary": {
            "total_analyzed": stats["total_images"],
            "crack_labeled": stats["crack_labeled"],
            "false_positives": len(stats["false_positives"]),
            "false_negatives": len(stats["false_negatives"]),
            "low_confidence": len(stats["low_confidence"])
        },
        "false_positives": stats["false_positives"][:100],
        "false_negatives": stats["false_negatives"][:100],
        "low_confidence": stats["low_confidence"][:100]
    }
    
    with open(OUTPUT_DIR / "analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {OUTPUT_DIR / 'analysis_report.json'}")
    
    # Copy sample problematic images for review
    print("\nCopying sample images for review...")
    for i, fp in enumerate(stats["false_positives"][:20]):
        src = Path(fp["image"])
        dst = OUTPUT_DIR / "false_positives" / src.name
        if src.exists():
            shutil.copy(src, dst)
    
    for i, fn in enumerate(stats["false_negatives"][:20]):
        src = Path(fn["image"])
        dst = OUTPUT_DIR / "false_negatives" / src.name
        if src.exists():
            shutil.copy(src, dst)
    
    print("Done!")

if __name__ == "__main__":
    analyze_predictions()
