#!/usr/bin/env python3
"""
Model Evaluation Script for ConcreteSpot
=========================================
Evaluates trained YOLOv8 model on a test dataset.
Outputs: mAP, Precision, Recall, F1, Confusion Matrix

Usage:
    python evaluate.py --model models/yolov8_concrete.pt --data dataset/data.yaml
    python evaluate.py --model models/yolov8_concrete.pt --images /path/to/test/folder
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install ultralytics matplotlib numpy")
    exit(1)


def evaluate_model(
    model_path: str,
    data_yaml: str = None,
    images_folder: str = None,
    output_dir: str = "evaluation_results",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
):
    """
    Evaluate model on test data.
    
    Args:
        model_path: Path to trained .pt model
        data_yaml: Path to data.yaml (uses test split)
        images_folder: Alternative: folder of images to evaluate
        output_dir: Where to save results
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ConcreteSpot Model Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml or images_folder}")
    print(f"Confidence: {conf_threshold}")
    print(f"IoU: {iou_threshold}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    if data_yaml:
        # Validate on test split
        results = model.val(
            data=data_yaml,
            split='test',
            conf=conf_threshold,
            iou=iou_threshold,
            plots=True,
            save_json=True,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model': str(model_path),
            'dataset': str(data_yaml),
            'metrics': {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-6)),
            },
            'per_class': {}
        }
        
        # Per-class metrics
        class_names = results.names
        for i, name in class_names.items():
            if i < len(results.box.ap50):
                metrics['per_class'][name] = {
                    'ap50': float(results.box.ap50[i]),
                    'ap': float(results.box.ap[i]) if i < len(results.box.ap) else 0.0
                }
        
        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"mAP@50:      {metrics['metrics']['mAP50']:.4f} ({metrics['metrics']['mAP50']*100:.2f}%)")
        print(f"mAP@50-95:   {metrics['metrics']['mAP50-95']:.4f} ({metrics['metrics']['mAP50-95']*100:.2f}%)")
        print(f"Precision:   {metrics['metrics']['precision']:.4f} ({metrics['metrics']['precision']*100:.2f}%)")
        print(f"Recall:      {metrics['metrics']['recall']:.4f} ({metrics['metrics']['recall']*100:.2f}%)")
        print(f"F1 Score:    {metrics['metrics']['f1']:.4f}")
        print(f"\nPer-Class AP@50:")
        for name, values in metrics['per_class'].items():
            print(f"  {name:15s}: {values['ap50']:.4f} ({values['ap50']*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # Save results
        results_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        return metrics
        
    elif images_folder:
        # Run inference on folder (no ground truth comparison)
        images_folder = Path(images_folder)
        image_files = list(images_folder.glob("*.jpg")) + \
                      list(images_folder.glob("*.jpeg")) + \
                      list(images_folder.glob("*.png"))
        
        print(f"Found {len(image_files)} images")
        
        all_detections = []
        for img_path in image_files:
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        detections.append({
                            'class': model.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist()
                        })
            
            all_detections.append({
                'image': str(img_path.name),
                'detections': detections,
                'count': len(detections)
            })
        
        # Summary
        total_detections = sum(d['count'] for d in all_detections)
        class_counts = {}
        for d in all_detections:
            for det in d['detections']:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n{'='*60}")
        print("INFERENCE RESULTS (No Ground Truth)")
        print(f"{'='*60}")
        print(f"Images processed: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"\nDetections per class:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls:15s}: {count}")
        print(f"{'='*60}\n")
        
        # Save results
        results_file = output_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model': str(model_path),
                'images_folder': str(images_folder),
                'total_images': len(image_files),
                'total_detections': total_detections,
                'class_counts': class_counts,
                'detections': all_detections
            }, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        return all_detections
    
    else:
        print("Error: Provide either --data or --images")
        return None


def create_comparison_plot(results_list: list, output_path: str):
    """
    Create comparison bar chart for multiple model evaluations.
    
    Args:
        results_list: List of (model_name, metrics_dict) tuples
        output_path: Where to save the plot
    """
    if not results_list:
        return
    
    models = [r[0] for r in results_list]
    metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_names))
    width = 0.8 / len(models)
    
    for i, (model_name, metrics) in enumerate(results_list):
        values = [metrics['metrics'].get(m, 0) for m in metrics_names]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_name)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConcreteSpot model")
    parser.add_argument("--model", type=str, default="models/yolov8_concrete.pt",
                       help="Path to trained model")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to data.yaml (uses test split)")
    parser.add_argument("--images", type=str, default=None,
                       help="Path to folder of images (no ground truth)")
    parser.add_argument("--output", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                       help="IoU threshold")
    
    args = parser.parse_args()
    
    if not args.data and not args.images:
        args.data = "dataset/data.yaml"
    
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        images_folder=args.images,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
