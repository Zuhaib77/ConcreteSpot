#!/usr/bin/env python3
"""
Progressive Training Script with Metric Graphs
===============================================
Trains YOLOv8 with increasing epochs and generates comparison graphs.

Usage:
    python train_progressive.py --data dataset/data.yaml --epochs 100 200 300
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install ultralytics matplotlib pandas numpy")
    exit(1)


def train_and_evaluate(
    data_yaml: str,
    epochs: int,
    model_size: str = "n",
    batch_size: int = 16,
    imgsz: int = 640,
    output_dir: str = "training_results",
    augmentation: str = "balanced"
):
    """
    Train a model and return metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training YOLOv8{model_size} for {epochs} epochs")
    print(f"Augmentation: {augmentation}")
    print(f"{'='*60}\n")
    
    # Load base model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Augmentation settings
    aug_settings = {}
    if augmentation == "aggressive":
        aug_settings = {
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            'degrees': 45.0,
            'translate': 0.2,
            'scale': 0.9,
            'shear': 10.0,
            'perspective': 0.001,
            'flipud': 0.5,
            'fliplr': 0.5,
            'hsv_h': 0.03,
            'hsv_s': 0.9,
            'hsv_v': 0.6,
            'erasing': 0.5,
        }
    elif augmentation == "balanced":
        aug_settings = {
            'mosaic': 0.5,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'erasing': 0.2,
        }
    
    # Train
    run_name = f"yolov8{model_size}_{epochs}ep_{augmentation}"
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=run_name,
        patience=0,  # Disable early stopping for full training
        workers=4,
        device=0,
        plots=True,
        save=True,
        **aug_settings
    )
    
    # Get results path
    results_dir = Path(f"runs/detect/{run_name}")
    
    # Load training results CSV
    results_csv = results_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
    else:
        df = None
    
    # Validate on test set
    best_model = results_dir / "weights" / "best.pt"
    if best_model.exists():
        val_model = YOLO(str(best_model))
        val_results = val_model.val(data=data_yaml, split='test', verbose=False)
        
        metrics = {
            'epochs': epochs,
            'model_size': model_size,
            'augmentation': augmentation,
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'best_model': str(best_model),
        }
    else:
        metrics = {'epochs': epochs, 'error': 'Training failed'}
    
    # Save metrics
    metrics_file = output_dir / f"metrics_{run_name}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics, df, results_dir


def create_training_curves(results_dirs: list, output_path: str):
    """
    Create combined training curves from multiple runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for run_dir, label in results_dirs:
        run_dir = Path(run_dir)
        results_csv = run_dir / "results.csv"
        
        if not results_csv.exists():
            continue
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # Plot metrics
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 0].plot(epochs, df['metrics/mAP50(B)'], label=label, marker='o', markersize=2)
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[0, 1].plot(epochs, df['metrics/mAP50-95(B)'], label=label, marker='o', markersize=2)
        if 'train/box_loss' in df.columns:
            axes[1, 0].plot(epochs, df['train/box_loss'], label=label, marker='o', markersize=2)
        if 'val/box_loss' in df.columns:
            axes[1, 1].plot(epochs, df['val/box_loss'], label=label, marker='o', markersize=2)
    
    axes[0, 0].set_title('mAP@50 vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('mAP@50')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('mAP@50-95 vs Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP@50-95')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Box Loss vs Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Box Loss vs Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Training curves saved to: {output_path}")
    plt.close()


def create_comparison_bar_chart(metrics_list: list, output_path: str):
    """
    Create bar chart comparing different training runs.
    """
    labels = [f"{m['epochs']}ep\n{m.get('augmentation', 'N/A')}" for m in metrics_list]
    
    metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [m.get(metric, 0) for m in metrics_list]
        offset = (i - len(metrics_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Training Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive training with graphs")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                       help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, nargs='+', default=[50, 100, 150],
                       help="List of epoch counts to train")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size")
    parser.add_argument("--augmentation", type=str, default="balanced",
                       choices=["balanced", "aggressive", "none"],
                       help="Augmentation level")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="training_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    all_metrics = []
    all_dirs = []
    
    for epoch_count in args.epochs:
        metrics, df, results_dir = train_and_evaluate(
            data_yaml=args.data,
            epochs=epoch_count,
            model_size=args.model,
            batch_size=args.batch,
            output_dir=args.output,
            augmentation=args.augmentation
        )
        all_metrics.append(metrics)
        all_dirs.append((results_dir, f"{epoch_count} epochs"))
        
        print(f"\n{epoch_count} epochs: mAP50={metrics.get('mAP50', 0):.4f}")
    
    # Create graphs
    output_dir = Path(args.output)
    create_training_curves(all_dirs, str(output_dir / "training_curves.png"))
    create_comparison_bar_chart(all_metrics, str(output_dir / "comparison_chart.png"))
    
    # Save all metrics
    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
