#!/usr/bin/env python3
"""
InceptionV3 Training Script for Severity Classification
========================================================
Trains InceptionV3 to classify: minor, moderate, severe

Dataset Structure Required:
    dataset/
    ├── train/
    │   ├── minor/      (images of minor damage)
    │   ├── moderate/   (images of moderate damage)
    │   └── severe/     (images of severe damage)
    └── val/
        ├── minor/
        ├── moderate/
        └── severe/
"""

import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models


def train_inception(
    dataset_path: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    dataset_path = Path(dataset_path)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists():
        print(f"Error: Training directory '{train_dir}' does not exist!")
        print("\nExpected structure:")
        print("  dataset/train/minor/")
        print("  dataset/train/moderate/")
        print("  dataset/train/severe/")
        print("  dataset/val/minor/")
        print("  dataset/val/moderate/")
        print("  dataset/val/severe/")
        return
    
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    print("\nLoading pretrained InceptionV3...")
    model = models.inception_v3(weights='IMAGENET1K_V1')
    
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_acc = 0.0
    output_path = Path("models/inceptionv3_severity.pt")
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"\nStarting training:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            
            loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_acc:.2f}%")
    print(f"   Model saved to: {output_path}")
    print(f"\n   Copy this file to your ConcreteSpot/models/ folder")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train InceptionV3 for severity classification")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    
    train_inception(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr
    )
