"""
Dataset Analyzer and Organizer
Analyzes all datasets, assesses quality, and organizes by damage type
"""
from pathlib import Path
import json
import shutil
from collections import Counter
from tqdm import tqdm

BASE = Path("dataset/dataset")

def analyze_datasets():
    """Analyze all dataset folders and return quality assessment."""
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    results = []
    
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        
        info = {
            'name': d.name,
            'path': d,
            'images': 0,
            'labels': 0,
            'format': 'unknown',
            'classes': {},
            'quality': 'unknown',
            'recommendation': 'REVIEW'
        }
        
        # Count images
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
            info['images'] += len(list(d.rglob(ext)))
        
        # Count labels
        txt_files = list(d.rglob('*.txt'))
        info['labels'] = len([f for f in txt_files 
                             if f.stem not in ['README', 'classes', 'README.dataset', 'README.roboflow']])
        
        # Check format
        yaml_files = list(d.rglob('data.yaml'))
        json_files = list(d.rglob('*.json'))
        
        if yaml_files:
            info['format'] = 'YOLO'
            try:
                with open(yaml_files[0]) as f:
                    content = f.read()
                    if 'names:' in content:
                        for line in content.split('\n'):
                            if 'names:' in line:
                                info['class_names'] = line.strip()
            except:
                pass
        elif json_files:
            info['format'] = 'JSON'
        
        # Sample class distribution
        sample_labels = list(d.rglob('*/labels/*.txt'))[:200]
        if not sample_labels:
            sample_labels = [f for f in txt_files if 'label' in str(f).lower()][:200]
        
        class_counts = Counter()
        valid_labels = 0
        for f in sample_labels:
            try:
                with open(f) as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_counts[int(parts[0])] += 1
                            valid_labels += 1
            except:
                pass
        
        info['classes'] = dict(class_counts)
        info['valid_labels'] = valid_labels
        
        # Quality assessment
        if info['images'] > 0 and info['labels'] > 0:
            label_ratio = info['labels'] / info['images']
            if label_ratio > 0.8 and valid_labels > 100:
                info['quality'] = 'HIGH'
                info['recommendation'] = 'KEEP'
            elif label_ratio > 0.5 and valid_labels > 50:
                info['quality'] = 'MEDIUM'
                info['recommendation'] = 'KEEP'
            else:
                info['quality'] = 'LOW'
                info['recommendation'] = 'REMOVE'
        elif info['images'] > 0 and info['labels'] == 0:
            info['quality'] = 'UNLABELED'
            info['recommendation'] = 'REMOVE'
        else:
            info['quality'] = 'EMPTY'
            info['recommendation'] = 'REMOVE'
        
        results.append(info)
        
        # Print summary
        print(f"\n{info['name']}")
        print(f"  Images: {info['images']:,} | Labels: {info['labels']:,} | Format: {info['format']}")
        print(f"  Quality: {info['quality']} | Recommendation: {info['recommendation']}")
        if info['classes']:
            print(f"  Classes: {info['classes']}")
    
    return results


def main():
    results = analyze_datasets()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    keep = [r for r in results if r['recommendation'] == 'KEEP']
    remove = [r for r in results if r['recommendation'] == 'REMOVE']
    
    print("\n✅ KEEP:")
    for r in keep:
        print(f"  - {r['name']} ({r['images']:,} images, {r['quality']})")
    
    print("\n❌ REMOVE:")
    for r in remove:
        print(f"  - {r['name']} ({r['images']:,} images, {r['quality']})")
    
    # Calculate space
    total_keep = sum(r['images'] for r in keep)
    total_remove = sum(r['images'] for r in remove)
    
    print(f"\nTotal to keep: {total_keep:,} images")
    print(f"Total to remove: {total_remove:,} images")


if __name__ == "__main__":
    main()
