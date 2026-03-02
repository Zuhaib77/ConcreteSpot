"""
Test Ensemble Detector
Verifies all 5 specialist models load and run correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np
from pathlib import Path

os.chdir(r"C:\Users\spect\ConcreteSpotDetection")

def test_ensemble():
    print("=" * 60)
    print("TESTING ENSEMBLE DETECTOR")
    print("=" * 60)
    
    # Import ensemble detector
    from core.ensemble_detector import EnsembleDetector
    
    # Initialize
    detector = EnsembleDetector()
    
    # Load models
    print("\n[1] Loading specialist models...")
    success = detector.load_models()
    
    if not success:
        print("WARNING: Not all models loaded!")
    
    # Get model info
    info = detector.get_model_info()
    print(f"\nModel Info:")
    print(f"  Device: {info['device']}")
    print(f"  Loaded: {info['total_models']}/{info['expected_models']} models")
    
    # Create test image (or use real one)
    print("\n[2] Testing detection on sample image...")
    
    # Try to find a test image
    test_images = list(Path("datasets").rglob("*.jpg"))[:1]
    if not test_images:
        test_images = list(Path("datasets").rglob("*.png"))[:1]
    
    if test_images:
        test_path = test_images[0]
        print(f"  Using: {test_path}")
        
        image = cv2.imread(str(test_path))
        if image is not None:
            # Run detection
            detections = detector.detect(image)
            
            print(f"\n[3] Detection Results:")
            print(f"  Total detections: {len(detections)}")
            
            # Count by class
            by_class = {}
            for det in detections:
                cls = det.damage_type.value
                by_class[cls] = by_class.get(cls, 0) + 1
            
            for cls, count in by_class.items():
                print(f"    {cls}: {count}")
            
            # Draw and save result
            output = detector.draw_detections(image, detections)
            output_path = "test_ensemble_result.jpg"
            cv2.imwrite(output_path, output)
            print(f"\n  Result saved to: {output_path}")
        else:
            print("  ERROR: Could not load image")
    else:
        print("  No test images found")
    
    print("\n" + "=" * 60)
    print("ENSEMBLE TEST COMPLETE!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    test_ensemble()
