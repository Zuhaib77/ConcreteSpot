"""
Dataset Downloader for Specialized Crack and Corrosion Detection
Downloads from Roboflow Universe for 90%+ per-class accuracy training.
"""
import os
from pathlib import Path


def download_crack_datasets():
    """Download crack detection datasets from Roboflow."""
    from roboflow import Roboflow
    
    output_dir = Path("dataset/dataset/crack_specialist")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Crack Detection Datasets")
    print("=" * 60)
    
    # Initialize Roboflow
    rf = Roboflow()
    
    # Dataset 1: Concrete Crack (4182 images)
    print("\n1. Downloading Concrete Crack dataset...")
    try:
        project = rf.workspace("yolo-qgdqb").project("concrete-crack-gnchb")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "concrete_crack"))
        print(f"   Downloaded to: {output_dir / 'concrete_crack'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Dataset 2: Crack Detection (2769 images)
    print("\n2. Downloading Crack Detection v3 dataset...")
    try:
        project = rf.workspace("yolo-concrete-damage-detection").project("crack-detection-x2zvg")
        dataset = project.version(3).download("yolov8", location=str(output_dir / "crack_detection_v3"))
        print(f"   Downloaded to: {output_dir / 'crack_detection_v3'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Dataset 3: Pavement Crack (larger, road cracks)
    print("\n3. Downloading Pavement Crack dataset...")
    try:
        project = rf.workspace("research-btupx").project("pavement-crack-cz7me")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "pavement_crack"))
        print(f"   Downloaded to: {output_dir / 'pavement_crack'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Crack datasets download complete!")
    return output_dir


def download_corrosion_datasets():
    """Download corrosion/rust detection datasets from Roboflow."""
    from roboflow import Roboflow
    
    output_dir = Path("dataset/dataset/corrosion_specialist")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Corrosion/Rust Detection Datasets")
    print("=" * 60)
    
    rf = Roboflow()
    
    # Dataset 1: Metal Corrosion
    print("\n1. Downloading Metal Corrosion dataset...")
    try:
        project = rf.workspace("yolov11-lqoxu").project("metal-corrosion-bsykj")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "metal_corrosion"))
        print(f"   Downloaded to: {output_dir / 'metal_corrosion'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Dataset 2: Rust Detection
    print("\n2. Downloading Rust Detection dataset...")
    try:
        project = rf.workspace("rust-detection-ai9yz").project("rust-detection-szhxy")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "rust_detection"))
        print(f"   Downloaded to: {output_dir / 'rust_detection'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Dataset 3: Corrosion on Concrete
    print("\n3. Downloading Corrosion on Structures dataset...")
    try:
        project = rf.workspace("corrosion-ldndc").project("corrosion-j1iog")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "corrosion_structures"))
        print(f"   Downloaded to: {output_dir / 'corrosion_structures'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Corrosion datasets download complete!")
    return output_dir


def download_rebar_datasets():
    """Download exposed rebar detection datasets."""
    from roboflow import Roboflow
    
    output_dir = Path("dataset/dataset/rebar_specialist")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Exposed Rebar Detection Datasets")
    print("=" * 60)
    
    rf = Roboflow()
    
    # Dataset 1: Rebar Exposure
    print("\n1. Downloading Rebar Exposure dataset...")
    try:
        project = rf.workspace("concrete-damage-i2d3t").project("rebar-exposure")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "rebar_exposure"))
        print(f"   Downloaded to: {output_dir / 'rebar_exposure'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Dataset 2: HRCDS (crack, corrosion, exposed rebar, spalling)
    print("\n2. Downloading HRCDS dataset...")
    try:
        project = rf.workspace("hrcds-oqojr").project("hrcds")
        dataset = project.version(1).download("yolov8", location=str(output_dir / "hrcds"))
        print(f"   Downloaded to: {output_dir / 'hrcds'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Rebar datasets download complete!")
    return output_dir


if __name__ == "__main__":
    print("=" * 60)
    print("ConcreteSpot Specialist Dataset Downloader")
    print("=" * 60)
    print("\nThis script will download specialized datasets for:")
    print("  - Crack detection")
    print("  - Corrosion/rust detection")
    print("  - Exposed rebar detection")
    print("\nNote: You need a Roboflow API key. Get one at https://app.roboflow.com/")
    
    # Download all
    download_crack_datasets()
    download_corrosion_datasets()
    download_rebar_datasets()
    
    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)
