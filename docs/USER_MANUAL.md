# ConcreteSpot User Manual

**Version 1.0.0**

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Single Image Analysis](#single-image-analysis)
6. [Batch Processing](#batch-processing)
7. [Video Analysis](#video-analysis)
8. [Viewing History](#viewing-history)
9. [Generating Reports](#generating-reports)
10. [Understanding Results](#understanding-results)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

**ConcreteSpot** is a desktop application for detecting and classifying damage in concrete structures using artificial intelligence. It can identify:

- **Cracks**: Linear fractures in concrete surfaces
- **Spalling**: Flaking, peeling, or chipping of concrete

Each detection is classified by severity:
- 🟢 **Minor**: Cosmetic damage, no immediate concern
- 🟡 **Moderate**: Requires monitoring or planned repair
- 🔴 **Severe**: Immediate attention recommended

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| OS | Windows 10+ or Ubuntu 22.04+ |
| RAM | 8 GB |
| Storage | 5 GB free space |
| Python | 3.10 or higher |

### Recommended (for GPU acceleration)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA with CUDA support |
| VRAM | 4 GB+ |
| RAM | 16 GB |

---

## Installation

### Step 1: Extract or Clone

Place the ConcreteSpot folder in your preferred location.

### Step 2: Set Up Environment

**Linux:**
```bash
cd ConcreteSpot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
cd ConcreteSpot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Add Models (Optional)

For best results, place trained models in the `models/` folder:
- `yolov8_concrete.pt` - Detection model
- `inceptionv3_severity.pt` - Severity classification

---

## Getting Started

### Launching the Application

**Linux:**
```bash
source venv/bin/activate
cd src
python main.py
```

**Windows:**
```cmd
venv\Scripts\activate
cd src
python main.py
```

### Main Interface

The application has four tabs:

| Tab | Purpose |
|-----|---------|
| **Single Image** | Analyze one image at a time |
| **Batch Processing** | Analyze multiple images from a folder |
| **Video Analysis** | Analyze video files frame by frame |
| **History** | View past analyses and generate reports |

The status bar at the bottom shows:
- Current status message
- GPU availability

---

## Single Image Analysis

### Step 1: Upload Image

1. Click **"Upload Image"** button
2. Select an image file (JPG, JPEG, PNG, or BMP)
3. The image appears in the viewer

### Step 2: Analyze

1. Click **"Analyze"** button
2. Wait for processing (progress shown in status bar)
3. Detected damage appears as colored boxes on the image

### Step 3: Review Results

The right panel shows:
- **Total Detections**: Count of all damage found
- **Cracks**: Number of crack detections
- **Spalling**: Number of spalling detections
- **Detection Table**: Detailed list with type, severity, confidence, and location

### Step 4: Save & Report

- **Save Results**: Stores analysis to database and copies image
- **Generate PDF**: Creates a detailed report document
- **Clear**: Reset the viewer for a new image

---

## Batch Processing

### Selecting Files

**Option A - Select Folder:**
1. Click **"Select Folder"**
2. Choose a folder containing images
3. All supported images are automatically listed

**Option B - Select Files:**
1. Click **"Select Files"**
2. Select multiple image files (Ctrl+Click)
3. Selected files appear in the list

### Processing

1. Click **"Process All"**
2. Progress bar shows completion percentage
3. Each file in the list updates with detection count

### Reviewing Results

- Click any file in the list to preview
- Annotated image shows in the viewer
- Results panel shows counts for selected image

### Saving All

Click **"Save All"** to store all analyses to the database.

---

## Video Analysis

### Loading a Video

1. Click **"Open Video"**
2. Select a video file (MP4, AVI, MKV, or MOV)
3. Video information displays (resolution, FPS, duration)

### Configuring Analysis

**Frame Interval**: Controls how often frames are analyzed
- Lower value = More frames analyzed (slower, more thorough)
- Higher value = Fewer frames analyzed (faster, may miss damage)
- Default: 30 (approximately 1 frame per second at 30fps)

### Processing

1. Set desired frame interval
2. Click **"Process Video"**
3. Progress bar shows current frame

### Reviewing Results

- Use the **slider** to navigate between analyzed frames
- Each frame shows detection overlay
- Frame number and results display below

### Saving

Click **"Save Results"** to store all frame analyses to database.

---

## Viewing History

### Accessing History

1. Click the **"History"** tab
2. Click **"Refresh"** to load latest analyses

### History Table

The table shows:
| Column | Description |
|--------|-------------|
| ID | Unique analysis identifier |
| Date | When analysis was performed |
| Source | Type (image/batch/video) |
| Detections | Total damage count |
| Cracks | Crack count |
| Spalling | Spalling count |

### Previewing Past Analyses

Click any row to preview:
- Original image displays with detection overlay
- Full detection details available

### Managing History

- **Generate Report**: Create PDF for selected analysis
- **Delete**: Remove selected analysis from database

---

## Generating Reports

### From Single Image Tab

After analysis:
1. Click **"Generate PDF"**
2. Report saves to `data/reports/[date]/report_[id].pdf`
3. Confirmation dialog shows file location

### From History Tab

1. Select an analysis
2. Click **"Generate Report"**
3. PDF is created and location displayed

### Report Contents

Each PDF report includes:
- **Header**: Analysis ID and timestamp
- **Source Information**: File path and type
- **Image**: Analyzed image (if available)
- **Summary Table**: Detection counts
- **Detection List**: All detections with severity and location

---

## Understanding Results

### Bounding Box Colors

| Color | Severity | Meaning |
|-------|----------|---------|
| 🟢 Green | Minor | Surface-level, cosmetic damage |
| 🟡 Orange | Moderate | Monitor or plan repair |
| 🔴 Red | Severe | Requires immediate attention |

### Confidence Score

Each detection includes a confidence percentage:
- **90%+**: High confidence detection
- **70-90%**: Moderate confidence
- **Below 70%**: Lower confidence, verify manually

### Location

Coordinates shown as (X, Y) from top-left corner of image.

---

## Troubleshooting

### Application Won't Start

**Linux - Qt Plugin Error:**
```bash
sudo apt install libxcb-cursor0
```

**Windows - DLL Missing:**
Reinstall requirements:
```cmd
pip install --force-reinstall PySide6
```

### No GPU Detected

1. Verify NVIDIA drivers installed
2. Check CUDA compatibility with PyTorch version
3. Status bar will show "CPU Mode" if GPU unavailable

### Slow Performance

- Enable GPU acceleration
- Increase frame interval for video
- Process smaller batches

### No Detections Found

- Verify image quality is sufficient
- Ensure damage is visible in image
- Check if custom models are loaded correctly

### Database Errors

Delete and recreate database:
```bash
rm data/database.db
```
Database will be recreated on next launch.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open image/video |
| Ctrl+S | Save results |
| Ctrl+P | Generate PDF |
| F5 | Refresh history |

---

## File Locations

| Content | Location |
|---------|----------|
| Uploaded images | `data/images/YYYY-MM-DD/` |
| PDF reports | `data/reports/YYYY-MM-DD/` |
| Database | `data/database.db` |
| Models | `models/` |

---

## Getting Help

For issues or feature requests, please refer to the project repository or contact the development team.

---

*ConcreteSpot v1.0.0 - User Manual*  
*Last Updated: January 2026*
