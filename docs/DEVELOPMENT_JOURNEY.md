# ConcreteSpot Development Journey

## Project Overview

**ConcreteSpot** is an offline concrete damage classification application that uses computer vision to detect and classify damage in concrete structures. This document chronicles the development process from concept to working MVP.

---

## Day 1: Requirements & Architecture

### Initial Requirements Gathering

The project began with a comprehensive requirements gathering session covering:

- **Target Users**: Civil engineers, field inspectors, maintenance teams
- **Core Functionality**: Image/video analysis for concrete damage detection
- **Platform Support**: Windows 10+ and Ubuntu 22.04+
- **Key Constraint**: Fully offline operation

### Technology Stack Decision

| Component | Technology | Rationale |
|-----------|------------|-----------|
| AI/Detection | YOLOv8 (Ultralytics) | Fast, accurate object detection |
| AI/Classification | InceptionV3 (PyTorch) | Proven image classification |
| GUI Framework | PySide6 | LGPL license, Qt6 features |
| Database | SQLite | Embedded, zero-config, offline |
| Reporting | ReportLab | Pure Python PDF generation |
| Packaging | PyInstaller | Cross-platform executables |

### Architecture Design

We chose a **two-stage inference pipeline**:
1. **Stage 1 (YOLOv8)**: Detect damage regions with bounding boxes
2. **Stage 2 (InceptionV3)**: Classify severity of each detected region

This approach provides both localization (WHERE is the damage) and classification (HOW severe is it).

---

## Day 2: Core Implementation

### Project Structure

Created a modular architecture separating concerns:

```
src/
├── app/          # UI layer (PySide6)
├── core/         # AI inference pipeline
├── data/         # Database & storage
├── reports/      # PDF generation
└── styles/       # Windows 7 theming
```

### AI Pipeline Development

**Detector (YOLOv8)**:
- Wraps Ultralytics YOLO for easy model loading
- Supports custom trained models or pre-trained fallback
- GPU acceleration via CUDA when available

**Classifier (InceptionV3)**:
- PyTorch implementation with custom head for 3 classes
- Classifies damage crops into: Minor, Moderate, Severe
- Preprocessing pipeline matches ImageNet standards

**Pipeline Orchestration**:
- Combines detector and classifier in single interface
- Batch processing with progress callbacks
- Video frame extraction and analysis

### Data Layer

**Database Schema**:
```sql
analyses: id, timestamp, source_type, source_path, counts
detections: id, analysis_id, damage_type, severity, bbox
```

**Storage Manager**:
- Date-based folder organization
- UUID-based file naming to prevent conflicts
- Automatic directory creation

---

## Day 3: User Interface

### Design Philosophy

Implemented a **Windows 7 Aero-inspired** theme:
- Gradient backgrounds and buttons
- 3D beveled effects
- Classic gray color scheme (#f0f0f0)
- Segoe UI typography

### UI Components Built

1. **Main Window**: Tab-based navigation with status bar
2. **Image Viewer**: Scalable display with detection overlay
3. **Single Image Tab**: Upload → Analyze → Save → Report workflow
4. **Batch Tab**: Folder/file selection, progress tracking
5. **Video Tab**: Frame interval control, slider navigation
6. **History Tab**: Past analyses with preview and delete

### Threading Model

All inference runs in background threads to keep UI responsive:
- `AnalysisWorker` for single images
- `BatchWorker` for multiple images
- `VideoWorker` for video processing

---

## Day 4: Reporting & Polish

### PDF Report Generation

Used ReportLab to create professional reports including:
- Analysis metadata (ID, timestamp, source)
- Embedded annotated image
- Summary statistics table
- Detailed detection list with color-coded severity

### Quality Improvements

- Error handling throughout the pipeline
- User-friendly error dialogs
- Automatic GPU detection with status display
- Proper resource cleanup on exit

---

## Day 5: Integration & Testing

### Dependency Installation

Required packages:
- PySide6 (GUI)
- torch, torchvision (PyTorch)
- ultralytics (YOLOv8)
- opencv-python (image processing)
- reportlab (PDF generation)

### Platform Compatibility

- Resolved Qt xcb plugin issue on Linux
- Tested virtual environment isolation
- Verified GPU detection on CUDA systems

---

## Technical Decisions & Trade-offs

### Why Two-Stage Pipeline?

**Pros**:
- Separate training data for detection vs. classification
- Can upgrade either model independently
- Better interpretability of results

**Cons**:
- Slightly slower than single-stage
- Requires two models instead of one

### Why SQLite?

**Pros**:
- Zero configuration (no database server)
- Single file backup
- Perfect for offline requirements

**Cons**:
- Limited concurrent write access
- Not suitable for multi-user scenarios

### Why Windows 7 Style?

User preference for retro, minimal aesthetic. PySide6's stylesheet system made this straightforward to implement without external dependencies.

---

## Lessons Learned

1. **Virtual Environment Isolation**: Essential for reproducible deployment
2. **Qt Plugin Path**: OpenCV's bundled Qt can conflict with PySide6
3. **Background Threads**: Critical for keeping GUI responsive during inference
4. **Modular Design**: Separating AI, data, and UI layers enables easier testing

---

## Future Roadmap

### v1.1.0 (Planned)
- [ ] PyInstaller packaging for Windows/Linux
- [ ] Custom model training pipeline
- [ ] Export results to CSV/Excel

### v2.0.0 (Future)
- [ ] Live camera feed support
- [ ] Annotation tools for labeling
- [ ] Plugin system for new damage types
- [ ] Multi-language support

---

## Contributors

**Solo Developer Project**
Developed as an offline damage classification tool for concrete structure inspection.

---

*Document Version: 1.0*  
*Last Updated: January 2026*
