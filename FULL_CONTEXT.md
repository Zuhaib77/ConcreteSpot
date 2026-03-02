# ConcreteSpot: Project Context & Knowledge Base
**Last Updated:** February 10, 2026

## 📌 Project Overview
**ConcreteSpot** is an AI-powered system for automated concrete damage detection. It utilizes a **Specialist Ensemble Architecture** based on **YOLOv8** to detect 5 key damage types:
1.  **Crack** (High accuracy: ~99%)
2.  **Spalling** (Good accuracy: ~98%)
3.  **Efflorescence** (High accuracy: ~99%)
4.  **Exposed Rebar** (Moderate: ~91%)
5.  **Corrosion** (Challenging: ~74%)

The project targets the **CVR 2026 Conference** and aims to replace manual visual inspection with a real-time (23 FPS) automated solution.

---

## 📂 Key Directory Structure
*   `src/app/`: **Desktop Application** (PySide6). Entry point: `src/main.py` (or `python -m src.app.main`).
*   `research_paper/`: **LaTeX Source** for CVR 2026 submission.
    *   `main.pdf`: **Final compiled paper** (14 pages).
    *   `sections/`: Individual LaTeX chapters.
    *   `figures/`: Generated plots and diagrams.
*   `models/`: Trained weights using `.pt` format.
    *   `codebrim_v1.pt`: **Latest single-model** trained on CODEBRIM (Feb 2026).
    *   `yolov8_concrete.pt`: Legacy baseline.
    *   `specialists/`: Individual class experts (if available).
*   `v2_fresh/`: **New Training Pipeline** (Clean start Feb 2026).
    *   `datasets/CODEBRIM_YOLO/`: Converted dataset.
    *   `train.py`: Training script for recent runs.
    *   `results.csv`: Training metrics.
*   `training/`: Legacy training scripts (V1-V6 experiments).

---

## 🚀 Current Status (Feb 2026)

### 1. Research Paper (READY)
*   **Novelty:** First specialist ensemble for concrete damage; addresses class imbalance via per-class specialization (30% mAP gain over single model).
*   **Visuals:** Flowchart, Confusion Matrix, Dataset Evolution, and Sample Detections added.
*   **Feedback:** All mentor feedback addressed (Reference DOIs, Author name fix, etc.).
*   **File:** `research_paper/main.pdf`

### 2. Model Performance
*   **Ensemble Strategy:** Combines 5 specialist models using **NMS (Non-Maximum Suppression)**.
*   **Latest Training (v2_fresh):** Trained YOLOv8s on **CODEBRIM** (800+ images).
    *   **Strengths:** Exposed Rebar (71%), Spalling (56%).
    *   **Weaknesses:** Crack (15% - unexpected low), Efflorescence (34%).
    *   **Note:** This single model serves as a "backup" or "comparison" to the ensemble.

### 3. Desktop App
*   **Tech Stack:** Python 3.13, PySide6, Ultralytics YOLOv8.
*   **Features:**
    *   Real-time webcam/video inference.
    *   Image/Batch processing.
    *   **Visuals:** Bounding boxes with class-specific colors.
    *   **Model Switching:** Dropdown to select between Ensemble, v2_fresh, or legacy models.

---

## 🛠️ How to Run

### Run the App
```bash
# Activate venv
.\venv\Scripts\activate

# Run App
python -m src.main
# OR
python src/app/main_window.py
```

### Compile Research Paper
```bash
cd research_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Train New Model (CODEBRIM)
```bash
# Configure v2_fresh/train.py then run:
python v2_fresh/train.py
```

---

## 🔮 Future Roadmap
1.  **Corrosion Improvement:** Collect more distinctive samples; corrosion often confused with rust stains/dirt.
2.  **Edge Deployment:** Quantize models (INT8) for mobile devices.
3.  **Severity Classification:** Add "Minor/Moderate/Severe" grading to detections.

---

## 📝 Recent History (Context for AI)
*   **Feb 10, 2026:** Finalized CVR 2026 paper. Fixed LaTeX errors in `results.tex`. Integrated `codebrim_v1.pt` into the app. Generated all paper figures using `generate_paper_figs.py` and `generate_paper_samples.py`.
*   **Jan 2026:** Created `v2_fresh` directory for clean slate training. Downloaded CODEBRIM dataset.
