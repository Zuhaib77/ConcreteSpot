# Best Per-Class Performance Record

> **CRITICAL FILE FOR ENSEMBLE MODEL**
> This file tracks the best mAP@50 achieved for each class across ALL training runs.
> Use this to select the best model/dataset per class for the final ensemble.

---

## 🏆 BEST RESULTS SUMMARY (Use These for Ensemble)

| Class | Best mAP@50 | Model/Version | Dataset | Weights Path |
|-------|-------------|---------------|---------|--------------|
| **crack** | **99.5%** | V3 | pure_quality (SDNET2018) | `runs/detect/models/v3_pure_quality/train/weights/best.pt` |
| **efflorescence** | **99.1%** | V4 | organized_v4 | `runs/detect/models/v4_organized/train/weights/best.pt` |
| **spalling** | **90%** | Original | Pre-V1 training | *Need to locate* |
| **exposed_rebar** | **80%+** | Original | Pre-V1 training | *Need to locate* |
| **corrosion** | **51.5%** | V3 | pure_quality | `runs/detect/models/v3_pure_quality/train/weights/best.pt` |

---

## 📊 DETAILED PER-VERSION RESULTS

### V1: 68K Balanced Dataset
- **Overall mAP@50:** 47.0%
- **Dataset:** balanced_95plus (68,000 images)
- **Weights:** `models/v1_68k_balanced_100ep/best.pt`

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| exposed_rebar | 69.6% | 61.7% |
| crack | 65.9% | 55.2% |
| corrosion | 37.6% | 27.4% |
| efflorescence | 32.1% | 25.4% |
| spalling | 29.6% | 19.6% |

---

### V2: 11K Curated Dataset
- **Overall mAP@50:** 60.3%
- **Dataset:** curated_quality (11,000 images)
- **Weights:** `runs/detect/models/v2_curated_quality/train/weights/best.pt`

| Class | mAP@50 | Notes |
|-------|--------|-------|
| efflorescence | **91.4%** | 🏆 Best efflorescence before V4 |
| crack | ~60% | Good |
| spalling | ~50% | Moderate |
| corrosion | ~45% | Moderate |
| exposed_rebar | ~40% | Low |

---

### V3: Pure Quality Dataset (SDNET + CODEBRIM)
- **Overall mAP@50:** 65.7%
- **Dataset:** pure_quality_v3 (7,540 images)
- **Weights:** `runs/detect/models/v3_pure_quality/train/weights/best.pt`

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| crack | **99.5%** | 99.5% | 🏆 Best overall |
| efflorescence | 82.4% | 82.3% |
| spalling | 54.6% | 54.4% |
| corrosion | 51.5% | 51.4% |
| exposed_rebar | 40.6% | 40.6% |

---

### V4: Organized Dataset (Fixed Class IDs)
- **Overall mAP@50:** 62.8%
- **Dataset:** organized_v4 (26,149 images)
- **Weights:** `runs/detect/models/v4_organized/train/weights/best.pt`

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| efflorescence | **99.1%** | 97.0% | 🏆 Best efflorescence |
| crack | 81.7% | 60.5% |
| spalling | 57.7% | 46.0% |
| corrosion | 42.1% | 20.6% |
| exposed_rebar | 33.5% | 22.0% |

---

### V5: dacl10k Dataset
- **Overall mAP@50:** *Training in progress*
- **Dataset:** dacl10k_v5 (6,990 images)
- **Weights:** `runs/detect/models/v5_dacl10k/train/weights/best.pt`

| Class | mAP@50 | Notes |
|-------|--------|-------|
| crack | ? | Pending |
| efflorescence | ? | Pending |
| spalling | ? | Pending |
| corrosion | ? | Pending |
| exposed_rebar | ? | Pending |

---

### V6: Semi-Auto Dataset
- **Overall mAP@50:** *Pending*
- **Dataset:** semiauto_v6 (26,149 images)
- **Weights:** `runs/detect/models/v6_semiauto/train/weights/best.pt`

| Class | mAP@50 | Notes |
|-------|--------|-------|
| crack | ? | Pending |
| efflorescence | ? | Pending |
| spalling | ? | Pending |
| corrosion | ? | Pending |
| exposed_rebar | ? | Pending |

---

### Original/Pre-V1 Training (From User Memory)
- **Spalling:** 90% mAP@50
- **Exposed Rebar:** 80%+ mAP@50
- **Notes:** Need to locate these weights or retrain to achieve similar results

---

## 🎯 ENSEMBLE STRATEGY

### Best Model Per Class
| Class | Use Model | Reason |
|-------|-----------|--------|
| crack | V3 | 99.5% - highest by far |
| efflorescence | V4 | 99.1% - highest |
| spalling | Original/Specialist | 90% - need to locate or retrain |
| exposed_rebar | Original/Specialist | 80% - need to locate or retrain |
| corrosion | TBD (V5/V6?) | 51.5% - needs improvement |

### Priority Actions
1. ✅ Use V3 for crack detection
2. ✅ Use V4 for efflorescence detection
3. ⚠️ Locate original training for spalling/rebar or train specialist
4. ⚠️ Improve corrosion to 80%+ (main blocker for 95% overall)

---

## 📁 MODEL WEIGHTS LOCATIONS

```
runs/detect/models/
├── v2_curated_quality/train/weights/best.pt  # Efflorescence 91.4%
├── v3_pure_quality/train/weights/best.pt     # Crack 99.5%
├── v4_organized/train/weights/best.pt        # Efflorescence 99.1%
├── v5_dacl10k/train/weights/best.pt          # Training...
└── v6_semiauto/train/weights/best.pt         # Pending

models/
├── v1_68k_balanced_100ep/best.pt             # Baseline
└── v4_organized/                             # Backup copy
```

---

*Last Updated: 2026-02-04*
