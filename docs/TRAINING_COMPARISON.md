# Progressive Training Comparison Report

**ConcreteSpot - YOLOv8 Epoch Comparison**

*Generated: January 17, 2026*

---

## Summary

| Model | Epochs | mAP@50 | mAP@50-95 | Precision | Recall | Improvement |
|-------|--------|--------|-----------|-----------|--------|-------------|
| Baseline | 100 | 68.18% | 49.11% | 71.20% | 62.72% | - |
| Exp-1 | 150 | 72.03% | 52.02% | 74.16% | 68.32% | +5.6% |
| **Exp-2** | **200** | **72.61%** | **53.49%** | **69.95%** | **72.17%** | **+6.5%** |

---

## Per-Class Performance

### 100 Epochs
| Class | mAP@50 |
|-------|--------|
| Crack | 32.1% |
| Spalling | 98.0% |
| Corrosion | 78.1% |
| Exposed Rebar | 64.6% |

### 150 Epochs
| Class | mAP@50 | Change from 100ep |
|-------|--------|-------------------|
| Crack | 34.0% | +1.9% |
| Spalling | 97.7% | -0.3% |
| Corrosion | 82.3% | +4.2% |
| Exposed Rebar | 74.1% | **+9.5%** |

### 200 Epochs
| Class | mAP@50 | Change from 100ep |
|-------|--------|-------------------|
| Crack | 34.0% | +1.9% |
| Spalling | 98.6% | +0.6% |
| Corrosion | 78.7% | +0.6% |
| Exposed Rebar | 79.2% | **+14.6%** |

---

## Key Observations

1. **Best Model**: **200 epochs** achieves highest mAP@50 at **72.61%** (+6.5% over baseline)
2. **Exposed Rebar**: Massive improvement at **+14.6%** from 100 to 200 epochs
3. **Recall improved significantly**: 72.17% at 200ep vs 62.72% at 100ep (+9.5%)
4. **Spalling**: Consistently high at 97-98% across all runs
5. **Crack**: Remains challenging at ~34% (complex class with high variation)

---

## Model Paths

| Epochs | Model Path |
|--------|------------|
| 100 | `models/yolov8_concrete.pt` |
| 150 | `runs/detect/yolov8n_150ep_balanced2/weights/best.pt` |
| 200 | `runs/detect/yolov8n_200ep_balanced3/weights/best.pt` |

---

## Training Loss Comparison

| Epochs | Final Box Loss | Final Cls Loss |
|--------|----------------|----------------|
| 100 | ~1.50 | ~1.50 |
| 150 | 1.24 | 1.08 |
| 200 | 1.23 | 1.03 |

---

## Dataset

- **Training**: 3334 images
- **Validation**: 375 images  
- **Test**: 238 images (used for evaluation)
- **Classes**: crack, spalling, corrosion, exposed_rebar

---

## Recommendations

1. ✅ Use **200-epoch model** as production model (best mAP50 and recall)
2. Consider data augmentation for crack class (underperforming)
3. Optional: Try 250/300 epochs to see if further improvement is possible
