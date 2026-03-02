# Model V1: 68K Balanced Dataset
## YOLOv8s - 100 Epochs

### Training Info
| Property | Value |
|----------|-------|
| Model | YOLOv8s (11.1M params) |
| Dataset | balanced_95plus (68,000 images) |
| Epochs | 100 |
| Batch Size | 16 |
| Image Size | 640×640 |
| Training Time | ~15 hours |

### Final Metrics
| Metric | Value |
|--------|-------|
| **mAP@50** | **47.0%** |
| **mAP@50-95** | **37.9%** |
| Precision | 53.0% |
| Recall | 46.1% |

### Per-Class Performance
| Class | mAP@50 | mAP@50-95 | Status |
|-------|--------|-----------|--------|
| exposed_rebar | **69.6%** | 61.7% | ⭐ Best |
| crack | **65.9%** | 55.2% | ✅ Good |
| corrosion | 37.6% | 27.4% | ⚠️ Needs work |
| efflorescence | 32.1% | 25.4% | ⚠️ Needs work |
| spalling | 29.6% | 19.6% | ❌ Worst |

### Files
- `best.pt` - Best mAP weights
- `last.pt` - Epoch 100 weights
- `results.csv` - Training history

### Notes
- Large dataset (68K) but mixed quality caused confusion
- Spalling dropped from 90% (original) to 30% (here)
- Crack improved from 42% to 66% with more data
- Use as **baseline comparison** for research paper
