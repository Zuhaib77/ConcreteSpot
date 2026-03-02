# GCP A100 Training Setup Guide
## Train YOLOv8 5-10x Faster with $300 Free Credits

---

## 📋 Prerequisites Checklist

- [ ] Google account (personal or university)
- [ ] Credit/Debit card (for verification, won't charge beyond credits)
- [ ] ~5GB dataset upload capability

---

## Phase 1: Get $300 Free Credits (5 minutes)

### Step 1: Create GCP Account
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click **"Get started for free"** or **"Try Free"**
3. Sign in with your Google account
4. Accept terms and conditions

### Step 2: Verify Identity
1. Enter billing information (credit/debit card)
2. **Important:** You will NOT be charged after credits expire
3. Google requires card for verification only
4. Credits expire in **90 days**

### Step 3: Verify Credits
1. Go to **Billing** → **Overview**
2. You should see **$300.00 credit balance**

---

## Phase 2: Request GPU Quota (10-30 minutes)

> ⚠️ **CRITICAL:** By default, new accounts have 0 GPU quota. You must request it!

### Step 1: Go to Quotas
1. Search **"All Quotas"** in GCP search bar
2. Or go to: `IAM & Admin` → `Quotas`

### Step 2: Request A100 Quota
1. Filter by: `GPUs (all regions)`
2. Find: **"NVIDIA A100 GPUs"** or **"NVIDIA V100 GPUs"**
3. Click the service → **"Edit Quotas"**
4. Request limit: **1** (one GPU is enough)
5. Justification: *"Student ML research project - training object detection model"*

### Step 3: Wait for Approval
- Usually approved in **10-30 minutes**
- Check email for confirmation
- If denied, try requesting V100 instead (higher approval rate)

---

## Phase 3: Create VM with GPU (5 minutes)

### Option A: Using Console (Easier)

1. Go to **Compute Engine** → **VM instances**
2. Click **"Create Instance"**
3. Configure:

| Setting | Value |
|---------|-------|
| Name | `yolo-training` |
| Region | `us-central1` (cheapest) |
| Zone | `us-central1-a` |
| Machine type | `a2-highgpu-1g` (for A100) |
| GPU | NVIDIA A100 (1x) |
| Boot disk | **Deep Learning on Linux** |
| Disk size | 200 GB SSD |

4. Click **"Create"**

### Option B: Using Cloud Shell (Advanced)

```bash
# Open Cloud Shell (terminal icon in top-right)

# Create A100 VM
gcloud compute instances create yolo-training \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-2-1-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE

# If A100 quota denied, use V100:
gcloud compute instances create yolo-training \
  --zone=us-west1-b \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-2-1-cu121-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

---

## Phase 4: Connect and Setup Environment (10 minutes)

### Step 1: SSH into VM
```bash
# From Cloud Shell
gcloud compute ssh yolo-training --zone=us-central1-a

# Or click "SSH" button in Console
```

### Step 2: Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip install ultralytics gdown tqdm

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
# Should show: NVIDIA A100-SXM4-40GB
```

### Step 3: Create Training Directory
```bash
mkdir -p ~/concretespot/dataset
cd ~/concretespot
```

---

## Phase 5: Upload Dataset (15-30 minutes)

### Option A: Google Drive (Recommended)

1. Upload `balanced_95plus.zip` to Google Drive
2. Get shareable link → Copy file ID from URL
3. Download on VM:

```bash
# Install gdown for Drive downloads
pip install gdown

# Download from Drive (replace FILE_ID)
gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O balanced_95plus.zip

# Extract
unzip balanced_95plus.zip -d dataset/
```

### Option B: Direct Upload via SCP

```bash
# From your LOCAL machine (PowerShell)
gcloud compute scp --recurse C:\Users\spect\ConcreteSpotDetection\dataset\dataset\balanced_95plus yolo-training:~/concretespot/dataset/ --zone=us-central1-a
```

### Option C: gsutil (Fastest for large files)

```bash
# On LOCAL machine: Upload to Cloud Storage bucket
gsutil -m cp -r dataset/dataset/balanced_95plus gs://your-bucket-name/

# On VM: Download from bucket
gsutil -m cp -r gs://your-bucket-name/balanced_95plus ~/concretespot/dataset/
```

---

## Phase 6: Start Training (5 minutes setup, ~4 hours training)

### Step 1: Create Training Script
```bash
cat << 'EOF' > train.py
from ultralytics import YOLO
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")

# Load model (use last.pt to resume, or yolov8s.pt for fresh start)
model = YOLO("yolov8s.pt")

# Train with A100-optimized settings
results = model.train(
    data="dataset/balanced_95plus/data.yaml",
    epochs=150,
    imgsz=640,
    batch=64,        # A100 can handle larger batches
    workers=8,
    device=0,
    patience=30,
    optimizer="AdamW",
    lr0=0.001,
    cls=1.5,
    project="runs",
    name="yolov8s_a100",
    exist_ok=True,
    save=True,
    save_period=25,
    plots=True
)

print("Training complete!")
print(f"Best model: runs/yolov8s_a100/weights/best.pt")
EOF
```

### Step 2: Run Training in Screen (Persistent)
```bash
# Start screen session (survives SSH disconnect)
screen -S training

# Run training
python train.py

# Detach from screen: Press Ctrl+A, then D
# Reattach later: screen -r training
```

### Step 3: Monitor Training
```bash
# In another SSH session or after reattaching
tail -f runs/yolov8s_a100/results.csv

# Or check GPU usage
nvidia-smi -l 1
```

---

## Phase 7: Download Results (5 minutes)

### After Training Completes:
```bash
# On VM: Zip results
cd ~/concretespot
zip -r results.zip runs/yolov8s_a100/

# Download to local (from LOCAL PowerShell)
gcloud compute scp yolo-training:~/concretespot/results.zip C:\Users\spect\ConcreteSpotDetection\ --zone=us-central1-a
```

---

## Phase 8: Stop VM to Save Credits! ⚠️

### CRITICAL: Stop VM when not training!

```bash
# From Cloud Shell or Console
gcloud compute instances stop yolo-training --zone=us-central1-a

# To restart later
gcloud compute instances start yolo-training --zone=us-central1-a
```

### Delete VM when done:
```bash
gcloud compute instances delete yolo-training --zone=us-central1-a
```

---

## 💰 Cost Estimation

| Resource | Cost/Hour | 150 Epochs Time | Total |
|----------|-----------|-----------------|-------|
| A100 (a2-highgpu-1g) | $3.67 | ~4 hours | ~$15 |
| V100 (n1-highmem-8) | $2.48 | ~8 hours | ~$20 |
| Storage (200GB SSD) | $0.02/hr | 4-8 hours | ~$0.20 |
| **Total** | - | - | **~$15-20** |

**Remaining credits after training: ~$280** 🎉

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU quota denied | Try V100, T4, or different region |
| SSH connection lost | Use `screen` session, reconnect with `screen -r` |
| Out of disk space | Delete old checkpoints, use larger disk |
| CUDA out of memory | Reduce batch size (64 → 32 → 16) |
| Training slow | Check GPU utilization with `nvidia-smi` |

---

## 📞 Quick Reference Commands

```bash
# Start VM
gcloud compute instances start yolo-training --zone=us-central1-a

# Stop VM (SAVE CREDITS!)
gcloud compute instances stop yolo-training --zone=us-central1-a

# SSH into VM
gcloud compute ssh yolo-training --zone=us-central1-a

# Check GPU
nvidia-smi

# Monitor training
tail -f runs/yolov8s_a100/results.csv

# Zip and download results
zip -r results.zip runs/
```
