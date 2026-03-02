# GCP Training Quick Commands
# Run these from PowerShell on your LOCAL machine

# ============================================
# STEP 1: Create VM (run once)
# ============================================
gcloud compute instances create yolo-training `
  --zone=us-central1-a `
  --machine-type=a2-highgpu-1g `
  --accelerator=type=nvidia-tesla-a100,count=1 `
  --image-family=pytorch-2-1-cu121-debian-11-py310 `
  --image-project=deeplearning-platform-release `
  --boot-disk-size=200GB `
  --boot-disk-type=pd-ssd `
  --maintenance-policy=TERMINATE

# If A100 quota denied, use V100:
gcloud compute instances create yolo-training `
  --zone=us-west1-b `
  --machine-type=n1-highmem-8 `
  --accelerator=type=nvidia-tesla-v100,count=1 `
  --image-family=pytorch-2-1-cu121-debian-11-py310 `
  --image-project=deeplearning-platform-release `
  --boot-disk-size=200GB `
  --boot-disk-type=pd-ssd `
  --maintenance-policy=TERMINATE

# ============================================
# STEP 2: Upload dataset
# ============================================
# First, zip your dataset locally
Compress-Archive -Path "dataset\dataset\balanced_95plus" -DestinationPath "balanced_95plus.zip"

# Upload to VM
gcloud compute scp balanced_95plus.zip yolo-training:~/ --zone=us-central1-a

# ============================================
# STEP 3: SSH and train
# ============================================
gcloud compute ssh yolo-training --zone=us-central1-a

# ============================================
# STEP 4: Download results (after training)
# ============================================
gcloud compute scp yolo-training:~/concretespot/runs/yolov8s_a100/weights/best.pt . --zone=us-central1-a
gcloud compute scp --recurse yolo-training:~/concretespot/runs/yolov8s_a100 ./gcp_results/ --zone=us-central1-a

# ============================================
# STEP 5: STOP VM (save credits!)
# ============================================
gcloud compute instances stop yolo-training --zone=us-central1-a

# Start VM again
gcloud compute instances start yolo-training --zone=us-central1-a

# Delete VM when done
gcloud compute instances delete yolo-training --zone=us-central1-a
