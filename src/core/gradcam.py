"""
YOLOv8 Activation Heatmap Visualization
Uses EigenCAM approach - no gradients needed, more stable for detection models.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from typing import Optional, List, Tuple


class YOLOv8CAM:
    """
    Activation-based CAM for YOLOv8 models.
    Uses feature map activations to show important regions.
    """
    
    CLASS_COLORS = {
        0: (255, 50, 50),     # crack - red
        1: (50, 255, 50),     # spalling - green  
        2: (255, 165, 0),     # corrosion - orange
        3: (50, 50, 255),     # exposed_rebar - blue
    }
    
    def __init__(self, model_path: str):
        """
        Initialize CAM visualizer.
        
        Args:
            model_path: Path to YOLOv8 model (.pt file)
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Loaded model with classes: {self.class_names}")
        
        # Hook storage
        self.activations = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on backbone layers to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations.append(output.detach())
            return hook
        
        # Register hooks on multiple backbone layers
        # These capture features at different scales
        target_layers = [9, 12, 15, 18, 21]  # P3, P4, P5 + neck layers
        
        for idx in target_layers:
            try:
                layer = self.model.model.model[idx]
                layer.register_forward_hook(get_activation(f'layer_{idx}'))
            except (IndexError, AttributeError):
                pass
    
    def generate_cam(self, image_path: str, conf_threshold: float = 0.25) -> dict:
        """
        Generate activation heatmap for an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Detection confidence threshold
            
        Returns:
            dict with visualization results
        """
        # Clear previous activations
        self.activations = []
        
        # Load original image  
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'Could not load image: {image_path}'}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Run inference (this triggers our hooks)
        results = self.model.predict(image_path, conf=conf_threshold, verbose=False)
        
        # Get detections
        detections = self._parse_detections(results[0])
        
        if not self.activations:
            return {
                'original': image_rgb,
                'overlay': image_rgb,
                'detections': detections,
                'message': 'No activations captured'
            }
        
        # Generate heatmap from activations (EigenCAM approach)
        heatmap = self._generate_heatmap(self.activations, (w, h))
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        # Draw detection boxes
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            color = self.CLASS_COLORS.get(det['class_id'], (255, 255, 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return {
            'original': image_rgb,
            'heatmap': heatmap_colored,
            'overlay': overlay,
            'detections': detections,
            'heatmap_raw': heatmap
        }
    
    def _parse_detections(self, result) -> List[dict]:
        """Parse YOLO results into detection list."""
        detections = []
        for box in result.boxes:
            detections.append({
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        return detections
    
    def _generate_heatmap(self, activations: List[torch.Tensor], target_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate heatmap from activation maps using EigenCAM approach.
        Uses first principal component of the activation channels.
        """
        combined = None
        
        for act in activations:
            # Take batch 0, average across channels
            if len(act.shape) == 4:
                # Shape: [B, C, H, W] -> [H, W]
                feat_map = act[0].mean(dim=0).cpu().numpy()
            else:
                continue
            
            # Resize to target size
            feat_resized = cv2.resize(feat_map, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Combine
            if combined is None:
                combined = feat_resized
            else:
                combined += feat_resized
        
        if combined is None:
            return np.zeros((target_size[1], target_size[0]))
        
        # Normalize
        combined = combined - combined.min()
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def save_visualization(self, result: dict, output_path: str, mode: str = 'comparison'):
        """
        Save visualization to file.
        
        Args:
            result: Result dict from generate_cam()
            output_path: Where to save
            mode: 'comparison' (side-by-side), 'overlay' (just overlay), 'all' (3-panel)
        """
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        if mode == 'comparison':
            # Side-by-side: original + overlay
            vis = np.hstack([result['original'], result['overlay']])
        elif mode == 'overlay':
            vis = result['overlay']
        elif mode == 'all':
            # Three panels: original + heatmap + overlay
            vis = np.hstack([result['original'], result['heatmap'], result['overlay']])
        else:
            vis = result['overlay']
        
        # Convert to BGR for OpenCV
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_bgr)
        print(f"Saved: {output_path}")
    
    def process_directory(self, input_dir: str, output_dir: str, max_images: int = 10):
        """Process multiple images from a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        images = images[:max_images]
        
        print(f"\nProcessing {len(images)} images...")
        
        for img_path in images:
            result = self.generate_cam(str(img_path))
            
            if result['detections']:
                output_file = output_path / f"cam_{img_path.name}"
                self.save_visualization(result, str(output_file))
                
                # Print detection summary
                det_summary = ", ".join([f"{d['class_name']}({d['confidence']:.0%})" 
                                        for d in result['detections']])
                print(f"  {img_path.name}: {det_summary}")


def demo():
    """Demo function to test CAM visualization."""
    print("=" * 60)
    print("YOLOv8 Activation CAM Demo")
    print("=" * 60)
    
    # Setup
    model_path = "runs/detect/yolov8n_200ep_cleaned/weights/best.pt"
    test_dir = "dataset/dataset/images/test"
    output_dir = "gradcam_results"
    
    # Initialize
    cam = YOLOv8CAM(model_path)
    
    # Process images
    cam.process_directory(test_dir, output_dir, max_images=10)
    
    print(f"\n✅ Done! Results saved to: {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single image mode
        image_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "runs/detect/yolov8n_200ep_cleaned/weights/best.pt"
        
        cam = YOLOv8CAM(model_path)
        result = cam.generate_cam(image_path)
        
        output_path = f"gradcam_results/cam_{Path(image_path).name}"
        Path("gradcam_results").mkdir(exist_ok=True)
        cam.save_visualization(result, output_path)
        
        print("\nDetections:")
        for det in result['detections']:
            print(f"  {det['class_name']}: {det['confidence']:.1%}")
    else:
        demo()
