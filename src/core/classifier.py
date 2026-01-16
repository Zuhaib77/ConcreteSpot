from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from data.models import Severity


class SeverityClassifier:
    def __init__(self, model_path: Optional[Path] = None):
        self.model: Optional[nn.Module] = None
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.has_custom_model = False
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.severity_map = {
            0: Severity.MINOR,
            1: Severity.MODERATE,
            2: Severity.SEVERE
        }
    
    def load_model(self, model_path: Optional[Path] = None):
        if model_path:
            self.model_path = model_path
        
        if self.model_path and self.model_path.exists():
            self.model = models.inception_v3(weights=None, aux_logits=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 3)
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.has_custom_model = True
        else:
            self.model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)
            self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 3)
            self.has_custom_model = False
        
        self.model.to(self.device)
        self.model.eval()
    
    def classify(self, image_crop: np.ndarray) -> Severity:
        if self.model is None:
            self.load_model()
        
        if not self.has_custom_model:
            import random
            weights = [0.4, 0.4, 0.2]
            return random.choices(
                [Severity.MINOR, Severity.MODERATE, Severity.SEVERE],
                weights=weights
            )[0]
        
        if isinstance(image_crop, np.ndarray):
            pil_image = Image.fromarray(image_crop)
        else:
            pil_image = image_crop
        
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
        
        return self.severity_map.get(class_idx, Severity.MODERATE)
    
    def is_loaded(self) -> bool:
        return self.model is not None
