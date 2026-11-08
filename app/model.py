"""Model definition, weights loading, and inference pipeline."""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class PneumoniaClassifier:
    """Production inference engine for Chest X-Ray 3-Class Diagnosis.
    
    Classes:
        - BACTERIAL (Bacterial Pneumonia)
        - NORMAL (Healthy lungs)
        - VIRAL (Viral Pneumonia)
    """

    DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts" / "pneumonia_3class.pt"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = Path(checkpoint_path or self.DEFAULT_CHECKPOINT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self._load_checkpoint()
        self._init_transforms()

    def _load_checkpoint(self) -> None:
        """Load checkpoint weights and architecture parameters."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = ckpt.get("class_names", ["BACTERIAL", "NORMAL", "VIRAL"])
        self.image_size = ckpt.get("image_size", 224)
        self.mean = ckpt.get("mean", [0.485, 0.456, 0.406])
        self.std = ckpt.get("std", [0.229, 0.224, 0.225])

        # Reconstruct MobileNetV2 architecture with 3-class classifier head
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(self.class_names))

        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def _init_transforms(self) -> None:
        """Prepare deterministic evaluation transform pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Convert path or PIL Image to normalized input tensor."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def predict(
        self, image: Union[str, Path, Image.Image]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Run classification on input chest X-ray.

        Returns:
            predicted_class (str): top predicted category ('BACTERIAL', 'NORMAL', or 'VIRAL')
            confidence (float): top probability in range [0, 1]
            scores (dict): full probability distribution across all 3 classes
        """
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        scores = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return predicted_class, confidence, scores

