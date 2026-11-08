"""Grad-CAM (Gradient-weighted Class Activation Mapping) Explainer.

Produces visual explanations highlighting lung regions influencing model decisions.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F


class GradCAMExplainer:
    """Generate Grad-CAM heatmaps from MobileNetV2's final convolutional layer."""

    def __init__(self, classifier_engine) -> None:
        self.engine = classifier_engine
        self.model = classifier_engine.model
        self.device = classifier_engine.device
        self.class_names = classifier_engine.class_names

    def explain(
        self,
        image: Union[str, Path, Image.Image],
        target_class: Optional[int] = None,
        alpha: float = 0.45
    ) -> Tuple[np.ndarray, Image.Image, int]:
        """Compute Grad-CAM heatmap and blended overlay image.

        Args:
            image: input chest X-ray (path or PIL Image).
            target_class: class index to explain (None = explain top predicted class).
            alpha: blending transparency for heatmap overlay [0.0 - 1.0].

        Returns:
            heatmap (np.ndarray): 2D float array [0, 1] of size (H, W).
            overlay_image (PIL.Image): RGB image of the X-ray with colored attention overlay.
            selected_class (int): index of the explained class.
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Convert to model input tensor
        x = self.engine.transform(pil_image).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        self.model.eval()

        captured = {}

        def capture_hook(_module, _inputs, output_tensor):
            captured["features"] = output_tensor

        # Hook to the last feature layer of MobileNetV2
        handle = self.model.features[-1].register_forward_hook(capture_hook)

        try:
            with torch.enable_grad():
                logits = self.model(x)
                if target_class is None:
                    selected_class = int(logits.argmax(dim=1).item())
                else:
                    selected_class = int(target_class)

                features = captured["features"]
                # Compute gradient of the target class score w.r.t. feature map
                gradients = torch.autograd.grad(logits[0, selected_class], features)[0]
                # Global average pooling on gradients
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                # Weighted combination of feature maps + ReLU
                cam = (weights * features).sum(dim=1, keepdim=True).relu()
                # Upsample CAM to input image resolution
                cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam[0, 0].detach().cpu().numpy()

                max_val = float(cam.max())
                cam = (cam / max_val) if max_val > 0 else np.zeros_like(cam)
        finally:
            handle.remove()

        # Generate blended overlay image
        orig_resized = pil_image.resize((self.engine.image_size, self.engine.image_size), Image.Resampling.BILINEAR)
        orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

        # Apply Jet colormap
        try:
            colormap = plt.get_cmap("jet")
        except AttributeError:
            colormap = cm.get_cmap("jet")
        colored_cam = colormap(cam)[:, :, :3]

        overlay = (1.0 - alpha) * orig_np + alpha * colored_cam
        overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

        return cam, overlay_pil, selected_class
