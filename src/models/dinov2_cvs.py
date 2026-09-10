"""DINOv2-based CVS classifier adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class DinoV2CVS:
    """DINOv2 CVS classifier adapter."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str,
        threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.threshold = float(threshold)

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-large")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 3)
        nn.init.zeros_(model.classifier.bias)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 3), dtype=torch.int32)

        pixel_values = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).int().cpu()

        return preds

    def predict_with_logits(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        if not images:
            empty = torch.empty((0, 3), dtype=torch.int32)
            return empty, torch.empty((0, 3), dtype=torch.float32)

        pixel_values = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            probs = torch.sigmoid(logits)
            preds = (probs >= self.threshold).int().cpu()

        return preds, logits.detach().cpu()

    def __call__(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self.predict(images)
