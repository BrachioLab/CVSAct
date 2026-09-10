"""EndoscapesNet model for coarse organ segmentation (Endoscapes)."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import namedtuple

ModelOutput = namedtuple("ModelOutput", ["logits", "pooler_output"])


class EndoscapesNet(nn.Module):
    """EndoscapesNet for coarse organ segmentation (Endoscapes-Seg50).

    This model segments images into 6 organ/tool classes + background:
    - Background (0)
    - Cystic Plate (1)
    - HC Triangle Dissection (2)
    - Cystic Artery (3)
    - Cystic Duct (4)
    - Gallbladder (5)
    - Tool (6)
    """

    ID2LABEL = {
        0: "Background",
        1: "Cystic Plate",
        2: "HC Triangle Dissection",
        3: "Cystic Artery",
        4: "Cystic Duct",
        5: "Gallbladder",
        6: "Tool",
    }

    LABEL2ID = {v: k for k, v in ID2LABEL.items()}

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 7,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        activation: str = "softmax2d",
    ):
        """Initialize EndoscapesNet.

        Args:
            n_channels: Number of input channels (3 for RGB)
            n_classes: Number of output classes (7 total)
            encoder_name: Name of encoder architecture
            encoder_weights: Pretrained weights to use
            activation: Activation function for output
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
            activation=activation,
        )

    def forward(self, x, return_tuple=True):
        """Forward pass through the network.

        Args:
            x: Input tensor [B, C, H, W]
            return_tuple: If True, return ModelOutput namedtuple

        Returns:
            ModelOutput with logits [B, n_classes, H, W] or just logits
        """
        N, C, H, W = x.shape
        assert H % 32 == 0 and W % 32 == 0, f"Height {H} and Width {W} must be divisible by 32"

        logits = self.unet(x)

        if return_tuple:
            return ModelOutput(logits=logits, pooler_output=None)
        return logits

    def predict(self, x):
        """Get predicted segmentation mask.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Predicted class indices [B, H, W]
        """
        output = self.forward(x, return_tuple=True)
        return output.logits.argmax(dim=1)

    def predict_proba(self, x):
        """Get class probabilities.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Class probabilities [B, n_classes, H, W]
        """
        output = self.forward(x, return_tuple=True)
        return output.logits

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        n_classes: int = 7,
    ) -> "EndoscapesNet":
        """Load EndoscapesNet from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            n_classes: Number of output classes

        Returns:
            Loaded EndoscapesNet model
        """
        model = cls(n_classes=n_classes)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        return model

    def process_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (640, 384),
    ) -> torch.Tensor:
        """Process PIL image for model input.

        Args:
            image: PIL Image
            target_size: Target (width, height) for resizing

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        image = image.resize(target_size, Image.BILINEAR)

        image_np = np.array(image).astype(np.float32) / 255.0

        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=0)
        else:
            image_np = image_np.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        return image_tensor

    def get_segmentation_mask(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (640, 384),
    ) -> np.ndarray:
        """Get segmentation mask for an image.

        Args:
            image: PIL Image
            target_size: Target (width, height) for resizing

        Returns:
            Segmentation mask as numpy array [H, W] with class indices
        """
        input_tensor = self.process_image(image, target_size)

        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            pred_mask = self.predict(input_tensor)

        mask = pred_mask[0].cpu().numpy()
        return mask

    def get_organ_presence(self, mask: np.ndarray, min_pixels: int = 50) -> Dict[str, bool]:
        """Get organ presence from segmentation mask.

        Args:
            mask: Segmentation mask [H, W]
            min_pixels: Minimum pixels for presence detection

        Returns:
            Dictionary mapping organ names to presence boolean
        """
        presence = {}

        for class_id, class_name in self.ID2LABEL.items():
            if class_id == 0:
                continue

            pixel_count = np.sum(mask == class_id)
            presence[class_name] = pixel_count >= min_pixels

        return presence

    def get_bounding_boxes(
        self, mask: np.ndarray, min_pixels: int = 50
    ) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """Get bounding boxes from segmentation mask.

        Args:
            mask: Segmentation mask [H, W]
            min_pixels: Minimum pixels for a valid region

        Returns:
            Dictionary mapping organ names to list of bounding boxes (x1, y1, x2, y2)
        """
        from scipy import ndimage

        bboxes = {}

        for class_id, class_name in self.ID2LABEL.items():
            if class_id == 0:
                continue

            class_mask = (mask == class_id).astype(np.uint8)

            if not class_mask.any():
                continue

            labeled_array, num_features = ndimage.label(class_mask)

            boxes = []
            for component_id in range(1, num_features + 1):
                component_mask = labeled_array == component_id

                if component_mask.sum() < min_pixels:
                    continue

                y_coords, x_coords = np.where(component_mask)

                if len(x_coords) > 0:
                    x_min = int(x_coords.min())
                    x_max = int(x_coords.max())
                    y_min = int(y_coords.min())
                    y_max = int(y_coords.max())

                    boxes.append((x_min, y_min, x_max, y_max))

            if boxes:
                bboxes[class_name] = boxes

        return bboxes

    def get_centroid_points(
        self, mask: np.ndarray, min_pixels: int = 50
    ) -> Dict[str, Tuple[int, int]]:
        """Get centroid points for each organ in the mask.

        Args:
            mask: Segmentation mask [H, W]
            min_pixels: Minimum pixels for a valid region

        Returns:
            Dictionary mapping organ names to (x, y) centroid coordinates
        """
        centroids = {}

        for class_id, class_name in self.ID2LABEL.items():
            if class_id == 0:
                continue

            class_mask = mask == class_id

            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            y_coords, x_coords = np.where(class_mask)
            x_center = int(x_coords.mean())
            y_center = int(y_coords.mean())

            centroids[class_name] = (x_center, y_center)

        return centroids


def load_endoscapes_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> EndoscapesNet:
    """Load EndoscapesNet model with default checkpoint."""
    if checkpoint_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        checkpoint_path = str(
            repo_root
            / "checkpoints"
            / "endoscapes_s1234_tts0_tvs0_all_0.001_cosine_dice_combo__best.pt"
        )
    return EndoscapesNet.load_from_checkpoint(checkpoint_path, device=device, n_classes=7)
