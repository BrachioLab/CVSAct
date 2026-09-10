"""EndoscapesNet model adapter for endopoint."""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch

from .base import ModelAdapter, Batch
from .endoscapesnet import EndoscapesNet, load_endoscapes_model


class EndoscapesNetAdapter(ModelAdapter):
    """EndoscapesNet adapter for coarse organ segmentation (Endoscapes)."""

    def __init__(
        self,
        model_name: str = "endoscapesnet",
        use_cache: bool = True,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        return_masks: bool = True,
        min_pixels: int = 50,
    ):
        """Initialize EndoscapesNet adapter."""
        self.model_name = model_name
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path(
            "/shared_data0/weiqiuy/llm_cholec_organ/cache/endoscapesnet"
        )
        self.device = device if torch.cuda.is_available() else "cpu"
        self.return_masks = return_masks
        self.min_pixels = min_pixels

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("Loading EndoscapesNet model...")

        self.model = load_endoscapes_model(checkpoint_path=checkpoint_path, device=self.device)

        if self.verbose:
            print(f"EndoscapesNet loaded successfully on {self.device}")

    def _get_cache_key(self, prompt: str, system_prompt: str, image_hash: str = "") -> str:
        combined = f"{system_prompt}\n{prompt}\n{image_hash}\n{self.return_masks}\n{self.min_pixels}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_image_hash(self, image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        return hashlib.sha256(img_bytes).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return data.get("response", "")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, response: str):
        if not self.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"response": response}, f)
        except Exception as e:
            if self.verbose:
                print(f"Error saving to cache: {e}")

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        mask_uint8 = mask.astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8, mode="L")

        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return mask_base64

    def _normalize_organ_name(self, name: str) -> str:
        name_mapping = {
            "Background": "Background",
            "Cystic Plate": "Cystic Plate",
            "HC Triangle": "HC Triangle Dissection",
            "HC Triangle Dissection": "HC Triangle Dissection",
            "Cystic Artery": "Cystic Artery",
            "Cystic Duct": "Cystic Duct",
            "Gallbladder": "Gallbladder",
            "Tool": "Tool",
        }
        return name_mapping.get(name, name)

    def _format_detection_response(
        self,
        mask: np.ndarray,
        presence: Dict[str, bool],
        bboxes: Dict[str, List[Tuple[int, int, int, int]]],
        centroids: Dict[str, Tuple[int, int]],
        requested_organs: List[str],
    ) -> str:
        result = {}

        if "Background" in requested_organs:
            background_pixels = np.sum(mask == 0)
            result["Background"] = {
                "present": background_pixels >= self.min_pixels,
                "bbox": None,
            }

        for organ_name in requested_organs:
            if organ_name == "Background":
                continue

            normalized = self._normalize_organ_name(organ_name)
            if normalized in self.model.LABEL2ID:
                is_present = bool(presence.get(normalized, False))
                organ_bboxes = bboxes.get(normalized, [])

                bbox = None
                if organ_bboxes:
                    x1, y1, x2, y2 = organ_bboxes[0]
                    bbox = [x1, y1, x2, y2]

                point = None
                if normalized in centroids:
                    x, y = centroids[normalized]
                    point = [x, y]

                result[organ_name] = {
                    "present": is_present,
                    "bbox": bbox,
                }
                if point:
                    result[organ_name]["point"] = point

                if self.return_masks and is_present:
                    class_id = self.model.LABEL2ID.get(normalized, -1)
                    if class_id >= 0:
                        organ_mask = (mask == class_id).astype(np.uint8) * 255
                        result[organ_name]["mask"] = self._mask_to_base64(organ_mask)
            else:
                result[organ_name] = {
                    "present": False,
                    "bbox": None,
                }

        if self.return_masks:
            result["_full_mask"] = {
                "encoded": self._mask_to_base64(mask),
                "shape": list(mask.shape),
                "classes": self.model.ID2LABEL,
            }

        return json.dumps(result, indent=2)

    def _extract_organs_from_prompt(self, prompt: str) -> List[str]:
        import re

        organ_list_match = re.search(r"following organs?:\s*\n((?:\s*-\s*[^\n]+\n?)+)", prompt, re.IGNORECASE)
        if organ_list_match:
            organ_list_text = organ_list_match.group(1)
            organ_names = re.findall(r"-\s*([^\n]+)", organ_list_text)
            return [name.strip() for name in organ_names]

        json_matches = re.findall(r'"([^"]+)":\s*\{[^}]*"present"', prompt)
        if json_matches:
            seen = set()
            organ_names = []
            for name in json_matches:
                if name not in seen:
                    seen.add(name)
                    organ_names.append(name)
            return organ_names

        return [
            "Cystic Plate",
            "HC Triangle Dissection",
            "Cystic Artery",
            "Cystic Duct",
            "Gallbladder",
            "Tool",
        ]

    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        responses = []

        for query in prompts:
            text_prompt = ""
            image = None

            for part in query:
                if isinstance(part, str):
                    text_prompt += part
                else:
                    image = part

            image_hash = self._get_image_hash(image) if image else ""

            cache_key = self._get_cache_key(text_prompt, system_prompt, image_hash)
            cached_response = self._load_from_cache(cache_key)

            if cached_response is not None:
                if self.verbose:
                    print("Using cached EndoscapesNet response")
                responses.append(cached_response)
                continue

            if image is None:
                if self.verbose:
                    print("Warning: No image provided for EndoscapesNet detection")
                response = json.dumps({"error": "No image provided"})
            else:
                try:
                    original_size = image.size

                    input_tensor = self.model.process_image(image, target_size=(640, 384))
                    device = next(self.model.parameters()).device
                    input_tensor = input_tensor.to(device)

                    with torch.no_grad():
                        output = self.model(input_tensor, return_tuple=True)

                    logits = output.logits
                    pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

                    target_height, target_width = original_size[1], original_size[0]

                    if target_width < target_height:
                        target_width, target_height = target_height, target_width

                    if pred_mask.shape != (target_height, target_width):
                        from scipy import ndimage

                        if self.verbose:
                            print(f"  Resizing mask from {pred_mask.shape} to ({target_height}, {target_width})")
                        pred_mask_resized = np.zeros((target_height, target_width), dtype=pred_mask.dtype)
                        for class_id in np.unique(pred_mask):
                            class_mask = (pred_mask == class_id).astype(np.float32)
                            class_mask_resized = ndimage.zoom(
                                class_mask,
                                (target_height / pred_mask.shape[0], target_width / pred_mask.shape[1]),
                                order=0,
                            )
                            pred_mask_resized[class_mask_resized > 0.5] = class_id
                        pred_mask = pred_mask_resized.astype(np.int32)
                        if self.verbose:
                            print(f"  Mask resized to shape: {pred_mask.shape}")

                    presence = self.model.get_organ_presence(pred_mask, min_pixels=self.min_pixels)
                    bboxes = self.model.get_bounding_boxes(pred_mask, min_pixels=self.min_pixels)
                    centroids = self.model.get_centroid_points(pred_mask, min_pixels=self.min_pixels)

                    if self.verbose:
                        print(f"EndoscapesNet detected: {[k for k, v in presence.items() if v]}")

                    requested_organs = self._extract_organs_from_prompt(text_prompt)
                    response = self._format_detection_response(pred_mask, presence, bboxes, centroids, requested_organs)

                except Exception as e:
                    if self.verbose:
                        print(f"Error in EndoscapesNet inference: {e}")
                    import traceback

                    traceback.print_exc()
                    response = json.dumps({"error": str(e)})

            self._save_to_cache(cache_key, response)
            responses.append(response)

        return responses
