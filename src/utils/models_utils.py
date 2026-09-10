"""Shared utilities for model adapters (ported from lvlm_cvs_reasoning)."""

from __future__ import annotations

import base64
import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Optional, Union

try:  # optional dependency
    import diskcache  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    diskcache = None

try:  # optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore


class _DictCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value


# Create a shared cache
CACHE_DIR = Path.home() / ".cache" / "endopoint" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
if diskcache is not None:
    cache = diskcache.Cache(str(CACHE_DIR))
else:  # pragma: no cover - fallback
    cache = _DictCache()


def _lazy_import_torch_np():
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        np = None  # type: ignore
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore
    try:
        from torchvision.transforms import functional as tvtf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        tvtf = None  # type: ignore
    return np, torch, tvtf


def is_image(x: Any) -> bool:
    """Check if the input is an image in a supported format."""
    if Image is not None and isinstance(x, Image.Image):
        return True
    np, torch, _ = _lazy_import_torch_np()
    if torch is not None and isinstance(x, torch.Tensor):
        return True
    if np is not None and isinstance(x, np.ndarray):
        return True
    return False


def image_to_base64(
    image: Union["torch.Tensor", "np.ndarray", "Image.Image"],
    image_format: str = "PNG",
) -> str:
    """Convert an image to a base64-encoded string."""
    np, torch, tvtf = _lazy_import_torch_np()
    if Image is None:
        raise ValueError("PIL is required for image conversion.")
    if tvtf is None:
        raise ValueError("torchvision is required for image conversion.")

    try:
        # Convert to PIL image if needed
        if torch is not None and isinstance(image, torch.Tensor):
            mode = "RGB" if image.ndim == 3 and image.shape[0] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)
        elif np is not None and isinstance(image, np.ndarray):
            mode = "RGB" if image.ndim == 3 and image.shape[2] == 3 else "L"
            image = tvtf.to_pil_image(image, mode=mode)

        if not isinstance(image, Image.Image):
            raise ValueError(f"Image is not PIL.Image.Image: {type(image)}")
        image.load()  # Force loading the image

        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format)
            return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to convert image to base64: {exc}") from exc


def to_pil_image(x: Any) -> "Image.Image":
    """Convert an image to a PIL Image object."""
    np, torch, tvtf = _lazy_import_torch_np()
    if Image is None:
        raise ValueError("PIL is required for image conversion.")
    if isinstance(x, Image.Image):
        return x
    if tvtf is not None and (
        (torch is not None and isinstance(x, torch.Tensor))
        or (np is not None and isinstance(x, np.ndarray))
    ):
        return tvtf.to_pil_image(x)
    raise ValueError(f"Invalid image type: {type(x)}")


def get_cache_key(
    model_name: str,
    prompt: Any,
    system_prompt: Optional[str] = None,
) -> str:
    """Convert a (system_prompt, prompt) into a stable hash string."""
    sys_part = system_prompt or ""

    if isinstance(prompt, str):
        obj = ("user_text", prompt)
    elif isinstance(prompt, tuple):
        objs = []
        for p in prompt:
            if isinstance(p, str):
                objs.append(("text", p))
            elif is_image(p):
                # base64-encode image deterministically for hashing
                objs.append(("image_b64", image_to_base64(p, "PNG")))
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        obj = ("user_tuple", tuple(objs))
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    return hashlib.sha256(pickle.dumps((model_name, sys_part, obj))).hexdigest()
