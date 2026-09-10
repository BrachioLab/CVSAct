"""CholeNet model adapter for endopoint."""

import json
import base64
import io
from typing import Sequence, Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch

from .base import ModelAdapter, Batch
from .cholenet import CholeNet, load_cholenet_model


class CholeNetAdapter(ModelAdapter):
    """CholeNet model adapter for multi-organ detection and segmentation masks.
    
    CholeNet provides full segmentation masks for multiple anatomical structures
    in cholecystectomy procedures.
    """
    
    def __init__(self, 
                 model_name: str = "cholenet",
                 use_cache: bool = True,
                 verbose: bool = True,
                 cache_dir: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda',
                 return_masks: bool = True,
                 min_pixels: int = 50):
        """Initialize CholeNet adapter.
        
        Args:
            model_name: Model identifier
            use_cache: Whether to use caching
            verbose: Whether to enable verbose logging
            cache_dir: Directory for caching responses
            checkpoint_path: Path to model checkpoint
            device: Device to run model on
            return_masks: Whether to include masks in response
            min_pixels: Minimum pixels for presence detection
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/shared_data0/weiqiuy/llm_cholec_organ/cache/cholenet")
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.return_masks = return_masks
        self.min_pixels = min_pixels
        
        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CholeNet model
        if self.verbose:
            print(f"Loading CholeNet model...")
        
        self.model = load_cholenet_model(checkpoint_path=checkpoint_path, device=self.device)
        
        if self.verbose:
            print(f"CholeNet loaded successfully on {self.device}")
    
    def _get_cache_key(self, prompt: str, system_prompt: str, image_hash: str = "") -> str:
        """Generate cache key for a prompt.
        
        Args:
            prompt: The text prompt
            system_prompt: The system prompt
            image_hash: Hash of the image content
            
        Returns:
            SHA-256 hash as cache key
        """
        combined = f"{system_prompt}\n{prompt}\n{image_hash}\n{self.return_masks}\n{self.min_pixels}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_image_hash(self, image) -> str:
        """Generate a hash for an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Hash string for the image
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return hashlib.sha256(img_bytes).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache if it exists."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('response', '')
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({'response': response}, f)
        except Exception as e:
            if self.verbose:
                print(f"Error saving to cache: {e}")
    
    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """Convert mask to base64 encoded string.
        
        Args:
            mask: Segmentation mask [H, W]
            
        Returns:
            Base64 encoded mask
        """
        # Convert to uint8 (class indices should be 0-255)
        mask_uint8 = mask.astype(np.uint8)
        
        # Create PIL Image
        mask_img = Image.fromarray(mask_uint8, mode='L')
        
        # Save to bytes
        buffer = io.BytesIO()
        mask_img.save(buffer, format='PNG')
        
        # Encode to base64
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return mask_base64
    
    def _normalize_organ_name(self, name: str) -> str:
        """Normalize organ name to match CholeNet labels.
        
        Args:
            name: Original organ name
            
        Returns:
            Normalized name matching CholeNet labels
        """
        # Common normalizations
        name_mapping = {
            "Black Background": "Background",
            "Abdominal Wall": "Abdominal Wall",
            "Gastrointestinal Tract": "Gastrointestinal Tract",
            "L-hook Electrocautery": "L-hook Electrocautery",
            "Hepatic Vein": "Hepatic Vein",
            "Liver Ligament": "Liver Ligament",
            # Direct matches
            "Liver": "Liver",
            "Gallbladder": "Gallbladder",
            "Hepatocystic Triangle": "Hepatocystic Triangle",
            "Fat": "Fat",
            "Grasper": "Grasper",
            "Connective Tissue": "Connective Tissue",
            "Blood": "Blood",
            "Cystic Duct": "Cystic Duct",
        }
        
        return name_mapping.get(name, name)
    
    def _format_detection_response(self, 
                                  mask: np.ndarray,
                                  presence: Dict[str, bool],
                                  bboxes: Dict[str, List[Tuple[int, int, int, int]]],
                                  centroids: Dict[str, Tuple[int, int]],
                                  requested_organs: List[str]) -> str:
        """Format CholeNet detection results as JSON response.
        
        Args:
            mask: Segmentation mask
            presence: Organ presence dictionary
            bboxes: Bounding boxes dictionary
            centroids: Centroid points dictionary
            requested_organs: List of requested organ names
            
        Returns:
            JSON string with detection results
        """
        result = {}
        
        # CholeNet only detects 3 organs: Liver, Gallbladder, Hepatocystic Triangle
        # Map these to requested organs
        cholenet_organs = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
        
        # Cross-dataset mapping for cholec_gonogo:
        # Hepatocystic Triangle (class 3) -> Go Zone (safe dissection area)
        # Everything else -> NoGo Zone
        cross_dataset_mapping = {
            "Go Zone": "Hepatocystic Triangle",  # Hepatocystic Triangle maps to Go Zone
            "NoGo Zone": None,  # CholeNet detects organs, not NoGo zone directly
            "Go (Safe to Incise)": "Hepatocystic Triangle",
            "NoGo (Unsafe to Incise)": None,
        }
        
        # Handle background class mapping - different datasets use different names
        if "Black Background" in requested_organs or "Background" in requested_organs:
            # Check if background (class 0) is actually present in the mask
            background_name = "Black Background" if "Black Background" in requested_organs else "Background"
            background_pixels = np.sum(mask == 0)
            
            # Get bounding box for background if present
            background_bboxes = []
            if background_pixels >= self.min_pixels:
                # Background could have multiple disconnected regions
                # but typically we don't report bbox for background
                pass
            
            result[background_name] = {
                "present": background_pixels >= self.min_pixels,
                "bbox": None  # Background typically doesn't have a meaningful bbox
            }
        
        for organ_name in requested_organs:
            # Skip if already handled
            if organ_name in ["Black Background", "Background"]:
                continue
            
            # Check for cross-dataset mapping (e.g., cholec_gonogo)
            if organ_name in cross_dataset_mapping:
                mapped_organ = cross_dataset_mapping[organ_name]
                if mapped_organ:
                    # Map zone to corresponding organ
                    is_present = bool(presence.get(mapped_organ, False))
                    organ_bboxes = bboxes.get(mapped_organ, [])
                    
                    # Format bbox (use first bbox if multiple)
                    bbox = None
                    if organ_bboxes:
                        x1, y1, x2, y2 = organ_bboxes[0]
                        bbox = [x1, y1, x2, y2]
                    
                    # Get centroid if present
                    point = None
                    if mapped_organ in centroids:
                        x, y = centroids[mapped_organ]
                        point = [x, y]
                    
                    result[organ_name] = {
                        "present": is_present,
                        "bbox": bbox
                    }
                    
                    # Add point if available
                    if point:
                        result[organ_name]["point"] = point
                    
                    # Add mask if requested
                    if self.return_masks and is_present:
                        # Create binary mask for this organ
                        class_id = self.model.LABEL2ID.get(mapped_organ, -1)
                        if class_id > 0:
                            organ_mask = (mask == class_id).astype(np.uint8) * 255
                            result[organ_name]["mask"] = self._mask_to_base64(organ_mask)
                elif organ_name == "NoGo Zone":
                    # NoGo Zone = everything except Hepatocystic Triangle (Go Zone)
                    # This includes Liver, Gallbladder, and other tissues
                    nogo_pixels = np.sum((mask != 3) & (mask != 0))  # Not Hepatocystic Triangle and not background
                    is_present = nogo_pixels >= self.min_pixels
                    
                    result[organ_name] = {
                        "present": is_present,
                        "bbox": None  # NoGo zone is typically multiple disconnected regions
                    }
                else:
                    # CholeNet cannot detect this
                    result[organ_name] = {
                        "present": False,
                        "bbox": None
                    }
                continue
            # Check if CholeNet can detect this organ
            if organ_name in cholenet_organs:
                # CholeNet can detect this organ
                is_present = bool(presence.get(organ_name, False))
                organ_bboxes = bboxes.get(organ_name, [])
                
                # Format bbox (use first bbox if multiple)
                bbox = None
                if organ_bboxes:
                    x1, y1, x2, y2 = organ_bboxes[0]
                    bbox = [x1, y1, x2, y2]
                
                # Get centroid if present
                point = None
                if organ_name in centroids:
                    x, y = centroids[organ_name]
                    point = [x, y]
                
                result[organ_name] = {
                    "present": is_present,
                    "bbox": bbox
                }
                
                # Add point if available
                if point:
                    result[organ_name]["point"] = point
                
                # Add mask if requested (mask is already at original resolution)
                if self.return_masks and is_present:
                    # Create binary mask for this organ
                    class_id = self.model.LABEL2ID.get(organ_name, -1)
                    if class_id > 0:
                        organ_mask = (mask == class_id).astype(np.uint8) * 255
                        result[organ_name]["mask"] = self._mask_to_base64(organ_mask)
            else:
                # CholeNet cannot detect this organ - always report as not present
                result[organ_name] = {
                    "present": False,
                    "bbox": None
                }
        
        # Add full segmentation mask if requested
        if self.return_masks:
            result["_full_mask"] = {
                "encoded": self._mask_to_base64(mask),
                "shape": list(mask.shape),
                "classes": self.model.ID2LABEL
            }
        
        return json.dumps(result, indent=2)
    
    def _extract_organs_from_prompt(self, prompt: str) -> List[str]:
        """Extract organ names from detection prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            List of organ names from prompt
        """
        import re
        
        # Look for organ list in the prompt
        organ_list_match = re.search(r'following organs?:\s*\n((?:\s*-\s*[^\n]+\n?)+)', prompt, re.IGNORECASE)
        if organ_list_match:
            organ_list_text = organ_list_match.group(1)
            organ_names = re.findall(r'-\s*([^\n]+)', organ_list_text)
            return [name.strip() for name in organ_names]
        
        # Fallback: Try JSON format in examples
        json_matches = re.findall(r'"([^"]+)":\s*\{[^}]*"present"', prompt)
        if json_matches:
            seen = set()
            organ_names = []
            for name in json_matches:
                if name not in seen:
                    seen.add(name)
                    organ_names.append(name)
            return organ_names
        
        # Default for CholecSeg8k-like organs
        return ["Liver", "Gallbladder", "Hepatocystic Triangle", "Fat", "Grasper",
                "Connective Tissue", "Blood", "Cystic Duct", "L-hook Electrocautery",
                "Abdominal Wall", "Gastrointestinal Tract", "Hepatic Vein", "Liver Ligament"]
    
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through CholeNet.
        
        Args:
            prompts: Batch of queries, each a tuple of text/image parts
            system_prompt: System prompt (ignored for CholeNet)
            
        Returns:
            List of JSON responses with organ detection results and masks
        """
        responses = []
        
        for query in prompts:
            # Extract text and image from query
            text_prompt = ""
            image = None
            
            for part in query:
                if isinstance(part, str):
                    text_prompt += part
                else:  # PIL Image
                    image = part
            
            # Generate image hash for cache key
            image_hash = self._get_image_hash(image) if image else ""
            
            # Check cache first
            cache_key = self._get_cache_key(text_prompt, system_prompt, image_hash)
            cached_response = self._load_from_cache(cache_key)
            
            if cached_response is not None:
                if self.verbose:
                    print(f"Using cached CholeNet response")
                responses.append(cached_response)
                continue
            
            # Process with CholeNet
            if image is None:
                if self.verbose:
                    print("Warning: No image provided for CholeNet detection")
                response = json.dumps({"error": "No image provided"})
            else:
                try:
                    # Store original image size
                    original_size = image.size  # (width, height)
                    
                    # Process image at model's expected size
                    input_tensor = self.model.process_image(image, target_size=(640, 384))
                    device = next(self.model.parameters()).device
                    input_tensor = input_tensor.to(device)
                    
                    # Get model output
                    with torch.no_grad():
                        output = self.model(input_tensor, return_tuple=True)
                        
                    # Extract logits from ModelOutput and convert to predictions
                    logits = output.logits  # [1, n_classes, H, W]
                    pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()  # [H, W] at 384x640
                    
                    # Resize mask back to original image size if different
                    # original_size is (width, height) from PIL
                    # pred_mask is (height, width) from model
                    target_height, target_width = original_size[1], original_size[0]
                    
                    # Check if dimensions are swapped (width should be > height for these datasets)
                    if target_width < target_height:
                        # Dimensions are likely swapped, correct them
                        target_width, target_height = target_height, target_width
                    
                    if pred_mask.shape != (target_height, target_width):
                        from scipy import ndimage
                        if self.verbose:
                            print(f"  Resizing mask from {pred_mask.shape} to ({target_height}, {target_width})")
                        # Resize mask to target dimensions
                        pred_mask_resized = np.zeros((target_height, target_width), dtype=pred_mask.dtype)
                        # Use nearest neighbor interpolation to preserve class indices
                        for class_id in np.unique(pred_mask):
                            class_mask = (pred_mask == class_id).astype(np.float32)
                            class_mask_resized = ndimage.zoom(class_mask, 
                                                             (target_height / pred_mask.shape[0],
                                                              target_width / pred_mask.shape[1]),
                                                             order=0)  # nearest neighbor
                            pred_mask_resized[class_mask_resized > 0.5] = class_id
                        pred_mask = pred_mask_resized.astype(np.int32)
                        if self.verbose:
                            print(f"  Mask resized to shape: {pred_mask.shape}")
                    
                    # Get organ presence
                    presence = self.model.get_organ_presence(pred_mask, min_pixels=self.min_pixels)
                    
                    # Get bounding boxes
                    bboxes = self.model.get_bounding_boxes(pred_mask, min_pixels=self.min_pixels)
                    
                    # Get centroids
                    centroids = self.model.get_centroid_points(pred_mask, min_pixels=self.min_pixels)
                    
                    if self.verbose:
                        print(f"CholeNet detected: {[k for k, v in presence.items() if v]}")
                    
                    # Extract requested organs from prompt
                    requested_organs = self._extract_organs_from_prompt(text_prompt)
                    
                    # Format response
                    response = self._format_detection_response(pred_mask, presence, bboxes, centroids, requested_organs)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in CholeNet inference: {e}")
                    import traceback
                    traceback.print_exc()
                    response = json.dumps({"error": str(e)})
            
            # Save to cache
            self._save_to_cache(cache_key, response)
            responses.append(response)
        
        return responses