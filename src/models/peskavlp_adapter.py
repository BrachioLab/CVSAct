"""PeskaVLP model adapter for endopoint."""

import json
from typing import Sequence, Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

from .base import ModelAdapter, Batch
from .peskavlp import load_peskavlp_cholecseg8k, load_peskavlp_cholec_organs, load_peskavlp_cholec_gonogo


class PeskaVLPAdapter(ModelAdapter):
    """PeskaVLP model adapter for organ presence detection.
    
    Important: PeskaVLP only provides organ presence detection, not bounding boxes.
    All bbox fields will be null/empty when using PeskaVLP.
    """
    
    def __init__(self, 
                 model_name: str = "peskavlp",
                 use_cache: bool = True,
                 verbose: bool = True,
                 cache_dir: Optional[str] = None,
                 dataset: Optional[str] = None):
        """Initialize PeskaVLP adapter.
        
        Args:
            model_name: Model identifier (e.g., "peskavlp", "peskavlp-cholecseg8k")
            use_cache: Whether to use caching
            verbose: Whether to enable verbose logging
            cache_dir: Directory for caching responses
            dataset: Dataset name to auto-select the right model
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/shared_data0/weiqiuy/llm_cholec_organ/cache/peskavlp")
        self.dataset = dataset
        
        # Ensure cache directory exists
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the appropriate PeskaVLP model based on dataset or model name
        if self.verbose:
            print(f"Initializing PeskaVLP model adapter...")
        
        # Select model based on dataset or model name
        if dataset == "cholec_organs" or "organs" in model_name.lower():
            self.model = load_peskavlp_cholec_organs()
        elif dataset == "cholec_gonogo" or "gonogo" in model_name.lower():
            self.model = load_peskavlp_cholec_gonogo()
        else:
            # Default to CholecSeg8k
            self.model = load_peskavlp_cholecseg8k()
        
        # Classes that PeskaVLP typically doesn't detect (backgrounds)
        # These are in the label files but rarely/never actually detected
        self.non_detectable_classes = {"background", "black background"}
    
    def _get_cache_key(self, prompt: str, system_prompt: str, image_hash: str = "") -> str:
        """Generate cache key for a prompt.
        
        Args:
            prompt: The text prompt
            system_prompt: The system prompt
            image_hash: Hash of the image content
            
        Returns:
            SHA-256 hash as cache key
        """
        combined = f"{system_prompt}\n{prompt}\n{image_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_image_hash(self, image) -> str:
        """Generate a hash for an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Hash string for the image
        """
        import io
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Generate hash
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
    
    def _format_detection_response(self, detected_organs: List[str], requested_organs: List[str], 
                                   original_organ_names: List[str]) -> str:
        """Format PeskaVLP detection results as JSON response.
        
        Args:
            detected_organs: List of organs detected by PeskaVLP (lowercase names from PeskaVLP)
            requested_organs: List of organs requested in the prompt (potentially lowercase)
            original_organ_names: Original organ names with proper capitalization from the prompt
            
        Returns:
            JSON string with detection results
        """
        # PeskaVLP returns lowercase organ names
        detected_lower = {organ.lower() for organ in detected_organs}
        
        # Create mapping from lowercase to original names
        name_mapping = {}
        for orig_name in original_organ_names:
            name_mapping[orig_name.lower()] = orig_name
        
        # Build response for original organ names
        result = {}
        for orig_name in original_organ_names:
            organ_lower = orig_name.lower()
            
            # Check if this is a non-detectable class (like "background")
            if organ_lower in self.non_detectable_classes:
                # Background is never detected by PeskaVLP
                is_present = False
            else:
                # Check if organ is present in detected list
                is_present = organ_lower in detected_lower
            
            # Use the original organ name (with proper capitalization) as key
            result[orig_name] = {
                "present": is_present,
                "bbox": None  # PeskaVLP doesn't provide bounding boxes
            }
        
        return json.dumps(result, indent=2)
    
    def _extract_organs_from_prompt(self, prompt: str) -> Tuple[List[str], List[str]]:
        """Extract organ names from detection prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Tuple of (original_names, lowercase_names) where:
            - original_names: List of organ names with original capitalization from prompt
            - lowercase_names: List of organ names in lowercase for PeskaVLP
        """
        # Look for organ list in the prompt (format: "- Organ Name")
        import re
        
        # First try to find the organ list section
        organ_list_match = re.search(r'following organs?:\s*\n((?:\s*-\s*[^\n]+\n?)+)', prompt, re.IGNORECASE)
        if organ_list_match:
            organ_list_text = organ_list_match.group(1)
            # Extract organ names from bullet list preserving original case
            original_names = re.findall(r'-\s*([^\n]+)', organ_list_text)
            original_names = [name.strip() for name in original_names]
            lowercase_names = [name.lower() for name in original_names]
            return original_names, lowercase_names
        
        # Fallback: Try JSON format in examples
        json_matches = re.findall(r'"([^"]+)":\s*\{[^}]*"present"', prompt)
        if json_matches:
            # Use unique organ names preserving order and case
            seen = set()
            original_names = []
            for name in json_matches:
                if name not in seen:
                    seen.add(name)
                    original_names.append(name)
            lowercase_names = [name.lower() for name in original_names]
            return original_names, lowercase_names
        
        # Final fallback based on dataset
        if "hepatocystic" in prompt.lower():
            # CholecOrgans dataset
            original_names = ["Liver", "Gallbladder", "Hepatocystic Triangle"]
        elif "go zone" in prompt.lower() or "no go zone" in prompt.lower():
            # CholecGoNoGo dataset  
            original_names = ["Go Zone", "No Go Zone"]
        else:
            # CholecSeg8k dataset
            original_names = ["Black Background", "Abdominal Wall", "Liver", "Gastrointestinal Tract", 
                            "Fat", "Grasper", "Connective Tissue", "Blood", "Cystic Duct",
                            "L-hook Electrocautery", "Gallbladder", "Hepatic Vein", "Liver Ligament"]
        
        lowercase_names = [name.lower() for name in original_names]
        return original_names, lowercase_names
    
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through PeskaVLP.
        
        Args:
            prompts: Batch of queries, each a tuple of text/image parts
            system_prompt: System prompt (ignored for PeskaVLP)
            
        Returns:
            List of JSON responses with organ detection results
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
                    print(f"Using cached PeskaVLP response")
                responses.append(cached_response)
                continue
            
            # Process with PeskaVLP
            if image is None:
                if self.verbose:
                    print("Warning: No image provided for PeskaVLP detection")
                response = json.dumps({"error": "No image provided"})
            else:
                try:
                    # Extract requested organs from prompt (both original and lowercase)
                    original_names, lowercase_names = self._extract_organs_from_prompt(text_prompt)
                    
                    if self.verbose:
                        print(f"Original organ names from prompt: {original_names}")
                        print(f"Lowercase names for PeskaVLP: {lowercase_names}")
                    
                    # Run PeskaVLP inference with lowercase labels
                    detected_organs = self.model.analyze_image(image, lowercase_names, threshold=0.65)
                    
                    if self.verbose:
                        print(f"PeskaVLP detected: {detected_organs}")
                    
                    # Format response with proper mapping back to original names
                    response = self._format_detection_response(detected_organs, lowercase_names, original_names)
                    
                    if self.verbose:
                        print(f"PeskaVLP response keys: {list(json.loads(response).keys())}")
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in PeskaVLP inference: {e}")
                    response = json.dumps({"error": str(e)})
            
            # Save to cache
            self._save_to_cache(cache_key, response)
            responses.append(response)
        
        return responses