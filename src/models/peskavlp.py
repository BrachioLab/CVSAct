#!/usr/bin/env python3
"""
PeskaVLP (Surgical Vision-Language Pre-training) model adapter.

This module provides an interface to use the PeskaVLP model for surgical image analysis
within the endopoint framework.
"""

import sys
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

# Add SurgVLP to path
sys.path.insert(0, '/shared_data0/weiqiuy/github/SurgVLP')

import surgvlp
from mmengine.config import Config


class PeskaVLPModel:
    """
    PeskaVLP model wrapper for surgical image analysis.
    """
    
    def __init__(self, 
                 config_path: str = '/shared_data0/weiqiuy/github/SurgVLP/tests/config_peskavlp.py',
                 device: Optional[str] = None):
        """
        Initialize PeskaVLP model.
        
        Args:
            config_path: Path to the PeskaVLP configuration file
            device: Device to load model on (auto-detected if None)
        """
        self.config_path = config_path
        
        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and preprocessor
        self.model = None
        self.preprocess = None
        self._load_model()
    
    def _load_model(self):
        """Load the PeskaVLP model and preprocessing transform."""
        print(f"Loading PeskaVLP model from config: {self.config_path}")
        
        # Load configuration
        configs = Config.fromfile(self.config_path)['config']
        
        # Load model and preprocessor
        self.model, self.preprocess = surgvlp.load(configs.model_config, device=self.device)
        
        print(f"✅ PeskaVLP model loaded on {self.device}")
    
    def analyze_image(self, 
                      image: Union[str, Image.Image],
                      class_labels: List[str],
                      threshold: float = 0.65) -> List[str]:
        """
        Analyze a single image using PeskaVLP with the given class labels.
        
        Args:
            image: Path to image file or PIL Image object
            class_labels: List of class labels to check for (will be lowercased)
            threshold: Confidence threshold for results
        
        Returns:
            List of detected class labels (in lowercase)
        """
        # Load image if path provided
        if isinstance(image, str):
            if not Path(image).exists():
                raise FileNotFoundError(f"Image not found: {image}")
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be either a file path or PIL Image object")
        
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Lowercase all class labels for consistency
        class_labels_lower = [label.lower() for label in class_labels]
        
        # Tokenize text labels
        text_tokens = surgvlp.tokenize(class_labels_lower, device=self.device)
        
        # Run inference
        with torch.no_grad():
            output_dict = self.model(image_tensor, text_tokens, mode='all')
            
            # Normalize embeddings
            image_embeddings = output_dict['img_emb']
            text_embeddings = output_dict['text_emb']
            
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            logits_per_image = 100.0 * image_embeddings @ text_embeddings.T
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
        # Filter by threshold
        detected_labels = []
        for i, (label, prob) in enumerate(zip(class_labels_lower, probs)):
            if prob >= threshold:
                detected_labels.append(label)
        
        return detected_labels
    
    def parse_output(self, detected_labels: List[str]) -> List[str]:
        """
        Parse the output from analyze_image (compatibility method).
        
        Args:
            detected_labels: List of detected labels
        
        Returns:
            Same list of detected labels (already parsed)
        """
        return detected_labels


def load_peskavlp_cholecseg8k() -> PeskaVLPModel:
    """
    Load PeskaVLP model for CholecSeg8k dataset.
    
    Returns:
        Configured PeskaVLPModel instance
    """
    return PeskaVLPModel()


def load_peskavlp_cholec_organs() -> PeskaVLPModel:
    """
    Load PeskaVLP model for CholecOrgans dataset.
    
    Returns:
        Configured PeskaVLPModel instance
    """
    return PeskaVLPModel()


def load_peskavlp_cholec_gonogo() -> PeskaVLPModel:
    """
    Load PeskaVLP model for CholecGoNoGo dataset.
    
    Returns:
        Configured PeskaVLPModel instance
    """
    return PeskaVLPModel()