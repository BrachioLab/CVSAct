"""
Llama Vision model adapter for endopoint.
Supports Llama-3.2-11B-Vision models including surgical-specific variants.
"""

import time
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, MllamaForConditionalGeneration

from .base import ModelAdapter, PromptPart, OneQuery, Batch
from .utils import cache, get_cache_key, is_image, to_pil_image


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama Vision models including surgical-specific variants."""
    
    def __init__(
        self,
        model_name: str = "nvidia/Llama-3.2-11B-Vision-Surgical-CholecT50",
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_cache: bool = True,
        device: str = "cuda",
        verbose: bool = True,
        num_tries_per_request: int = 3,
    ):
        """Initialize Llama Vision model adapter.
        
        Args:
            model_name: HuggingFace model ID
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens to generate
            use_cache: Whether to cache responses
            device: Device to run model on
            verbose: Whether to print debug info
            num_tries_per_request: Number of retries on failure
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self.device = device
        self.verbose = verbose
        self.num_tries_per_request = num_tries_per_request
        self.model_id = model_name  # For compatibility
        
        self._load_model()
    
    def _load_model(self):
        """Load the Llama Vision model and processor."""
        if self.verbose:
            print(f"Loading Llama Vision model: {self.model_name}")
        
        # Determine torch dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        
        # Load model with auto device mapping
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.verbose:
            print(f"Model loaded successfully with dtype {torch_dtype}")
    
    def _format_prompt_parts(self, parts: OneQuery, system_prompt: str) -> tuple:
        """Format prompt parts into Llama chat format with multiple images support.
        
        Args:
            parts: Tuple of prompt parts (text and/or images)
            system_prompt: System prompt to prepend
            
        Returns:
            Tuple of (messages, images_list)
        """
        # Build content list with interleaved text and images
        content = []
        images = []
        
        # Add system prompt as first text if provided
        if system_prompt:
            content.append({"type": "text", "text": system_prompt + "\n"})
        
        # Process each part maintaining order
        for part in parts:
            if is_image(part):
                # Add image placeholder to content
                content.append({"type": "image"})
                # Add actual image to images list
                images.append(to_pil_image(part).convert("RGB"))
            else:
                # Add text content
                content.append({"type": "text", "text": str(part)})
        
        # Build messages in chat format
        messages = [{
            "role": "user",
            "content": content
        }]
        
        return messages, images
    
    def _generate_single(self, query: OneQuery, system_prompt: str) -> str:
        """Generate response for a single query.
        
        Args:
            query: Single query tuple
            system_prompt: System prompt
            
        Returns:
            Generated text response
        """
        # Format the prompt parts
        messages, images = self._format_prompt_parts(query, system_prompt)
        
        # Apply chat template (no tokenization yet)
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Tokenize text + images together
        if images:
            inputs = self.processor(
                text=[prompt_text],  # Keep as list for batch dimension
                images=images,       # List of images (works for 1 or more)
                return_tensors="pt",
            )
        else:
            # Text only
            inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
            )
        
        # Move to device
        inputs = inputs.to(self.model.device)
        
        # Generate with appropriate sampling
        with torch.no_grad():
            if self.temperature == 0:
                # Deterministic generation
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
            else:
                # Sample with temperature
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                )
        
        # Decode response
        full_response = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        
        # Extract only the generated part (remove input prompt from response)
        # The Llama model includes the full conversation with "user" and "assistant" markers
        
        # Try multiple extraction methods
        response = full_response
        
        # Method 1: Look for "assistant\n\n" marker (common in chat models)
        if "assistant\n\n" in response:
            response = response.split("assistant\n\n")[-1].strip()
        # Method 2: Look for "assistant\n" without double newline
        elif "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        # Method 3: Look for the exact prompt text and remove it
        elif prompt_text in response:
            response = response.replace(prompt_text, "").strip()
        # Method 4: Look for "user\n" marker and take everything after the last occurrence
        elif "user\n" in response:
            # Split by "user\n" and take the last part, which should be after all prompts
            parts = response.split("user\n")
            if len(parts) > 1:
                response = parts[-1].strip()
                # If there's still an "assistant" marker, remove it
                if response.startswith("assistant\n"):
                    response = response[len("assistant\n"):].strip()
        # Method 5: Look for specific header markers
        elif "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Clean up any remaining "user" prefix that might be at the start
        if response.startswith("user\n"):
            response = response[5:].strip()
        
        return response
    
    def __call__(self, prompts: Batch, *, system_prompt: str = "") -> Sequence[str]:
        """Process a batch of prompts through the model.
        
        Args:
            prompts: Batch of queries, each a tuple of text/image parts
            system_prompt: System prompt to use for all queries
            
        Returns:
            List of model responses, one per query
        """
        responses = []
        
        for query in prompts:
            # Check cache if enabled
            if self.use_cache:
                cache_key = get_cache_key(self.model_name, query, system_prompt)
                cached_response = cache.get(cache_key)
                if cached_response is not None and cached_response != "":
                    if self.verbose:
                        print(f"Using cached response")
                    responses.append(cached_response)
                    continue
            
            # Generate response with retry logic
            response_text = ""
            last_error = None
            
            for attempt in range(self.num_tries_per_request):
                try:
                    response_text = self._generate_single(query, system_prompt)
                    if response_text:
                        break
                    elif self.verbose:
                        print(f"Warning: Empty response (attempt {attempt + 1})")
                        
                except Exception as e:
                    last_error = e
                    if self.verbose:
                        print(f"❌ Error in Llama generation (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                    if attempt < self.num_tries_per_request - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            if not response_text:
                if self.verbose:
                    print(f"⚠️ Warning: Empty response from {self.model_name} after {self.num_tries_per_request} attempts")
                    if last_error:
                        print(f"   Last error: {last_error}")
                response_text = ""  # Return empty string on failure
            
            # Cache if enabled and successful
            if self.use_cache and response_text:
                cache.set(cache_key, response_text)
            
            responses.append(response_text)
        
        return responses