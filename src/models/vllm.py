"""
Vision-Language Model implementations for the endopoint package.
Supports open-source VLMs including LLaVA, Qwen-VL, Pixtral, and DeepSeek-VL.
"""

import time
from typing import List, Optional, Union
import concurrent.futures

import torch
try:
    import PIL.Image
    from PIL import Image as PILImage
except ImportError:
    PIL = None
    PILImage = None

# Import shared utilities from utils module
from .utils import (
    cache,
    get_cache_key,
    is_image,
    image_to_base64,
    to_pil_image
)


class LLaVAModel:
    """LLaVA 1.6 (LLaVA-NeXT) model implementation using vLLM for efficient inference."""
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        use_vllm: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.use_vllm = use_vllm
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the LLaVA model using vLLM or HuggingFace transformers."""
        if self.use_vllm:
            try:
                # Check if multiprocessing is set to spawn (required for vLLM with CUDA)
                import multiprocessing
                if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                    if self.verbose:
                        print("Warning: vLLM requires 'spawn' multiprocessing. Trying to set it...")
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                    except RuntimeError:
                        if self.verbose:
                            print("Could not set spawn method. Falling back to transformers.")
                        self.use_vllm = False
                        self._load_transformers_model()
                        return
                
                from vllm import LLM, SamplingParams
                
                if self.verbose:
                    print(f"Loading LLaVA model with vLLM: {self.model_name}")
                
                # vLLM configuration for LLaVA
                # Increase max_model_len for few-shot learning with multiple images
                self.model = LLM(
                    model=self.model_name,
                    dtype="auto",
                    max_model_len=8192,  # Increased from 4096 to handle few-shot examples
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 30},  # Allow up to 30 images per prompt
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.processor = None  # vLLM handles tokenization internally
                
            except (ImportError, Exception) as e:
                if self.verbose:
                    print(f"vLLM loading failed: {e}")
                    print("Falling back to HuggingFace transformers")
                self.use_vllm = False
                self._load_transformers_model()
        else:
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Fallback to HuggingFace transformers if vLLM is not available."""
        # Check model type and use appropriate class
        if "v1.6" in self.model_name or "next" in self.model_name.lower():
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
            model_class = LlavaNextForConditionalGeneration
            processor_class = LlavaNextProcessor
        else:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            model_class = LlavaForConditionalGeneration
            processor_class = AutoProcessor
        
        if self.verbose:
            print(f"Loading LLaVA model with HuggingFace: {self.model_name}")
        
        self.model = model_class.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.processor = processor_class.from_pretrained(self.model_name)
    
    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        """Process one or more prompts through the LLaVA model."""
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        
        # Batch processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]
    
    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Process a single prompt through the model."""
        # Check cache first
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            cached_result = cache.get(cache_key)
            if cached_result is not None and cached_result != "":
                return cached_result
        
        # Prepare inputs
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
        elif isinstance(prompt, tuple):
            # For vLLM, we need to build the prompt with <image> tokens interleaved
            texts_and_images = []
            images = []
            for p in prompt:
                if isinstance(p, str):
                    texts_and_images.append(('text', p))
                elif is_image(p):
                    images.append(to_pil_image(p))
                    texts_and_images.append(('image', None))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            
            # Build text prompt with <image> tokens for vLLM
            if self.use_vllm and images:
                text_parts = []
                for item_type, content in texts_and_images:
                    if item_type == 'text':
                        text_parts.append(content)
                    elif item_type == 'image':
                        text_parts.append('<image>')
                text_prompt = " ".join(text_parts)
            else:
                # For transformers, just concatenate text parts
                text_prompt = " ".join([content for item_type, content in texts_and_images if item_type == 'text'])
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Add system prompt if provided
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                if self.use_vllm:
                    response_text = self._generate_vllm(full_prompt, images)
                else:
                    response_text = self._generate_transformers(full_prompt, images)
                
                if response_text:
                    break
                elif self.verbose:
                    print(f"Warning: Empty response from LLaVA (attempt {attempt + 1})")
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in LLaVA generation for {self.model_name} (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                if attempt == self.num_tries_per_request - 1:
                    # On last attempt, return empty string rather than crashing
                    if self.verbose:
                        print(f"⚠️  Warning: Empty response from {self.model_name} after {self.num_tries_per_request} attempts")
                    print(f"Failed after {self.num_tries_per_request} attempts: {e}")
                    return ""
                time.sleep(3)
        
        # Cache the result
        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        
        return response_text
    
    def _generate_vllm(self, prompt: str, images: List[PIL.Image.Image]) -> str:
        """Generate response using vLLM."""
        if self.verbose:
            print(f"vLLM generate called with {len(images)} images")
        
        # Based on official vLLM example for LLaVA
        # Single image case (most common for our use)
        if images and len(images) == 1:
            outputs = self.model.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": images[0]  # vLLM expects single image, not list
                    }
                },
                sampling_params=self.sampling_params
            )
        elif images and len(images) > 1:
            # Multiple images - pass as list
            outputs = self.model.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": images  # List of PIL images for multi-image
                    }
                },
                sampling_params=self.sampling_params
            )
        else:
            # Text only
            outputs = self.model.generate(
                {"prompt": prompt},
                sampling_params=self.sampling_params
            )
        
        # Extract text from outputs
        generated_text = ""
        for o in outputs:
            generated_text += o.outputs[0].text
        
        if self.verbose and not generated_text:
            print("Warning: vLLM returned empty text")
        
        return generated_text.strip()
    
    def _generate_transformers(self, prompt: str, images: List[PIL.Image.Image]) -> str:
        """Generate response using HuggingFace transformers."""
        # Check if using LLaVA-NeXT
        is_llava_next = "v1.6" in self.model_name or "next" in self.model_name.lower()
        
        # Prepare inputs based on model type
        if is_llava_next:
            # LLaVA-NeXT uses conversation format
            content = []
            if images:
                content.append({"type": "text", "text": prompt})
                for _ in images:
                    content.append({"type": "image"})
            else:
                content = [{"type": "text", "text": prompt}]
            
            conversation = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            if images:
                inputs = self.processor(text=text, images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=text, return_tensors="pt")
        else:
            # Original LLaVA format
            if images:
                inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=prompt, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        
        # Decode - get only the generated part
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_len:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return output_text.strip()


class QwenVLModel:
    """Qwen-2.5-VL model implementation with vLLM support."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        use_vllm: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.use_vllm = use_vllm
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen-VL model."""
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                
                if self.verbose:
                    print(f"Loading Qwen-VL model with vLLM: {self.model_name}")
                
                self.model = LLM(
                    model=self.model_name,
                    dtype="auto",
                    max_model_len=8192,
                    trust_remote_code=True,
                    limit_mm_per_prompt={"image": 30},
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self.processor = None
                
            except ImportError:
                if self.verbose:
                    print("vLLM not available, falling back to HuggingFace transformers")
                self.use_vllm = False
                self._load_transformers_model()
        else:
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Load using HuggingFace transformers."""
        from transformers import AutoProcessor
        
        if self.verbose:
            print(f"Loading Qwen-VL model with HuggingFace: {self.model_name}")
        
        # Based on the reference vlm_utils.py, Qwen2.5-VL should be loaded this way
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            if self.verbose:
                print("Successfully loaded Qwen2_5_VLForConditionalGeneration")
        except ImportError as e:
            if self.verbose:
                print(f"Failed to import Qwen2_5_VLForConditionalGeneration: {e}")
                print("Attempting to load with AutoModelForCausalLM and trust_remote_code...")
            # Fallback to AutoModelForCausalLM with trust_remote_code
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Import the process_vision_info function if available
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            if self.verbose:
                print("Loaded qwen_vl_utils.process_vision_info")
        except ImportError:
            self.process_vision_info = None
            if self.verbose:
                print("qwen_vl_utils not available, will use standard processing")
    
    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        """Process prompts through the model."""
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]
    
    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Process a single prompt."""
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            cached_result = cache.get(cache_key)
            if cached_result is not None and cached_result != "":
                return cached_result
        
        # Prepare inputs with proper interleaving for few-shot
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
            structured_content = [("text", prompt)]
        elif isinstance(prompt, tuple):
            # Track interleaved structure for few-shot
            structured_content = []
            texts = []
            images = []
            
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                    structured_content.append(("text", p))
                elif is_image(p):
                    img = to_pil_image(p)
                    images.append(img)
                    structured_content.append(("image", img))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Format prompt for Qwen
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build user message with proper interleaving
        user_content = []
        
        # Check if this is a few-shot scenario (multiple images with text between them)
        has_multiple_images = len(images) > 1
        
        if has_multiple_images and isinstance(prompt, tuple):
            # Few-shot case: preserve the interleaved structure
            for content_type, content in structured_content:
                if content_type == "text" and content.strip():
                    user_content.append({"type": "text", "text": content})
                elif content_type == "image":
                    user_content.append({"type": "image", "image": content})
        else:
            # Simple case: all images then text (backward compatible)
            for img in images:
                user_content.append({"type": "image", "image": img})
            if text_prompt.strip():
                user_content.append({"type": "text", "text": text_prompt})
        
        messages.append({"role": "user", "content": user_content})
        
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                if self.use_vllm:
                    response_text = self._generate_vllm(messages)
                else:
                    response_text = self._generate_transformers(messages)
                
                if response_text:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in Qwen-VL generation for {self.model_name} (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)
        
        if not response_text and self.verbose:
            print(f"⚠️  Warning: Empty response from {self.model_name}")
            print(f"   System prompt length: {len(system_prompt)} chars")
            print(f"   Prompt items: {len(prompt)}")
        
        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        
        return response_text
    
    def _generate_vllm(self, messages: List[dict]) -> str:
        """Generate using vLLM."""
        if self.verbose:
            print(f"Qwen vLLM generate called")
        
        # For vLLM, we need to format the prompt manually since processor is None
        # Build the prompt in Qwen's chat format
        text_parts = []
        images = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                if isinstance(content, list):
                    # Extract text and images
                    user_text = ""
                    for item in content:
                        if item.get("type") == "text":
                            user_text = item["text"]
                        elif item.get("type") == "image":
                            images.append(item["image"])
                    # For Qwen with vLLM, try using the expected format
                    # Each image needs a placeholder token
                    image_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
                    text_parts.append(f"<|im_start|>user\n{image_tokens}{user_text}<|im_end|>")
                else:
                    text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant prompt
        text_parts.append("<|im_start|>assistant\n")
        text = "\n".join(text_parts)
        
        if self.verbose:
            print(f"Qwen prompt has {len(images)} images")
            print(f"Prompt preview: {text[:200]}...")
        
        # vLLM generate with proper format
        if images:
            # Use the new dictionary format for vLLM
            outputs = self.model.generate(
                {
                    "prompt": text,
                    "multi_modal_data": {"image": images if len(images) > 1 else images[0]}
                },
                sampling_params=self.sampling_params
            )
        else:
            outputs = self.model.generate(
                {"prompt": text},
                sampling_params=self.sampling_params
            )
        
        # Extract text from outputs
        generated_text = ""
        for o in outputs:
            generated_text += o.outputs[0].text
        
        return generated_text.strip()
    
    def _generate_transformers(self, messages: List[dict]) -> str:
        """Generate using transformers."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process vision info if qwen_vl_utils is available (for Qwen2.5-VL)
        if hasattr(self, 'process_vision_info') and self.process_vision_info is not None:
            # Use the Qwen-specific vision processing
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Standard processing
            images = []
            for msg in messages:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            images.append(item["image"])
            
            if images:
                inputs = self.processor(text=text, images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=text, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        
        # Decode only the generated part
        # inputs is a dict, so we need to access input_ids with dict notation
        input_ids = inputs['input_ids'] if 'input_ids' in inputs else inputs.get('input_ids')
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""


class PixtralModel:
    """Pixtral-12B model implementation with vLLM support."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Pixtral-12B-2409",  # Correct model name without full date
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load Pixtral model using vLLM."""
        from vllm import LLM, SamplingParams
        
        if self.verbose:
            print(f"Loading Pixtral model with vLLM: {self.model_name}")
        
        self.model = LLM(
            model=self.model_name,
            tokenizer_mode="mistral",
            dtype="auto",
            max_model_len=16384,
            limit_mm_per_prompt={"image": 10},
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    
    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        """Process prompts through the model."""
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]
    
    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Process a single prompt."""
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            cached_result = cache.get(cache_key)
            if cached_result is not None and cached_result != "":
                return cached_result
        
        # Prepare inputs with proper interleaving for few-shot
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
            structured_content = [("text", prompt)]
        elif isinstance(prompt, tuple):
            # Track interleaved structure for few-shot
            structured_content = []
            texts = []
            images = []
            
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                    structured_content.append(("text", p))
                elif is_image(p):
                    img = to_pil_image(p)
                    images.append(img)
                    structured_content.append(("image", img))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Build message format for Pixtral
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Build user message with proper interleaving
        user_content = []
        
        # Check if this is a few-shot scenario
        has_multiple_images = len(images) > 1
        
        if has_multiple_images and isinstance(prompt, tuple):
            # Few-shot case: preserve the interleaved structure
            for content_type, content in structured_content:
                if content_type == "text" and content.strip():
                    user_content.append({"type": "text", "text": content})
                elif content_type == "image":
                    # Convert PIL image to base64 for vLLM
                    img_base64 = image_to_base64(content)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    })
        else:
            # Simple case: all images then text (backward compatible)
            for img in images:
                img_base64 = image_to_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
            if text_prompt.strip():
                user_content.append({"type": "text", "text": text_prompt})
        
        messages.append({"role": "user", "content": user_content})
        
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                outputs = self.model.chat(messages, sampling_params=self.sampling_params)
                response_text = outputs[0].outputs[0].text.strip()
                
                if response_text:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in Pixtral generation for {self.model_name} (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)
        
        if not response_text and self.verbose:
            print(f"⚠️  Warning: Empty response from {self.model_name}")
            print(f"   System prompt length: {len(system_prompt)} chars")
            print(f"   Prompt items: {len(prompt)}")
        
        if self.use_cache and response_text:
            cache.set(cache_key, response_text)
        
        return response_text


class DeepSeekVL2Model:
    """DeepSeek-VL2 model implementation with vLLM support."""
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl2",
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        use_cache: bool = True,
        batch_size: int = 8,
        verbose: bool = True,
        device: str = "cuda",
        use_vllm: bool = True,
    ):
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose
        self.device = device
        self.use_vllm = use_vllm
        
        # Set model_id for compatibility
        self.model_id = model_name
        
        self._load_model()
    
    def _load_model(self):
        """Load DeepSeek-VL2 model using vLLM or HuggingFace transformers."""
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                
                if self.verbose:
                    print(f"Loading DeepSeek-VL2 model with vLLM: {self.model_name}")
                
                # Try to use smaller variant if main model fails
                model_to_load = self.model_name
                
                # Check if we should try the tiny variant for better compatibility
                if "deepseek-vl2" in self.model_name and not "tiny" in self.model_name and not "small" in self.model_name:
                    # Try the tiny variant first as it has fewer dependencies
                    if self.verbose:
                        print("Note: DeepSeek-VL2 full model may have vision model dependencies.")
                        print("Consider using 'deepseek-ai/deepseek-vl2-tiny' for better compatibility.")
                
                # vLLM configuration for DeepSeek-VL2 with required architecture override
                self.model = LLM(
                    model=model_to_load,
                    trust_remote_code=True,
                    max_model_len=4096,  # Match model's max_position_embeddings
                    dtype="float16",
                    gpu_memory_utilization=0.9,
                    # Required architecture override for DeepSeek-VL2
                    hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
                    # Disable custom vision processor if it's causing issues
                    disable_mm_preprocessor_cache=True,
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop_token_ids=None,
                )
                
                # When using vLLM, we don't need the processor
                self.processor = None
                
                if self.verbose:
                    print("DeepSeek-VL2 model loaded successfully with vLLM")
                
                # Success - return early to avoid loading transformers
                return
                
            except RuntimeError as e:
                if "Unknown model" in str(e) and "vit_so400m" in str(e):
                    if self.verbose:
                        print(f"vLLM loading failed due to missing vision model: {e}")
                        print("This is a known issue with DeepSeek-VL2 full model.")
                        print("Try using 'deepseek-ai/deepseek-vl2-tiny' instead.")
                        print("Falling back to HuggingFace transformers (requires deepseek_vl2 package)")
                else:
                    if self.verbose:
                        print(f"vLLM loading failed: {e}")
                        print("Falling back to HuggingFace transformers")
                self.use_vllm = False
                # Fall through to load transformers model
            except Exception as e:
                if self.verbose:
                    print(f"vLLM loading failed: {e}")
                    print("Falling back to HuggingFace transformers")
                self.use_vllm = False
                # Fall through to load transformers model
        
        # Load with transformers (fallback or explicit choice)
        self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Fallback to HuggingFace transformers if vLLM is not available."""
        from transformers import AutoModelForCausalLM
        
        if self.verbose:
            print(f"Loading DeepSeek-VL2 model with HuggingFace: {self.model_name}")
        
        # Import DeepSeek-specific modules
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        except ImportError:
            raise ImportError(
                "DeepSeek-VL2 requires the deepseek_vl2 package. "
                "Install it with: pip install deepseek-vl2"
            )
        
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        
        # Load model without specifying dtype, then convert
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Convert entire model to float16 to avoid dtype mismatches
        self.model = self.model.half()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
    
    def __call__(self, prompts: Union[str, List[Union[str, tuple]]], system_prompt: Optional[str] = None):
        """Process prompts through the model."""
        if isinstance(prompts, (str, tuple)):
            return self.one_call(prompts, system_prompt=system_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self.one_call, prompt=p, system_prompt=system_prompt) for p in prompts]
            return [f.result() for f in futures]
    
    def one_call(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Process a single prompt."""
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            cached_result = cache.get(cache_key)
            if cached_result is not None and cached_result != "":
                return cached_result
        
        if self.use_vllm:
            response_text = self._generate_vllm(prompt, system_prompt)
        else:
            response_text = self._generate_transformers(prompt, system_prompt)
        
        if self.use_cache and response_text:
            cache_key = get_cache_key(self.model_name, prompt, system_prompt)
            cache.set(cache_key, response_text)
        
        return response_text
    
    def _generate_vllm(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Generate response using vLLM."""
        # Prepare inputs with proper interleaving
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
            structured_content = [("text", prompt)]
        elif isinstance(prompt, tuple):
            # Track interleaved structure for few-shot
            structured_content = []
            texts = []
            images = []
            
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                    structured_content.append(("text", p))
                elif is_image(p):
                    img = to_pil_image(p)
                    images.append(img)
                    structured_content.append(("image", img))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Build conversation format for DeepSeek-VL2 with vLLM
        if system_prompt:
            content = f"System: {system_prompt}\n\n"
        else:
            content = ""
        
        # Check if this is a few-shot scenario
        has_multiple_images = len(images) > 1
        
        if has_multiple_images and isinstance(prompt, tuple):
            # Few-shot case: properly interleave text and images
            for content_type, item in structured_content:
                if content_type == "text" and item.strip():
                    content += item + " "
                elif content_type == "image":
                    content += "<image>\n"
        else:
            # Simple case: all images then text (backward compatible)
            content += '<image>\n' * len(images) + text_prompt
        
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                # vLLM generates with inputs as a list (positional argument)
                inputs = [{
                    "prompt": content,
                    "multi_modal_data": {"image": images} if images else {}
                }]
                outputs = self.model.generate(
                    inputs,  # Pass as positional argument, not keyword
                    sampling_params=self.sampling_params,
                )
                
                if outputs and len(outputs) > 0:
                    response_text = outputs[0].outputs[0].text.strip()
                    if response_text:
                        break
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in DeepSeek-VL2 vLLM generation for {self.model_name} (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                if attempt == self.num_tries_per_request - 1:
                    response_text = ""
                    if self.verbose:
                        print(f"⚠️  Warning: Empty response from {self.model_name} after {self.num_tries_per_request} attempts")
        
        return response_text
    
    def _generate_transformers(self, prompt, system_prompt: Optional[str] = None) -> str:
        """Generate response using HuggingFace transformers (fallback)."""
        # Prepare inputs
        if isinstance(prompt, str):
            text_prompt = prompt
            images = []
        elif isinstance(prompt, tuple):
            texts = []
            images = []
            for p in prompt:
                if isinstance(p, str):
                    texts.append(p)
                elif is_image(p):
                    images.append(to_pil_image(p))
                else:
                    raise ValueError(f"Invalid prompt type: {type(p)}")
            text_prompt = " ".join(texts)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Build conversation for DeepSeek-VL2
        messages = []
        
        # Add images and text
        content = '<image>\n' * len(images) + f'<|ref|>{text_prompt}<|/ref|>'
        messages.append({'role': '<|User|>', 'content': content})
        messages.append({'role': '<|Assistant|>', 'content': ''})
        
        response_text = ""
        for attempt in range(self.num_tries_per_request):
            try:
                # Prepare inputs
                prepare_inputs = self.processor(
                    conversations=messages,
                    images=images,
                    force_batchify=True,
                    system_prompt=system_prompt
                )
                
                # Move to device - handle the special BatchCollateOutput type
                if hasattr(prepare_inputs, 'to'):
                    # This is a BatchCollateOutput which needs device as positional arg
                    prepare_inputs = prepare_inputs.to(self.model.device)
                
                # Get image embeddings
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                # Force inputs_embeds to float16 regardless of original dtype
                inputs_embeds = inputs_embeds.to(torch.float16)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.language.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        bos_token_id=self.processor.tokenizer.bos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                        use_cache=True
                    )
                
                response_text = self.processor.tokenizer.decode(
                    outputs[0].cpu().tolist(), 
                    skip_special_tokens=True
                ).strip()
                
                if response_text:
                    break
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in DeepSeek-VL2 generation for {self.model_name} (attempt {attempt + 1}/{self.num_tries_per_request}): {e}")
                time.sleep(3)
        
        return response_text


def load_vlm_model(model_name: str, **kwargs):
    """Factory function to load VLM models by name."""
    model_name_lower = model_name.lower()
    
    if "llava" in model_name_lower:
        return LLaVAModel(model_name=model_name, **kwargs)
    elif "qwen" in model_name_lower:
        return QwenVLModel(model_name=model_name, **kwargs)
    elif "pixtral" in model_name_lower:
        return PixtralModel(model_name=model_name, **kwargs)
    elif "deepseek" in model_name_lower:
        return DeepSeekVL2Model(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown VLM model: {model_name}")
