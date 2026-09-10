"""Google Gemini model adapter."""

import concurrent.futures
import os
import time
from typing import Optional, Sequence

from google import genai
from google.genai import types as genai_types

from .base import Batch, OneQuery
from .utils import cache, get_cache_key, image_to_base64, is_image, to_pil_image


class GoogleAdapter:
    """Google Gemini API adapter."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        num_tries_per_request: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        batch_size: int = 24,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """Initialize Google adapter.
        
        Args:
            model_name: Google model name
            api_key: Google API key (uses env var if not provided)
            num_tries_per_request: Number of retries for failed requests
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            batch_size: Maximum concurrent requests
            use_cache: Whether to use disk cache
            verbose: Whether to print error messages
        """
        self.model_name = model_name
        self.num_tries_per_request = num_tries_per_request
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.verbose = verbose

        self.use_vertex = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if self.use_vertex:
            try:
                from vertexai import init as vertex_init
                from vertexai.generative_models import GenerativeModel, Image as VertexImage
            except Exception as e:
                raise ValueError(f"Vertex AI SDK not available: {e}")

            project_id = (
                os.getenv("VERTEX_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or "surgery-483823"
            )
            location = os.getenv("VERTEX_LOCATION") or "us-central1"
            vertex_init(project=project_id, location=location)
            self.vertex_model = GenerativeModel(self.model_name)
            self.vertex_image_cls = VertexImage
        else:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            self.client = genai.Client(api_key=self.api_key)
    
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Process a batch of prompts through the model.
        
        Args:
            prompts: Batch of queries
            system_prompt: System prompt
            
        Returns:
            List of model responses
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [
                executor.submit(self._one_call, prompt=p, system_prompt=system_prompt)
                for p in prompts
            ]
            return [f.result() for f in futures]
    
    def _one_call(self, prompt: OneQuery, system_prompt: str) -> str:
        """Process a single prompt.
        
        Args:
            prompt: Single query tuple
            system_prompt: System prompt
            
        Returns:
            Model response string
        """
        if self.use_cache:
            ret = cache.get(get_cache_key(self.model_name, prompt, system_prompt))
            if ret is not None and ret != "":
                return ret
        
        # Build content parts from tuple
        content_parts = []
        
        # Add system prompt first
        if system_prompt:
            content_parts.append(system_prompt)
        
        def _to_vertex_image(pil_img):
            if hasattr(self.vertex_image_cls, "from_bytes"):
                from io import BytesIO
                buffer = BytesIO()
                pil_img.save(buffer, format="PNG")
                return self.vertex_image_cls.from_bytes(buffer.getvalue())
            if hasattr(self.vertex_image_cls, "from_pil_image"):
                return self.vertex_image_cls.from_pil_image(pil_img)
            return pil_img

        # Process user prompt parts
        for p in prompt:
            if isinstance(p, str):
                content_parts.append(p)
            elif is_image(p):
                # Convert to PIL if needed
                pil_image = to_pil_image(p)
                content_parts.append(_to_vertex_image(pil_image) if self.use_vertex else pil_image)
            else:
                raise ValueError(f"Invalid prompt type: {type(p)}")
        
        response_text = ""
        for _ in range(self.num_tries_per_request):
            try:
                if self.use_vertex:
                    response = self.vertex_model.generate_content(content_parts)
                else:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                        config=genai_types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                    )
                # Extract text from response
                response_text = response.text.strip() if response.text else ""
                if response_text:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error calling Google API for {self.model_name}: {e}")
                    print(f"   Attempt {_ + 1}/{self.num_tries_per_request}")
                time.sleep(3)
        
        if not response_text and self.verbose:
            print(f"⚠️  Warning: Empty response from {self.model_name} after {self.num_tries_per_request} attempts")
            print(f"   System prompt length: {len(system_prompt) if system_prompt else 0} chars")
            print(f"   Content parts: {len(content_parts)}")
        
        if self.use_cache and response_text:
            cache.set(get_cache_key(self.model_name, prompt, system_prompt), response_text)
        
        return response_text
