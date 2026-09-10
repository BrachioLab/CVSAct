"""Model adapters for endopoint."""

from typing import Optional
from .base import ModelAdapter, PromptPart, OneQuery, Batch
from .openai_gpt import OpenAIAdapter
from .anthropic_claude import AnthropicAdapter
from .google_gemini import GoogleAdapter
from .vllm import LLaVAModel, QwenVLModel, PixtralModel, DeepSeekVL2Model
try:
    from .raso_adapter import RASOAdapter
except Exception:
    RASOAdapter = None
try:
    from .peskavlp_adapter import PeskaVLPAdapter
except Exception:
    PeskaVLPAdapter = None
try:
    from .gonogonet_adapter import GoNoGoNetAdapter
except Exception:
    GoNoGoNetAdapter = None
try:
    from .cholenet_adapter import CholeNetAdapter
except Exception:
    CholeNetAdapter = None
try:
    from .endoscapesnet_adapter import EndoscapesNetAdapter
except Exception:
    EndoscapesNetAdapter = None
# Lazy/optional import
try:
    from .llama import LlamaAdapter  # only defined if transformers supports Mllama
except Exception:
    LlamaAdapter = None  # consumers must check before using


def create_model(model_id: str, use_cache: bool = True, verbose: bool = True, dataset: Optional[str] = None):
    """Create a model adapter based on model ID.
    
    Args:
        model_id: Model identifier string
        use_cache: Whether to use caching for responses
        verbose: Whether to enable verbose error logging
        dataset: Optional dataset name for model-specific configuration
        
    Returns:
        Appropriate model adapter instance
    """
    # Map model IDs to adapters
    if 'gonogo' in model_id.lower():
        adapter = GoNoGoNetAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'cholenet' in model_id.lower():
        adapter = CholeNetAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'endoscapes' in model_id.lower():
        adapter = EndoscapesNetAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'raso' in model_id.lower():
        adapter = RASOAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose, dataset=dataset)
    elif 'peskavlp' in model_id.lower():
        adapter = PeskaVLPAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose, dataset=dataset)
    elif 'gpt' in model_id.lower():
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'claude' in model_id.lower():
        adapter = AnthropicAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'gemini' in model_id.lower():
        adapter = GoogleAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'llama' in model_id.lower():
        adapter = LlamaAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    elif 'llava' in model_id.lower():
        adapter = LLaVAModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'qwen' in model_id.lower():
        adapter = QwenVLModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'pixtral' in model_id.lower():
        adapter = PixtralModel(model_id, use_cache=use_cache, verbose=verbose)
    elif 'deepseek' in model_id.lower():
        adapter = DeepSeekVL2Model(model_id, use_cache=use_cache, verbose=verbose)
    else:
        # Default to OpenAI adapter
        adapter = OpenAIAdapter(model_name=model_id, use_cache=use_cache, verbose=verbose)
    
    # Add model_id attribute for compatibility
    adapter.model_id = model_id
    return adapter


if LlamaAdapter is not None:
    __all__ = [
        "ModelAdapter",
        "PromptPart",
        "OneQuery", 
        "Batch",
        "OpenAIAdapter",
        "AnthropicAdapter",
        "GoogleAdapter",
        "LlamaAdapter",
        "RASOAdapter",
        "PeskaVLPAdapter",
        "GoNoGoNetAdapter",
        "CholeNetAdapter",
        "EndoscapesNetAdapter",
        "LLaVAModel",
        "QwenVLModel",
        "PixtralModel",
        "DeepSeekVL2Model",
        "create_model",
    ]
else:
    __all__ = [
        "ModelAdapter",
        "PromptPart",
        "OneQuery", 
        "Batch",
        "OpenAIAdapter",
        "AnthropicAdapter",
        "GoogleAdapter",
        "RASOAdapter",
        "PeskaVLPAdapter",
        "GoNoGoNetAdapter",
        "CholeNetAdapter",
        "EndoscapesNetAdapter",
        "LLaVAModel",
        "QwenVLModel",
        "PixtralModel",
        "DeepSeekVL2Model",
        "create_model",
    ]
