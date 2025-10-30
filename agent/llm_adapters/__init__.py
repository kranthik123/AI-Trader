from typing import Any, Dict
from .base_adapter import BaseAdapter

def create_adapter(model_config: Dict[str, Any]) -> BaseAdapter:
    """
    Factory function to create an LLM adapter based on the model configuration.

    Args:
        model_config: A dictionary containing the model configuration.

    Returns:
        An instance of the appropriate LLM adapter.
    """
    backend = model_config.get("backend", "openai_api")

    if backend == "openai_api":
        from .openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model_config)
    elif backend == "ollama_local":
        from .ollama_adapter import OllamaAdapter
        return OllamaAdapter(model_config)
    elif backend == "lmstudio_local":
        from .lmstudio_adapter import LMStudioAdapter
        return LMStudioAdapter(model_config)
    elif backend == "free_api":
        from .free_api_adapter import FreeApiAdapter
        return FreeApiAdapter(model_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
