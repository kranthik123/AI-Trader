from typing import Any, Dict
from .base_adapter import BaseAdapter

def create_adapter(model_config: Dict[str, Any]) -> BaseAdapter:
    """
    Factory function to create an LLM adapter based on the model configuration.

    Args:
        model_config: A dictionary containing the model configuration.
        
    Supported backends:
    - openai_api: For OpenAI-compatible APIs (Gemini, Groq)
    - ollama_cloud: For Ollama cloud services (Minimax-M2)

    Returns:
        An instance of the appropriate LLM adapter.
    """
    backend = model_config.get("backend", "openai_api")

    if backend == "openai_api":
        from .openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model_config)
    elif backend == "ollama_cloud":
        # Ollama cloud uses the same adapter code; expect `server_url` and `api_key` to point
        # to the Ollama Cloud endpoint and API key (set via environment/.env).
        from .ollama_adapter import OllamaAdapter
        return OllamaAdapter(model_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
