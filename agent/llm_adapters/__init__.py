from typing import Any, Dict
from agent.llm.provider import ProviderFactory

def create_adapter(model_config: Dict[str, Any]):
    """
    Factory function to create an LLM adapter based on the model configuration.
    This is a shim for backward compatibility.
    """
    provider_name = model_config.get("provider", "google")
    model_name = model_config.get("model")
    return ProviderFactory.get(provider_name, model_name, model_config)
