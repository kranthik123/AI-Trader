from typing import Any, Dict
from .base_adapter import BaseAdapter

# Prefer the maintained langchain-ollama package (no deprecation warning).
# Fall back to the older langchain_community implementation if the new
# package isn't installed, and provide a clear error if neither is present.
try:
    from langchain_ollama import ChatOllama  # type: ignore
    _OLLAMA_SOURCE = "langchain_ollama"
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
        _OLLAMA_SOURCE = "langchain_community"
    except Exception:
        ChatOllama = None  # type: ignore
        _OLLAMA_SOURCE = None


class OllamaAdapter(BaseAdapter):
    """
    Adapter for Ollama models (local or cloud).

    This adapter prefers `langchain-ollama` (package name `langchain_ollama`).
    If that's not installed it will fall back to `langchain_community.chat_models.ChatOllama`.

    The adapter normalizes provided `server_url` by stripping any trailing
    `/api` to avoid duplicated paths like `.../api/api/...`. For cloud use it
    will set an Authorization header when an `api_key` is provided.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the Ollama adapter.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model_config = model_config

    def get_llm(self) -> Any:
        """Return a ChatOllama instance.

        Raises:
            RuntimeError: if no ChatOllama implementation is available.
        """
        if ChatOllama is None:
            raise RuntimeError(
                "No ChatOllama implementation available. Please install `langchain-ollama` "
                "(pip install -U langchain-ollama) or ensure `langchain_community` is present."
            )

        basemodel = self.model_config.get("basemodel", "")
        if not basemodel:
            raise ValueError("Ollama model config must include 'basemodel'")

        server_url = self.model_config.get("server_url", "http://localhost:11434")
        api_key = self.model_config.get("api_key", None)

        # Normalize server_url: remove trailing '/api' if present
        if server_url.endswith("/api"):
            base_url = server_url[: -len("/api")]
        else:
            base_url = server_url.rstrip("/")

        headers = None
        if api_key and isinstance(api_key, str) and api_key.strip() and api_key != "ollama":
            headers = {"Authorization": f"Bearer {api_key}"}

        # Create and return the ChatOllama client. Different packages may
        # accept slightly different parameter names but both support model/base_url/headers.
        return ChatOllama(model=basemodel, base_url=base_url, headers=headers)
