from typing import Any, Dict
from langchain_openai import ChatOpenAI
from .base_adapter import BaseAdapter

class OllamaAdapter(BaseAdapter):
    """
    Adapter for Ollama models.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the Ollama adapter.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model_config = model_config

    def get_llm(self) -> Any:
        """
        Get the Ollama language model instance.

        Returns:
            An instance of the ChatOpenAI language model configured for Ollama.
        """
        return ChatOpenAI(
            model=self.model_config["basemodel"],
            base_url=self.model_config["server_url"],
            api_key=self.model_config.get("api_key", "ollama"),
            max_retries=3,
            timeout=30
        )
