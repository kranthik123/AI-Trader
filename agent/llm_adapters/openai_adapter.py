import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from .base_adapter import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI models.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the OpenAI adapter.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model_config = model_config

    def get_llm(self) -> Any:
        """
        Get the OpenAI language model instance.

        Returns:
            An instance of the ChatOpenAI language model.
        """
        base_url = self.model_config.get("openai_base_url") or os.getenv("OPENAI_API_BASE")
        api_key = self.model_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "dummy_key")

        return ChatOpenAI(
            model=self.model_config["basemodel"],
            base_url=base_url,
            api_key=api_key,
            max_retries=3,
            timeout=30
        )
