import os
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from .base_adapter import BaseAdapter

class FreeApiAdapter(BaseAdapter):
    """
    Adapter for free API-based models.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the Free API adapter.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model_config = model_config

    def get_llm(self) -> Any:
        """
        Get the Free API language model instance.

        Returns:
            An instance of the ChatOpenAI language model configured for the free API.
        """
        api_key = os.getenv(self.model_config["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {self.model_config['api_key_env']} not set.")

        return ChatOpenAI(
            model=self.model_config["basemodel"],
            base_url=self.model_config["api_url"],
            api_key=api_key,
            max_retries=3,
            timeout=30
        )
