from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    """

    @abstractmethod
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the adapter with the model configuration.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        pass

    @abstractmethod
    def get_llm(self) -> Any:
        """
        Get the language model instance.

        Returns:
            An instance of the language model.
        """
        pass
