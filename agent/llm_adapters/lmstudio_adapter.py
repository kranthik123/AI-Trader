from typing import Any, Dict
from langchain_community.llms import LlamaCpp
from .base_adapter import BaseAdapter

class LMStudioAdapter(BaseAdapter):
    """
    Adapter for LM Studio local models.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the LM Studio adapter.

        Args:
            model_config: A dictionary containing the model configuration.
        """
        self.model_config = model_config

    def get_llm(self) -> Any:
        """
        Get the LM Studio language model instance.

        Returns:
            An instance of the LlamaCpp language model.
        """
        return LlamaCpp(
            model_path=self.model_config["model_path"],
            n_gpu_layers=-1 if self.model_config.get("hardware_hint") == "gpu" else 0,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            verbose=True,
        )
