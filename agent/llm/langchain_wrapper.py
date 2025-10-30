from typing import Any, Dict, List, Optional
from langchain_core.language_models.llms import LLM
from .provider import BaseProvider

class LangChainProviderWrapper(LLM):
    """A LangChain-compatible wrapper for the BaseProvider."""

    provider: BaseProvider

    def __init__(self, provider: BaseProvider, **kwargs: Any):
        super().__init__(provider=provider, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        # This is a synchronous wrapper for the async generate method.
        # For a truly async application, you would use _acall.
        import asyncio
        return asyncio.run(self.provider.generate(prompt, **kwargs))["text"]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        response = await self.provider.generate(prompt, **kwargs)
        return response["text"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.provider.model,
            "provider": self.provider.__class__.__name__,
        }
