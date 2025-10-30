from abc import ABC, abstractmethod
from typing import Any, Dict
import json
import logging
import hashlib
import time
from .cache import Cache
from .metrics import LLM_REQUESTS_TOTAL, LLM_REQUEST_LATENCY_SECONDS, LLM_ERRORS_TOTAL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaseProvider(ABC):
    def __init__(self, model: str, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.cache = Cache()

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text given a prompt. Return unified response dict with keys:
           { 'text': str, 'raw': object, 'usage': {...} }
        """
        provider_name = self.__class__.__name__
        LLM_REQUESTS_TOTAL.labels(provider=provider_name, model=self.model).inc()
        start_time = time.time()

        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        logging.info(f"Generating response for prompt_hash: {prompt_hash}")

        cache_key = self.cache.get_cache_key(
            provider=provider_name,
            model=self.model,
            prompt=prompt,
            **kwargs
        )

        cached_response = await self.cache.get(cache_key)
        if cached_response:
            logging.info(f"Returning cached response for prompt_hash: {prompt_hash}")
            # The raw response from Google and Ollama is not serializable to JSON
            # so we'll just return the text for now.
            try:
                return json.loads(cached_response)
            except (json.JSONDecodeError, TypeError):
                 return {"text": cached_response, "raw": None, "usage": None}

        try:
            response = await self._generate(prompt, **kwargs)
            logging.info(f"Successfully generated response for prompt_hash: {prompt_hash}")

            # The raw response from Google and Ollama is not serializable to JSON
            # so we'll just cache the text for now.
            try:
                await self.cache.set(cache_key, json.dumps(response))
            except TypeError:
                await self.cache.set(cache_key, response.get("text", ""))

            return response
        except Exception as e:
            LLM_ERRORS_TOTAL.labels(provider=provider_name, model=self.model).inc()
            logging.error(f"Error generating response for prompt_hash: {prompt_hash}", exc_info=True)
            raise e
        finally:
            duration = time.time() - start_time
            LLM_REQUEST_LATENCY_SECONDS.labels(provider=provider_name, model=self.model).observe(duration)


    @abstractmethod
    async def _generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text given a prompt. Return unified response dict with keys:
           { 'text': str, 'raw': object, 'usage': {...} }
        """
        pass

    def get_llm(self) -> Any:
        """Get the language model instance."""
        from .langchain_wrapper import LangChainProviderWrapper
        return LangChainProviderWrapper(provider=self)

class ProviderFactory:
    @staticmethod
    def get(provider_name: str, model: str, config: Dict[str, Any]) -> "BaseProvider":
        provider_config = config.get("providers", {}).get(provider_name, {})
        model_config = provider_config.get("models", {}).get(model, {})

        if not provider_config.get("enabled", False) or not model_config.get("enabled", False):
            print(f"⚠️ Provider '{provider_name}' or model '{model}' is disabled. Falling back to default provider.")
            default_provider_name = config.get("default_provider")
            if not default_provider_name or default_provider_name == provider_name:
                 raise ValueError("Default provider is not configured or is the same as the disabled provider.")

            default_provider_config = config.get("providers", {}).get(default_provider_name, {})
            default_model_name = default_provider_config.get("default_model")

            if not default_model_name:
                raise ValueError("Default model is not configured for the default provider.")

            return ProviderFactory.get(default_provider_name, default_model_name, config)


        if provider_name == "google":
            from .google_provider import GoogleProvider
            return GoogleProvider(model, provider_config)
        elif provider_name == "ollama":
            from .ollama_provider import OllamaProvider
            return OllamaProvider(model, provider_config)
        elif provider_name == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(model, provider_config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
