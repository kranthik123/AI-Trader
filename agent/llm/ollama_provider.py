import os
from .provider import BaseProvider
from .async_client import get_http_client
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.llms import Ollama

class OllamaProvider(BaseProvider):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.base = config.get("api_base", os.getenv("OLLAMA_API_BASE", "https://api.ollama.com"))
        self.api_key = os.getenv("OLLAMA_API_KEY")
        self._client = get_http_client()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate(self, prompt, **kwargs):
        url = f"{self.base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": kwargs.get("max_tokens", 1024),
                "stop": kwargs.get("stop")
            }
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = await self._client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data.get("response")
        return {"text": text, "raw": data}
