import os
from .provider import BaseProvider
from .async_client import get_http_client
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI

class OpenRouterProvider(BaseProvider):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.base = config.get("api_base", "https://openrouter.ai/api/v1")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self._client = get_http_client()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate(self, prompt, **kwargs):
        url = f"{self.base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        r = await self._client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

        if "choices" in data and len(data["choices"]) > 0:
            text = data["choices"][0].get("message", {}).get("content", "")
        else:
            text = data.get("text", "")

        return {"text": text, "raw": data}
