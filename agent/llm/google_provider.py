import os
from google import genai
from .provider import BaseProvider
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI

class GoogleProvider(BaseProvider):
    def __init__(self, model, config):
        super().__init__(model, config)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate(self, prompt, **kwargs):
        # The google-genai SDK has sync and async - choose accordingly
        model = genai.GenerativeModel(self.model)
        response = await model.generate_content_async(
            prompt,
        )
        text = response.text
        return {"text": text, "raw": response, "usage": getattr(response, "usage_metadata", None)}
