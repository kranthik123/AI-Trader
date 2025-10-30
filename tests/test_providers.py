import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from agent.llm.provider import ProviderFactory

@pytest.fixture
def mock_config():
    """Returns a mock config for testing."""
    return {
        "default_provider": "google",
        "providers": {
            "google": {
                "enabled": True,
                "default_model": "gemini-pro-2.5",
                "models": {
                    "gemini-pro-2.5": {
                        "enabled": True,
                        "display_name": "Google Gemini Pro 2.5",
                        "type": "google"
                    }
                }
            },
            "ollama": {
                "enabled": True,
                "default_model": "glm-4.6:cloud",
                "api_base": "https://api.ollama.com",
                "models": {
                    "glm-4.6:cloud": {
                        "enabled": True
                    }
                }
            },
            "openrouter": {
                "enabled": True,
                "api_base": "https://api.openrouter.ai/v1",
                "default_model": "minimax/minimax-m2:free",
                "models": {
                    "minimax/minimax-m2:free": {
                        "enabled": True
                    }
                }
            }
        }
    }

@pytest.mark.asyncio
async def test_google_provider(mock_config):
    """Tests the Google provider."""
    with patch('os.getenv') as mock_getenv, \
         patch('agent.llm.google_provider.genai') as mock_genai:

        mock_getenv.side_effect = lambda key, default=None: {
            "GOOGLE_API_KEY": "test_key",
            "REDIS_URL": None
        }.get(key, default)

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test response from Google."
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        provider = ProviderFactory.get("google", "gemini-pro-2.5", mock_config)
        response = await provider.generate("test prompt")

        assert response["text"] == "This is a test response from Google."

@pytest.mark.asyncio
async def test_ollama_provider(mock_config):
    """Tests the Ollama provider."""
    with patch('os.getenv', return_value=None), \
         patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response from Ollama."}
        mock_post.return_value = mock_response

        provider = ProviderFactory.get("ollama", "glm-4.6:cloud", mock_config)
        response = await provider.generate("test prompt")

        assert response["text"] == "This is a test response from Ollama."

@pytest.mark.asyncio
async def test_openrouter_provider(mock_config):
    """Tests the OpenRouter provider."""
    with patch('os.getenv', return_value=None), \
         patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is a test response from OpenRouter."
                }
            }]
        }
        mock_post.return_value = mock_response

        provider = ProviderFactory.get("openrouter", "minimax/minimax-m2:free", mock_config)
        response = await provider.generate("test prompt")

        assert response["text"] == "This is a test response from OpenRouter."

@pytest.mark.asyncio
async def test_fallback_behavior(mock_config):
    """Tests the fallback behavior when a provider is disabled."""
    mock_config["providers"]["google"]["enabled"] = False

    # Since google is disabled, it should fall back to the default provider, which is also google
    # so it should raise a ValueError.
    with pytest.raises(ValueError):
        ProviderFactory.get("google", "gemini-pro-2.5", mock_config)


    # Now, let's set a different default provider.
    mock_config["default_provider"] = "ollama"
    mock_config["providers"]["google"]["enabled"] = False

    with patch('os.getenv', return_value=None), \
         patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_post:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "This is a test response from the fallback provider."}
        mock_post.return_value = mock_response

        provider = ProviderFactory.get("google", "gemini-pro-2.5", mock_config)
        response = await provider.generate("test prompt")

        assert response["text"] == "This is a test response from the fallback provider."
