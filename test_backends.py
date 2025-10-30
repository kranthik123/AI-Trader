import os
import asyncio
from unittest.mock import patch
from main import load_config, get_agent_class

async def test_backend(model_config):
    """
    Tests a single backend configuration.
    """
    print(f"--- Testing backend: {model_config.get('backend', 'openai_api')} ---")
    try:
        AgentClass = get_agent_class("BaseAgent")
        agent = AgentClass(model_config=model_config)
        await agent.initialize()
        print(f"✅ Backend {model_config.get('backend', 'openai_api')} initialized successfully.")
    except Exception as e:
        print(f"❌ Backend {model_config.get('backend', 'openai_api')} failed to initialize: {e}")

async def main():
    """
    Runs tests for all backend configurations.
    """
    # Load the configuration
    config = load_config()
    models = config["models"]

    # Mock the MCP client to avoid connection errors
    with patch("langchain_mcp_adapters.client.MultiServerMCPClient.get_tools", return_value=[]):
        for model_config in models:
            if model_config["enabled"]:
                await test_backend(model_config)

if __name__ == "__main__":
    # Set dummy environment variables for testing
    os.environ["DEEPSEEK_API_BASE"] = "https://api.deepseek.com"
    os.environ["DEEPSEEK_API_KEY"] = "dummy_key"
    os.environ["GEMINI_API_BASE"] = "https://generativelanguage.googleapis.com"
    os.environ["GEMINI_API_KEY"] = "dummy_key"
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OLLAMA_API_KEY"] = "ollama"
    os.environ["LM_STUDIO_MODEL_PATH"] = "dummy_path"
    os.environ["FREE_API_URL"] = "https://api.example.com"
    os.environ["FREE_API_KEY"] = "dummy_key"

    asyncio.run(main())
