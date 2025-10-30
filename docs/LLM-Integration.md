# LLM Integration

This document describes how to configure and use the different LLM providers in the application.

## Configuration

The LLM configuration is located in `configs/models_config.yaml`. This file defines the available providers, models, and their settings.

### Toggles

You can enable or disable providers and models using the `enabled` flag in the configuration file.

- **Provider-level toggle:** `providers.<provider_name>.enabled`
- **Model-level toggle:** `providers.<provider_name>.models.<model_name>.enabled`

### Fallback Behavior

If a provider or model is disabled, the application will fall back to the default provider specified in the `default_provider` field.

## Environment Variables

You can override the configuration settings using environment variables.

- `GOOGLE_API_KEY`: Your Google API key.
- `OLLAMA_API_KEY`: Your Ollama API key.
- `OPENROUTER_API_KEY`: Your OpenRouter API key.
- `REDIS_URL`: The URL of your Redis instance.

## Adding a New Provider

To add a new provider, you need to:

1.  Create a new provider class that inherits from `BaseProvider`.
2.  Implement the `_generate` and `get_llm` methods.
3.  Add the provider to the `ProviderFactory`.
4.  Add the provider to the `configs/models_config.yaml` file.
