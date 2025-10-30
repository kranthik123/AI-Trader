import json
import yaml
from pathlib import Path

def migrate_config():
    """Migrates the default_config.json to models_config.yaml."""
    json_path = Path(__file__).parent.parent / "configs" / "default_config.json"
    yaml_path = Path(__file__).parent.parent / "configs" / "models_config.yaml"

    if not json_path.exists():
        print(f"❌ {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        config = json.load(f)

    # Transform the config to the new format
    new_config = {
        "default_provider": "google",
        "providers": {
            "google": {
                "enabled": True,
                "default_model": "gemini-pro-2.5",
                "models": {}
            },
            "ollama": {
                "enabled": False,
                "default_model": "glm-4.6:cloud",
                "api_base": "https://api.ollama.com",
                "models": {}
            },
            "openrouter": {
                "enabled": False,
                "api_base": "https://api.openrouter.ai/v1",
                "default_model": "minimax/minimax-m2:free",
                "models": {}
            }
        }
    }

    for model in config.get("models", []):
        provider = model.get("backend", "openai_api").replace("_api", "").replace("_local", "")
        if provider == "openai":
            # This is a bit of a guess, but we'll assume openai models are openrouter models
            provider = "openrouter"

        model_name = model.get("basemodel")

        if provider in new_config["providers"]:
            new_config["providers"][provider]["models"][model_name] = {
                "enabled": model.get("enabled", True),
                "display_name": model.get("name"),
            }


    # Add other config sections from the old config
    for key, value in config.items():
        if key != "models":
            new_config[key] = value


    with open(yaml_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

    print(f"✅ Successfully migrated {json_path} to {yaml_path}")

if __name__ == "__main__":
    migrate_config()
