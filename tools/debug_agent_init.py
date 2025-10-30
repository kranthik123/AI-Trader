import asyncio
import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is on sys.path for imports
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from main import load_config, get_agent_class

async def debug_init(config_path=None):
    config = load_config(config_path)
    enabled_models = [m for m in config['models'] if m.get('enabled', True)]
    if not enabled_models:
        print('No enabled models')
        return
    model_config = enabled_models[0]
    model_name = model_config.get('name')
    signature = model_config.get('signature')
    basemodel = model_config.get('basemodel')
    print(f"Debug initializing model: {model_name} ({signature}) basemodel={basemodel}")

    AgentClass = get_agent_class(config.get('agent_type', 'BaseAgent'))
    agent = AgentClass(
        model_config=model_config,
        stock_symbols=[],
        log_path="./data/agent_data",
        max_steps=1,
        max_retries=1,
        base_delay=0.1,
        initial_cash=10000.0,
        init_date=config['date_range']['init_date']
    )
    try:
        await agent.initialize()
        print('Initialization succeeded')
    except Exception as e:
        print('Initialization failed with exception:')
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(debug_init(config_path))
