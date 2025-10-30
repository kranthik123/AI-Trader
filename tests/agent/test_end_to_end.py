import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agent.base_agent.base_agent import BaseAgent

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
            }
        },
        "agent_config": {
            "max_steps": 2,
        }
    }

@patch('langchain.agents.create_agent')
@patch('agent.base_agent.base_agent.BaseAgent.__init__', return_value=None)
@patch('agent.base_agent.base_agent.BaseAgent._ainvoke_with_retry', new_callable=AsyncMock)
@patch('agent.base_agent.base_agent.extract_conversation', return_value="The trading period is over.")
@pytest.mark.asyncio
async def test_end_to_end_happy_path(mock_extract_conversation, mock_ainvoke, mock_init, mock_create_agent, mock_config):
    """Tests the end-to-end happy path."""

    # Mock the agent's response
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="The trading period is over.")]})
    mock_create_agent.return_value = mock_agent

    agent = BaseAgent(config=mock_config, model_config={})
    agent.model = MagicMock()
    agent.tools = []
    agent.signature = "test"
    agent.max_steps = 2
    agent.agent = mock_agent
    agent.base_log_path = "/tmp"

    def _setup_logging(today_date):
        return "/dev/null"

    agent._setup_logging = _setup_logging

    def _log_message(log_file, new_messages):
        pass

    agent._log_message = _log_message

    async def _handle_trading_result(today_date):
        pass

    agent._handle_trading_result = _handle_trading_result


    with patch('langchain.agents.create_agent', return_value=mock_agent):
        await agent.run_trading_session("2025-10-13")

    # Assert that the generate method was called
    assert mock_ainvoke.called
