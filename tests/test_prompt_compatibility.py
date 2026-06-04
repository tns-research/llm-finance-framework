"""
Guard tests for the system prompt produced in production.

The system prompt is built by PromptBuilder, and SYSTEM_PROMPT() in config.py
is a thin wrapper over it. These tests assert on the content of the produced
prompt and that the production wrapper stays in sync with PromptBuilder, rather
than comparing two parallel implementations.
"""

from src import config
from src.configuration_manager import ConfigurationManager
from src.prompt_builder import PromptBuilder


def test_prompt_builder_chain_of_thought_content():
    """PromptBuilder includes chain of thought instructions when enabled."""
    config_manager = ConfigurationManager()

    if config_manager._config.active_experiment in config_manager._config.experiments:
        exp_config = config_manager._config.experiments[
            config_manager._config.active_experiment
        ]
        exp_config.features.memory.chain_of_thought = True

    prompt = PromptBuilder(config_manager).build_system_prompt()

    assert (
        "structured analytical process" in prompt.lower()
    ), "PromptBuilder should include chain of thought instructions when enabled"


def test_system_prompt_has_expected_structure():
    """The production system prompt has the expected behavioral and format content."""
    prompt = config.SYSTEM_PROMPT()

    assert len(prompt) > 500, "System prompt should have substantial content"
    assert "Your role is to beat the" in prompt, "Prompt should state the role"
    assert "Behavioral profile" in prompt, "Prompt should include behavioral profile"
    assert "Output format (strict)" in prompt, "Prompt should include output format"
    assert "BUY or HOLD or SELL" in prompt, "Prompt should list the trading actions"


def test_system_prompt_is_prompt_builder_wrapper():
    """SYSTEM_PROMPT() must stay a thin wrapper over PromptBuilder, same config."""
    config_manager = ConfigurationManager()

    if config_manager._config.active_experiment in config_manager._config.experiments:
        exp_config = config_manager._config.experiments[
            config_manager._config.active_experiment
        ]
        exp_config.features.memory.chain_of_thought = config.ENABLE_CHAIN_OF_THOUGHT

    expected = PromptBuilder(config_manager).build_system_prompt()

    assert config.SYSTEM_PROMPT() == expected, (
        "SYSTEM_PROMPT() should delegate to PromptBuilder; it has drifted from "
        "the production prompt builder."
    )
