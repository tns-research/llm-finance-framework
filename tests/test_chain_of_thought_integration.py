#!/usr/bin/env python3
"""
Comprehensive integration tests for chain of thought reasoning feature.
Tests all 8 combinations of feature flags (strategic_journal × feeling_log × chain_of_thought).
"""

from unittest.mock import patch

import pytest

from src.backtest import parse_response_text
from src.dummy_model import dummy_call_model
from src.prompt_builder import PromptBuilder


class MockConfigManager:
    """Mock configuration manager for testing different feature flag combinations"""

    def __init__(
        self,
        enable_chain_of_thought=False,
        enable_strategic_journal=False,
        enable_feeling_log=False,
    ):
        self.enable_cot = enable_chain_of_thought
        self.enable_journal = enable_strategic_journal
        self.enable_feeling = enable_feeling_log

    def get_feature_flags(self):
        return {
            "ENABLE_CHAIN_OF_THOUGHT": self.enable_cot,
            "ENABLE_STRATEGIC_JOURNAL": self.enable_journal,
            "ENABLE_FEELING_LOG": self.enable_feeling,
            "ENABLE_TECHNICAL_INDICATORS": True,
            "SHOW_DATE_TO_LLM": False,
        }

    def get_symbol_info(self):
        return "SPY", "SPY ETF"

    def get_active_personality(self):
        from src.config_classes import TraderPersonality

        return TraderPersonality(
            name="balanced",
            description="balanced trader",
            bias_description="seeks moderate risk-adjusted returns",
        )


@pytest.mark.parametrize(
    "strategic,feeling,cot,expected_lines",
    [
        (False, False, False, 3),  # Baseline: decision, prob, explanation
        (False, False, True, 4),  # Chain of thought only
        (False, True, False, 4),  # Feeling log only
        (False, True, True, 5),  # Feeling + chain of thought
        (True, False, False, 4),  # Strategic journal only
        (True, False, True, 5),  # Strategic + chain of thought
        (True, True, False, 5),  # Strategic + feeling
        (True, True, True, 6),  # All features enabled
    ],
)
def test_feature_flag_combinations(strategic, feeling, cot, expected_lines):
    """Test all 8 combinations of feature flags for correct parsing and line counts"""

    # Patch global variables that dummy_call_model uses, matching the working end-to-end test
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", cot), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", strategic
    ), patch("src.config.ENABLE_FEELING_LOG", feeling):

        # Create mock config for this combination (for prompt generation)
        config_manager = MockConfigManager(cot, strategic, feeling)

        # Generate prompt
        prompt_builder = PromptBuilder(config_manager)
        prompt = prompt_builder.build_system_prompt()

        # Verify prompt contains expected elements
        has_cot_instructions = "Before making your trading decision" in prompt
        assert (
            has_cot_instructions == cot
        ), f"Chain of thought instructions presence mismatch for cot={cot}"

        # Generate dummy response
        response = dummy_call_model(prompt, "Sample user prompt about market data")

        # Count actual lines in response
        actual_lines = len([line for line in response.split("\n") if line.strip()])

        # Verify line count matches expectation
        assert (
            actual_lines == expected_lines
        ), f"Expected {expected_lines} lines, got {actual_lines} for config (s={strategic}, f={feeling}, c={cot})"

        # Verify parsing succeeds
        try:
            result = parse_response_text(response)
            assert len(result) == 6, f"Expected 6-tuple result, got {len(result)}"

            (
                decision,
                prob,
                explanation,
                chain_of_thought,
                strategic_journal,
                feeling_log,
            ) = result

            # Verify decision is valid
            assert decision in ("BUY", "HOLD", "SELL"), f"Invalid decision: {decision}"

            # Verify probability is valid
            assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

            # Verify chain of thought presence
            if cot:
                assert (
                    chain_of_thought != "Chain of thought reasoning disabled."
                ), "Chain of thought should be present when enabled"
                assert (
                    "Analyzing market conditions" in chain_of_thought
                ), "Chain of thought should contain analytical content"
            else:
                assert (
                    chain_of_thought == "Chain of thought reasoning disabled."
                ), "Chain of thought should be disabled message when flag is off"

            # Verify strategic journal presence
            if strategic:
                assert (
                    strategic_journal
                    != "Strategic journal disabled in this configuration."
                ), "Strategic journal should be present when enabled"
            else:
                assert (
                    strategic_journal
                    == "Strategic journal disabled in this configuration."
                ), "Strategic journal should be disabled when flag is off"

            # Verify feeling log presence
            if feeling:
                assert (
                    feeling_log != "Feeling log disabled in this configuration."
                ), "Feeling log should be present when enabled"
            else:
                assert (
                    feeling_log == "Feeling log disabled in this configuration."
                ), "Feeling log should be disabled when flag is off"

        except Exception as e:
            pytest.fail(
                f"Parsing failed for config (s={strategic}, f={feeling}, c={cot}): {e}"
            )


def test_chain_of_thought_content_quality():
    """Test that chain of thought responses contain proper analytical reasoning"""

    # Enable chain of thought, disable others for focused testing
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", True), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", False
    ), patch("src.config.ENABLE_FEELING_LOG", False):

        config_manager = MockConfigManager(
            enable_chain_of_thought=True,
            enable_strategic_journal=False,
            enable_feeling_log=False,
        )

        prompt_builder = PromptBuilder(config_manager)
        prompt = prompt_builder.build_system_prompt()

        # Generate multiple responses to test variety
        responses = [dummy_call_model(prompt, "Sample market data") for _ in range(5)]

        for response in responses:
            result = parse_response_text(response)
            chain_of_thought = result[3]  # Index 3 is chain_of_thought

            # Verify chain of thought contains analytical elements
            assert (
                len(chain_of_thought) > 50
            ), "Chain of thought should be substantial content"

            # Check for key analytical components (from the 5-step framework)
            analytical_elements = [
                "market conditions",
                "volatility",
                "momentum",
                "risk assessment",
                "performance",
                "market timing",
                "capital preservation",
            ]

            found_elements = sum(
                1
                for element in analytical_elements
                if element.lower() in chain_of_thought.lower()
            )
            assert (
                found_elements >= 3
            ), f"Chain of thought should contain multiple analytical elements, found {found_elements}"

            # Verify it's not just repeating the same content
            assert (
                "RSI" in chain_of_thought
                or "MACD" in chain_of_thought
                or "volatility" in chain_of_thought
            ), "Chain of thought should reference technical indicators or market factors"


def test_backward_compatibility():
    """Ensure existing functionality works when chain of thought is disabled"""

    # Test with chain of thought disabled - should behave like original system
    # Need to patch both the MockConfigManager for prompt building and config for dummy model
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", False), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", True
    ), patch("src.config.ENABLE_FEELING_LOG", True):

        config_manager = MockConfigManager(
            enable_chain_of_thought=False,
            enable_strategic_journal=True,
            enable_feeling_log=True,
        )

        prompt_builder = PromptBuilder(config_manager)
        prompt = prompt_builder.build_system_prompt()

        # Verify chain of thought instructions are NOT present
        assert (
            "Before making your trading decision" not in prompt
        ), "Chain of thought instructions should not appear when disabled"

        # Generate and parse response
        response = dummy_call_model(prompt, "Sample market data")
        result = parse_response_text(response)

        (
            decision,
            prob,
            explanation,
            chain_of_thought,
            strategic_journal,
            feeling_log,
        ) = result

        # Verify chain of thought is disabled message
        assert (
            chain_of_thought == "Chain of thought reasoning disabled."
        ), "Should show disabled message when feature is off"

        # Verify other features still work
        assert (
            strategic_journal != "Strategic journal disabled in this configuration."
        ), "Strategic journal should work when enabled"
        assert (
            feeling_log != "Feeling log disabled in this configuration."
        ), "Feeling log should work when enabled"

        # Verify line count is correct (decision, prob, explanation, strategic, feeling = 5 lines)
        lines = [line for line in response.split("\n") if line.strip()]
        assert (
            len(lines) == 5
        ), f"Expected 5 lines for config without chain of thought, got {len(lines)}"


def test_error_handling():
    """Test error handling for malformed responses"""

    # Test with insufficient lines - no patching needed as it uses default parsing
    try:
        parse_response_text("BUY\n0.5")
        assert False, "Should have raised ValueError for insufficient lines"
    except ValueError as e:
        assert "Expected at least" in str(e), f"Unexpected error message: {e}"

    # Test with invalid decision - patch to match baseline config (no features enabled)
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", False), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", False
    ), patch("src.config.ENABLE_FEELING_LOG", False):

        config_manager = MockConfigManager(
            enable_chain_of_thought=False,
            enable_strategic_journal=False,
            enable_feeling_log=False,
        )

        prompt_builder = PromptBuilder(config_manager)
        prompt = prompt_builder.build_system_prompt()

        # Manually create invalid response
        invalid_response = "INVALID_DECISION\n0.5\nSome explanation."

        try:
            parse_response_text(invalid_response)
            assert False, "Should have raised ValueError for invalid decision"
        except ValueError as e:
            assert "Invalid decision word" in str(e), f"Unexpected error message: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
