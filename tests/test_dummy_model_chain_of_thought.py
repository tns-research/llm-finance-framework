#!/usr/bin/env python3
"""
Tests for dummy model chain of thought response generation.
Validates that dummy model generates proper chain of thought responses.
"""

from unittest.mock import patch

from src.backtest import parse_response_text
from src.dummy_model import dummy_call_model


def test_chain_of_thought_response_format():
    """Validate dummy model generates proper chain of thought responses for all combinations"""

    # Test all 8 combinations
    combinations = [
        (False, False, False, 3),  # Baseline
        (False, False, True, 4),  # Chain of thought only
        (False, True, False, 4),  # Feeling log only
        (False, True, True, 5),  # Feeling + chain of thought
        (True, False, False, 4),  # Strategic journal only
        (True, False, True, 5),  # Strategic + chain of thought
        (True, True, False, 5),  # Strategic + feeling
        (True, True, True, 6),  # All features
    ]

    for strategic, feeling, cot, expected_lines in combinations:
        # Mock the feature flags by patching config (same approach as test_feature_flag_synchronization.py)
        # This ensures parse_response_text reads the correct flags
        with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", cot), patch(
            "src.config.ENABLE_STRATEGIC_JOURNAL", strategic
        ), patch("src.config.ENABLE_FEELING_LOG", feeling):

            # Generate response
            response = dummy_call_model("Test system prompt", "Test user prompt")

            # Count lines
            lines = [line for line in response.split("\n") if line.strip()]
            actual_lines = len(lines)

            assert (
                actual_lines == expected_lines
            ), f"Expected {expected_lines} lines for config (s={strategic}, f={feeling}, c={cot}), got {actual_lines}"

            # Verify parsing succeeds
            result = parse_response_text(response)
            assert len(result) == 6, f"Should return 6-tuple, got {len(result)}"

            (
                decision,
                prob,
                explanation,
                chain_of_thought,
                strategic_journal,
                feeling_log,
            ) = result

            # Validate decision
            assert decision in ("BUY", "HOLD", "SELL"), f"Invalid decision: {decision}"

            # Validate probability
            assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

            # Validate chain of thought content
            if cot:
                assert (
                    chain_of_thought != "Chain of thought reasoning disabled."
                ), "Chain of thought should be present when enabled"
                assert (
                    "Analyzing market conditions" in chain_of_thought
                ), "Chain of thought should contain analytical framework"
            else:
                assert (
                    chain_of_thought == "Chain of thought reasoning disabled."
                ), "Chain of thought should be disabled message when flag is off"

            # Validate strategic journal content
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

            # Validate feeling log content
            if feeling:
                assert (
                    feeling_log != "Feeling log disabled in this configuration."
                ), "Feeling log should be present when enabled"
            else:
                assert (
                    feeling_log == "Feeling log disabled in this configuration."
                ), "Feeling log should be disabled when flag is off"

    print("✓ Chain of thought response format test passed!")


def test_chain_of_thought_content_quality():
    """Test that chain of thought responses contain proper analytical reasoning"""

    # Enable chain of thought by patching config
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", True), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", False
    ), patch("src.config.ENABLE_FEELING_LOG", False):

        # Generate multiple responses to test content variety
        responses = []
        for i in range(10):
            user_prompt = f"Market data scenario {i}: RSI at {50 + i*2}, MACD {'positive' if i % 2 == 0 else 'negative'}"
            response = dummy_call_model("Test system prompt", user_prompt)
            responses.append(response)

        # Analyze chain of thought content across responses
        chain_of_thoughts = []
        for response in responses:
            result = parse_response_text(response)
            chain_of_thoughts.append(result[3])  # Index 3 is chain_of_thought

        # Verify all contain analytical framework
        for i, cot in enumerate(chain_of_thoughts):
            assert len(cot) > 50, f"Chain of thought {i} should be substantial"
            assert (
                "Analyzing market conditions" in cot
            ), f"Chain of thought {i} should contain analytical framework"

            # Check for analytical elements
            analytical_terms = [
                "RSI",
                "MACD",
                "volatility",
                "momentum",
                "risk assessment",
                "performance",
            ]
            found_terms = sum(
                1 for term in analytical_terms if term.lower() in cot.lower()
            )
            assert (
                found_terms >= 2
            ), f"Chain of thought {i} should contain multiple analytical terms"

        # Note: Dummy model generates consistent responses for testing reliability
        # In a real model, responses would vary based on input data

        print("✓ Chain of thought content quality test passed!")
        print(f"  Generated {len(responses)} responses")
        print(
            f"  Average length: {sum(len(cot) for cot in chain_of_thoughts) / len(chain_of_thoughts):.0f} characters"
        )


if __name__ == "__main__":
    print("Running Dummy Model Chain of Thought Tests")
    print("=" * 50)

    test_chain_of_thought_response_format()
    test_chain_of_thought_content_quality()

    print("\n" + "=" * 50)
    print("✓ All dummy model tests passed!")
