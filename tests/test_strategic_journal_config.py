#!/usr/bin/env python3
"""
Test script to verify the strategic journal configuration works correctly.
"""

import pytest

from src.backtest import parse_response_text
from src.config import (
    ENABLE_CHAIN_OF_THOUGHT,
    ENABLE_FEELING_LOG,
    ENABLE_STRATEGIC_JOURNAL,
    SYSTEM_PROMPT,
)


def test_config():
    print("=" * 80)
    print("STRATEGIC JOURNAL CONFIGURATION TEST")
    print("=" * 80)
    print(f"\nENABLE_STRATEGIC_JOURNAL: {ENABLE_STRATEGIC_JOURNAL}")
    print(f"ENABLE_FEELING_LOG: {ENABLE_FEELING_LOG}")

    # Calculate expected lines
    expected_lines = 3
    if ENABLE_CHAIN_OF_THOUGHT:
        expected_lines += 1
    if ENABLE_STRATEGIC_JOURNAL:
        expected_lines += 1
    if ENABLE_FEELING_LOG:
        expected_lines += 1

    print(f"\nExpected response lines: {expected_lines}")

    print("\n" + "=" * 80)
    print("GENERATED SYSTEM PROMPT:")
    print("=" * 80)
    print(SYSTEM_PROMPT)

    # Test parsing
    print("\n" + "=" * 80)
    print("TESTING RESPONSE PARSING:")
    print("=" * 80)

    # Build a response matching the parser's line order for the current config:
    # [chain_of_thought], decision, prob, explanation, [journal], [feeling].
    test_lines = []

    if ENABLE_CHAIN_OF_THOUGHT:
        test_lines.append(
            "RSI neutral and MACD weak; risk moderate; strategic review favors a cautious long."
        )

    test_lines += [
        "BUY",
        "0.65",
        "The market shows positive momentum with increasing volume.",
    ]

    if ENABLE_STRATEGIC_JOURNAL:
        test_lines.append(
            "Yesterday's HOLD was cautious but we missed gains. Will be more aggressive on clear signals."
        )

    if ENABLE_FEELING_LOG:
        test_lines.append(
            "Feeling more confident after reviewing recent performance metrics."
        )

    test_response = "\n".join(test_lines)

    print(f"\nTest response ({len(test_lines)} lines):")
    print("-" * 40)
    print(test_response)
    print("-" * 40)

    decision, prob, explanation, chain_of_thought, journal, feeling = (
        parse_response_text(test_response)
    )
    print("\n✓ Parsing successful!")
    print(f"  Decision: {decision}")
    print(f"  Probability: {prob}")
    print(f"  Explanation: {explanation}")
    print(f"  Chain of Thought: {chain_of_thought}")
    print(f"  Strategic Journal: {journal}")
    print(f"  Feeling Log: {feeling}")

    assert len(test_lines) == expected_lines
    assert decision == "BUY"
    assert prob == pytest.approx(0.65)

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_config()
