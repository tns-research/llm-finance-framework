#!/usr/bin/env python3
"""
Test script to verify the strategic journal configuration works correctly.
"""

import sys

sys.path.insert(0, ".")

from src.config import SYSTEM_PROMPT, ENABLE_STRATEGIC_JOURNAL, ENABLE_FEELING_LOG
from src.backtest import parse_response_text


def test_config():
    print("=" * 80)
    print("STRATEGIC JOURNAL CONFIGURATION TEST")
    print("=" * 80)
    print(f"\nENABLE_STRATEGIC_JOURNAL: {ENABLE_STRATEGIC_JOURNAL}")
    print(f"ENABLE_FEELING_LOG: {ENABLE_FEELING_LOG}")

    # Calculate expected lines
    expected_lines = 3
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

    # Build test response based on config
    test_lines = [
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

    try:
        decision, prob, explanation, journal, feeling = parse_response_text(
            test_response
        )
        print("\n✓ Parsing successful!")
        print(f"  Decision: {decision}")
        print(f"  Probability: {prob}")
        print(f"  Explanation: {explanation}")
        print(f"  Strategic Journal: {journal}")
        print(f"  Feeling Log: {feeling}")
    except Exception as e:
        print(f"\n✗ Parsing failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
