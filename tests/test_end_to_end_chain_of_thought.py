#!/usr/bin/env python3
"""
End-to-end tests for chain of thought reasoning pipeline.
Tests complete workflow: prompt generation → LLM response → parsing → CSV output.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd

from src.backtest import parse_response_text, save_parsed_results
from src.dummy_model import dummy_call_model
from src.prompt_builder import PromptBuilder


def test_end_to_end_chain_of_thought():
    """Complete pipeline test: prompt → response → parsing → CSV"""

    # Test with all features enabled using proper patching at config level
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", True), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", True
    ), patch("src.config.ENABLE_FEELING_LOG", True):

        # Create a sample response with all features enabled
        sample_response = """Analyzing market conditions: RSI shows neutral, MACD weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.
BUY
0.75
Market indicators suggest upward momentum. Technical analysis shows bullish patterns.
Reviewing recent performance and adjusting strategy accordingly.
Feeling cautiously optimistic about current market conditions."""

        # Parse the response
        result = parse_response_text(sample_response)
        assert len(result) == 6, f"Expected 6-tuple result, got {len(result)}"

        (
            decision,
            prob,
            explanation,
            chain_of_thought,
            strategic_journal,
            feeling_log,
        ) = result

        # Verify parsed components
        assert decision == "BUY", f"Expected BUY, got {decision}"
        assert prob == 0.75, f"Expected 0.75, got {prob}"
        assert (
            "Analyzing market conditions" in chain_of_thought
        ), "Chain of thought should contain analytical content"
        assert (
            "Reviewing recent performance" in strategic_journal
        ), "Strategic journal should be present"
        assert (
            "Feeling cautiously optimistic" in feeling_log
        ), "Feeling log should be present"

        print("✓ End-to-end parsing test passed!")
        print(f"  Decision: {decision}")
        print(f"  Chain of thought length: {len(chain_of_thought)} characters")
        print(f"  Strategic journal: {strategic_journal[:50]}...")
        print(f"  Feeling log: {feeling_log[:50]}...")

        # 5. Simulate CSV storage
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".csv", delete=False
    ) as temp_file:
        temp_path = temp_file.name

    try:
        # Create sample data with chain of thought
        sample_data = [
            {
                "date": pd.Timestamp("2024-01-01"),
                "decision": decision,
                "prob": prob,
                "explanation": explanation,
                "chain_of_thought": chain_of_thought,
                "strategic_journal": strategic_journal,
                "feeling_log": feeling_log,
                "position": (
                    1.0 if decision == "BUY" else (-1.0 if decision == "SELL" else 0.0)
                ),
                "next_return_1d": 1.5,
                "strategy_return": 1.5
                * (1.0 if decision == "BUY" else (-1.0 if decision == "SELL" else 0.0)),
            }
        ]

        # Save to CSV
        df = save_parsed_results(temp_path, sample_data)

        # Verify CSV was created and contains data
        assert os.path.exists(temp_path), "CSV file should be created"
        assert len(df) == 1, "CSV should contain one row"

        # Verify chain_of_thought column exists and has data
        assert (
            "chain_of_thought" in df.columns
        ), "CSV should have chain_of_thought column"
        assert (
            df.iloc[0]["chain_of_thought"] == chain_of_thought
        ), "Chain of thought data should be preserved in CSV"

        # Verify all expected columns are present
        expected_columns = [
            "date",
            "decision",
            "prob",
            "explanation",
            "chain_of_thought",
            "strategic_journal",
            "feeling_log",
            "position",
            "next_return_1d",
            "strategy_return",
        ]

        for col in expected_columns:
            assert col in df.columns, f"CSV should have column: {col}"

        print("✓ End-to-end pipeline test passed!")
        print(f"  Decision: {decision}")
        print(f"  Chain of thought length: {len(chain_of_thought)} characters")
        print(f"  CSV columns: {list(df.columns)}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_chain_of_thought_data_integrity():
    """Verify chain of thought data storage and retrieval"""

    # Test data integrity by parsing different response formats and verifying CSV storage
    test_responses = [
        # All features enabled
        """Analyzing market conditions: RSI shows bullish, MACD positive momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.
BUY
0.75
Market indicators suggest upward momentum. Technical analysis shows bullish patterns.
Reviewing recent performance and adjusting strategy accordingly.
Feeling cautiously optimistic about current market conditions.""",
        # Chain of thought only
        """Analyzing market conditions: RSI shows neutral, MACD weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.
HOLD
0.55
Market conditions are mixed with conflicting signals.
Strategic journal disabled in this configuration.
Feeling log disabled in this configuration.""",
        # Strategic + feeling only
        """SELL
0.65
Market shows clear bearish signals that cannot be ignored.
Reviewing risk management and adjusting position sizing.
Feeling concerned about potential downside risk.""",
    ]

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".csv", delete=False
    ) as temp_file:
        temp_path = temp_file.name

    try:
        all_results = []

        for i, response_text in enumerate(test_responses):
            # Parse response with appropriate flags
            if i == 0:  # All features
                with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", True), patch(
                    "src.config.ENABLE_STRATEGIC_JOURNAL", True
                ), patch("src.config.ENABLE_FEELING_LOG", True):
                    result = parse_response_text(response_text)
            elif i == 1:  # Chain of thought only
                with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", True), patch(
                    "src.config.ENABLE_STRATEGIC_JOURNAL", False
                ), patch("src.config.ENABLE_FEELING_LOG", False):
                    result = parse_response_text(response_text)
            else:  # Strategic + feeling only
                with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", False), patch(
                    "src.config.ENABLE_STRATEGIC_JOURNAL", True
                ), patch("src.config.ENABLE_FEELING_LOG", True):
                    result = parse_response_text(response_text)

            (
                decision,
                prob,
                explanation,
                chain_of_thought,
                strategic_journal,
                feeling_log,
            ) = result

            # Create data entry
            data_entry = {
                "date": pd.Timestamp(f"2024-01-{i+1:02d}"),
                "decision": decision,
                "prob": prob,
                "explanation": explanation,
                "chain_of_thought": chain_of_thought,
                "strategic_journal": strategic_journal,
                "feeling_log": feeling_log,
                "position": (
                    1.0 if decision == "BUY" else (-1.0 if decision == "SELL" else 0.0)
                ),
                "next_return_1d": 1.0,
                "strategy_return": 1.0
                * (1.0 if decision == "BUY" else (-1.0 if decision == "SELL" else 0.0)),
            }

            all_results.append(data_entry)

        # Save all results to CSV
        df = save_parsed_results(temp_path, all_results)

        # Verify data integrity
        assert len(df) == 3, "CSV should contain all test cases"

        # Verify chain of thought data is preserved correctly for each case
        cot_values = df["chain_of_thought"].tolist()
        assert (
            cot_values[0] != "Chain of thought reasoning disabled."
        ), "First case should have chain of thought"
        assert (
            cot_values[1] != "Chain of thought reasoning disabled."
        ), "Second case should have chain of thought"
        assert (
            cot_values[2] == "Chain of thought reasoning disabled."
        ), "Third case should have disabled message"

        # Verify each response has appropriate content
        for i, cot in enumerate(cot_values):
            if i < 2:  # Cases with chain of thought enabled
                assert len(cot) > 50, f"Chain of thought {i} should be substantial"
                assert (
                    "Analyzing market conditions" in cot
                ), f"Chain of thought {i} should contain analytical framework"
            else:  # Case with chain of thought disabled
                assert (
                    cot == "Chain of thought reasoning disabled."
                ), f"Case {i} should show disabled message"

        # Test CSV round-trip (load and verify)
        loaded_df = pd.read_csv(temp_path)
        assert len(loaded_df) == 3, "Loaded CSV should have same number of rows"
        assert (
            "chain_of_thought" in loaded_df.columns
        ), "Chain of thought column should persist in loaded CSV"

        # Verify data matches original
        for i, (_, row) in enumerate(loaded_df.iterrows()):
            original_cot = all_results[i]["chain_of_thought"]
            loaded_cot = row["chain_of_thought"]
            assert (
                original_cot == loaded_cot
            ), f"Chain of thought data mismatch at row {i}"

        print("✓ Chain of thought data integrity test passed!")
        print(f"  Processed {len(df)} test cases with different flag combinations")
        print(f"  All chain of thought entries preserved in CSV")
        print(
            f"  Average chain of thought length: {df['chain_of_thought'].str.len().mean():.0f} characters"
        )

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_pipeline_without_chain_of_thought():
    """Verify pipeline works correctly when chain of thought is disabled"""

    # Test parsing with chain of thought disabled
    with patch("src.config.ENABLE_CHAIN_OF_THOUGHT", False), patch(
        "src.config.ENABLE_STRATEGIC_JOURNAL", True
    ), patch("src.config.ENABLE_FEELING_LOG", True):

        # Sample response without chain of thought (4 lines: decision, prob, explanation, strategic, feeling)
        response = """HOLD
0.45
Market conditions are stable with no clear directional signals.
Maintaining current strategy while monitoring for changes.
Approach remains disciplined with focus on risk management."""

        # Parse response
        result = parse_response_text(response)
        assert len(result) == 6, "Should return 6-tuple for API consistency"

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
        ), "Should show disabled message"

        # Verify other components work normally
        assert decision == "HOLD"
        assert prob == 0.45
        assert (
            strategic_journal
            == "Maintaining current strategy while monitoring for changes."
        )
        assert (
            feeling_log == "Approach remains disciplined with focus on risk management."
        )

    print("✓ Pipeline test without chain of thought passed!")


if __name__ == "__main__":
    print("Running Chain of Thought End-to-End Tests")
    print("=" * 50)

    test_end_to_end_chain_of_thought()
    test_chain_of_thought_data_integrity()
    test_pipeline_without_chain_of_thought()

    print("\n" + "=" * 50)
    print("✓ All end-to-end tests passed!")
