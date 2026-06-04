#!/usr/bin/env python3
"""
Tests for chain of thought reasoning quality analysis functions.
"""

import os
import tempfile

import pandas as pd

from src.chain_of_thought_analysis import (
    analyze_chain_of_thought_quality,
    correlate_reasoning_with_performance,
    create_reasoning_quality_visualizations,
)


def test_analyze_chain_of_thought_quality():
    """Test chain of thought quality analysis function"""
    # Create test data
    test_data = {
        "chain_of_thought": [
            "Analyzing market conditions: RSI shows neutral, MACD weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.",
            "Market regime assessment shows bullish trend. Technical indicators confirm strength. Risk is acceptable. Strategic timing favors entry.",
            "Considering current market volatility and technical signals.",  # Shorter, less complete
            "Chain of thought reasoning disabled.",  # Disabled case
        ],
        "prob": [0.75, 0.85, 0.65, 0.5],
    }

    df = pd.DataFrame(test_data)

    # Run analysis
    result = analyze_chain_of_thought_quality(df)

    # Verify results
    assert "error" not in result
    assert result["total_reasoning_entries"] == 4
    assert result["average_reasoning_length"] > 0
    assert "framework_completeness" in result
    assert "confidence_correlation" in result

    # Check that framework completeness scoring works
    completeness = result["framework_completeness"]
    assert completeness["mean_score"] > 0
    assert completeness["high_quality"] >= 0
    assert completeness["low_quality"] >= 0

    print("✓ Chain of thought quality analysis test passed")


def test_correlate_reasoning_with_performance():
    """Test reasoning-performance correlation analysis"""
    # Create test data
    test_data = {
        "chain_of_thought": [
            "Analyzing market conditions: RSI shows neutral, MACD weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.",
            "Market regime assessment shows bullish trend. Technical indicators confirm strength. Risk is acceptable. Strategic timing favors entry.",
        ],
        "strategy_return": [0.02, -0.01],  # One win, one loss
    }

    df = pd.DataFrame(test_data)

    # Run analysis
    result = correlate_reasoning_with_performance(df)

    # Verify results
    assert "error" not in result
    assert result["sample_size"] == 2
    assert "quality_score_distribution" in result
    assert "return_distribution" in result
    assert "quality_return_correlation" in result
    assert "quality_tercile_analysis" in result

    print("✓ Reasoning-performance correlation test passed")


def test_visualizations():
    """Test visualization generation"""
    # Create test data
    test_data = {
        "chain_of_thought": [
            "Analyzing market conditions: RSI shows neutral, MACD weak momentum. Risk assessment suggests moderate volatility. Strategic review indicates opportunity for careful positioning.",
            "Market regime assessment shows bullish trend. Technical indicators confirm strength. Risk is acceptable. Strategic timing favors entry.",
        ],
        "strategy_return": [0.02, -0.01],
        "decision": ["BUY", "SELL"],
    }

    df = pd.DataFrame(test_data)

    # Create temporary directory for plots
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run visualization
        create_reasoning_quality_visualizations(df, "test_model", temp_dir)

        # Check if plot file was created
        expected_file = os.path.join(
            temp_dir, "test_model_chain_of_thought_quality.png"
        )
        assert os.path.exists(expected_file), "Visualization file should be created"

    print("✓ Visualization test passed")


def test_error_handling():
    """Test error handling for missing data"""
    # Test with no chain_of_thought column
    df_no_cot = pd.DataFrame({"other_column": [1, 2, 3]})

    result = analyze_chain_of_thought_quality(df_no_cot)
    assert "error" in result
    assert "chain_of_thought column not found" in result["error"]

    # Test with empty chain_of_thought data
    df_empty_cot = pd.DataFrame({"chain_of_thought": [None, None, None]})

    result = analyze_chain_of_thought_quality(df_empty_cot)
    assert "error" in result
    assert "No chain of thought data available" in result["error"]

    # Test performance correlation with missing columns
    df_missing_cols = pd.DataFrame({"chain_of_thought": ["test"]})

    result = correlate_reasoning_with_performance(df_missing_cols)
    assert "error" in result
    assert "strategy_return" in result["error"]

    print("✓ Error handling test passed")


if __name__ == "__main__":
    print("Running Chain of Thought Analysis Tests")
    print("=" * 50)

    test_analyze_chain_of_thought_quality()
    test_correlate_reasoning_with_performance()
    test_visualizations()
    test_error_handling()

    print("\n" + "=" * 50)
    print("✓ All chain of thought analysis tests passed!")
