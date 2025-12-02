#!/usr/bin/env python3
"""Test script to generate calibration plots for existing results."""

import os
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.reporting import create_calibration_plot


def test_calibration_plots():
    """Generate calibration plots for all existing parsed results."""

    # Go up one directory from tests/ to project root
    base_dir = os.path.dirname(os.path.dirname(__file__))
    parsed_dir = os.path.join(base_dir, "results", "parsed")
    plots_dir = os.path.join(base_dir, "results", "plots")

    # Skip test if results directory doesn't exist (e.g., in CI without running framework)
    if not os.path.exists(parsed_dir):
        print(f"Skipping calibration test - results directory not found: {parsed_dir}")
        print(
            "This test requires running the framework first with 'python -m src.main'"
        )
        return

    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # Find all parsed result files
    parsed_files = [f for f in os.listdir(parsed_dir) if f.endswith("_parsed.csv")]

    if not parsed_files:
        print("No parsed result files found!")
        return

    print(f"Found {len(parsed_files)} parsed result files")
    print("-" * 60)

    for filename in parsed_files:
        model_tag = filename.replace("_parsed.csv", "")
        parsed_path = os.path.join(parsed_dir, filename)
        plot_path = os.path.join(plots_dir, f"{model_tag}_calibration.png")

        print(f"\nProcessing {model_tag}...")

        try:
            # Load the parsed results
            df = pd.read_csv(parsed_path)
            df["date"] = pd.to_datetime(df["date"])

            # Generate calibration plot
            calibration_data = create_calibration_plot(df, model_tag, plot_path)

            print(f"  Total predictions: {len(df)}")
            print(f"  Number of bins with data: {len(calibration_data)}")
            print(f"  Overall win rate: {(df['strategy_return'] > 0).mean():.2%}")
            print(f"  Mean predicted prob: {df['prob'].mean():.2%}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Calibration plots saved to: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    test_calibration_plots()
