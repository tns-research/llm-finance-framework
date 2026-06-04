# src/report/data_collection.py
"""On-disk artefact collection for the report pipeline.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
Reads the analysis/parsed/plots artefacts a model run leaves on disk and assembles
the loose-schema ``data_sources`` dict consumed by the report renderers.
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def collect_data_sources(model_tag: str, analysis_dir, plots_dir) -> Dict:
    """
    Collect all available data sources for the model.
    """
    analysis_path = Path(analysis_dir)
    plots_path = Path(plots_dir)
    sources = {}

    # Statistical validation JSON
    stat_validation_file = analysis_path / f"{model_tag}_statistical_validation.json"
    if stat_validation_file.exists():
        with open(stat_validation_file, "r") as f:
            sources["statistical_validation"] = json.load(f)
    else:
        # Generate statistical validation if it doesn't exist
        print("  Statistical validation data not found - generating...")
        try:
            from ..statistical_validation import (
                comprehensive_statistical_validation,
            )

            # Define parsed_dir inline since it's used later
            temp_parsed_dir = analysis_path.parent / "parsed"
            parsed_file = temp_parsed_dir / f"{model_tag}_parsed.csv"
            if parsed_file.exists():
                parsed_df = pd.read_csv(parsed_file, parse_dates=["date"])
                stat_results = comprehensive_statistical_validation(
                    parsed_df, model_tag
                )
                sources["statistical_validation"] = stat_results

                # Save for future use
                with open(stat_validation_file, "w") as f:
                    json.dump(stat_results, f, indent=2, default=str)
                print("  ✓ Statistical validation generated")
            else:
                print(
                    "  ✗ Parsed data not found - cannot generate statistical validation"
                )
        except Exception as e:
            print(f"  ✗ Failed to generate statistical validation: {e}")

    # Baseline comparison CSV
    baseline_csv = analysis_path / f"{model_tag}_baseline_comparison.csv"
    if baseline_csv.exists():
        sources["baseline_comparison"] = pd.read_csv(baseline_csv)

    # Calibration analysis markdown
    calibration_md = analysis_path / f"{model_tag}_calibration_analysis.md"
    if calibration_md.exists():
        with open(calibration_md, "r") as f:
            sources["calibration_analysis"] = f.read()

    # Pattern analysis markdown
    pattern_md = analysis_path / f"{model_tag}_pattern_analysis.md"
    if pattern_md.exists():
        with open(pattern_md, "r") as f:
            sources["pattern_analysis"] = f.read()

    # Parsed results CSV (for additional analysis)
    parsed_dir = analysis_path.parent / "parsed"  # results/parsed from results/analysis
    parsed_csv = parsed_dir / f"{model_tag}_parsed.csv"
    if parsed_csv.exists():
        sources["parsed_data"] = pd.read_csv(parsed_csv, parse_dates=["date"])

    # Features data (technical indicators)
    features_csv = (
        Path(analysis_dir).parent.parent / "data" / "processed" / "features.csv"
    )
    if features_csv.exists():
        sources["features_data"] = pd.read_csv(features_csv, parse_dates=["date"])

    # Collect plot files - generate paths relative to reports directory
    sources["plots"] = {}
    plot_extensions = [".png", ".jpg", ".jpeg"]

    for ext in plot_extensions:
        for plot_file in plots_path.glob(f"{model_tag}*{ext}"):
            plot_name = plot_file.stem.replace(f"{model_tag}_", "")
            # Path from reports/ to plots/ is ../plots/filename.png
            sources["plots"][plot_name] = f"../plots/{plot_file.name}"

    # LLM Indicator Alignment Analysis
    if "parsed_data" in sources:
        try:
            # Load features data for indicator analysis
            data_dir = Path(analysis_dir).parent.parent / "data" / "processed"
            features_file = data_dir / "features.csv"
            if features_file.exists():
                features_df = pd.read_csv(features_file, parse_dates=["date"])
                from ..decision_analysis import analyze_llm_indicator_alignment

                alignment_results = analyze_llm_indicator_alignment(
                    sources["parsed_data"], features_df
                )
                sources["llm_indicator_alignment"] = alignment_results
                print("  ✓ LLM indicator alignment analysis completed")
            else:
                print("  ⚠ Features data not found - skipping LLM indicator alignment")
        except Exception as e:
            print(f"  ⚠ LLM indicator alignment analysis failed: {e}")

    # Chain of Thought Analysis
    if (
        "parsed_data" in sources
        and "chain_of_thought" in sources["parsed_data"].columns
    ):
        try:
            from ..chain_of_thought_analysis import (
                analyze_chain_of_thought_quality,
                correlate_reasoning_with_performance,
            )

            # Run chain of thought analyses
            sources["chain_of_thought_quality"] = analyze_chain_of_thought_quality(
                sources["parsed_data"]
            )
            sources["chain_of_thought_performance"] = (
                correlate_reasoning_with_performance(sources["parsed_data"])
            )
            print("  ✓ Chain of thought analysis completed")
        except Exception as e:
            print(f"  ⚠ Chain of thought analysis failed: {e}")

    return sources
