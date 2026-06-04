# src/report/pipeline.py
"""Top-level report pipeline orchestration.

Moved out of report_generator.py (pure structural split, no logic change).
Holds the public entry point ``generate_comprehensive_report`` and its helper
``generate_additional_charts``. report_generator.py re-exports both via the
compat facade.

Note on ``__file__`` depth: this module lives one directory deeper than the
original report_generator.py, so the default-base_dir computation walks up one
extra parent (``.parent.parent.parent``) to land on the repo root exactly as
before.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

from ..baselines import calculate_category_performance
from .charts import create_statistical_visualizations
from .data_collection import collect_data_sources
from .document import (
    generate_master_report,
    generate_master_report_html,
)


def generate_comprehensive_report(
    model_tag: str,
    base_dir: str = None,
    include_additional_charts: bool = True,
    output_format: str = "markdown",
) -> str:
    """
    Generate a comprehensive experiment report consolidating all analyses.

    Args:
        model_tag: Model identifier (e.g., 'dummy_model_memory_only')
        base_dir: Base directory (defaults to script location)
        include_additional_charts: Generate missing charts
        output_format: Output format - "markdown" or "html"

    Returns:
        Path to generated report file
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent

    base_path = Path(base_dir)

    # Setup directories
    results_dir = base_path / "results"
    analysis_dir = results_dir / "analysis"
    plots_dir = results_dir / "plots"
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    print(f"Generating comprehensive report for {model_tag}...")

    try:
        # Collect all data sources
        print("  Collecting data sources...")
        data_sources = collect_data_sources(model_tag, analysis_dir, plots_dir)
        print(f"  Found {len(data_sources)} data sources")

        # Debug: Print what data sources we have
        for key, value in data_sources.items():
            if key == "plots":
                print(f"    {key}: {len(value)} plot files")
            elif hasattr(value, "__len__") and not isinstance(value, str):
                print(f"    {key}: {len(value)} items")
            else:
                print(f"    {key}: available")

        # Generate additional charts if requested
        if include_additional_charts:
            print("  Generating additional charts...")
            additional_charts = generate_additional_charts(
                data_sources, model_tag, plots_dir
            )
            data_sources.update(additional_charts)
            print(f"  Generated {len(additional_charts)} additional charts")

        # Generate the master report
        print(f"  Generating master report in {output_format} format...")
        if output_format.lower() == "html":
            report_content = generate_master_report_html(
                model_tag, data_sources, analysis_dir, plots_dir
            )
            report_filename = f"{model_tag}_comprehensive_report.html"
        else:
            report_content = generate_master_report(
                model_tag, data_sources, analysis_dir, plots_dir
            )
            report_filename = f"{model_tag}_comprehensive_report.md"

        print(f"  Report content length: {len(report_content)} characters")

        if len(report_content.strip()) == 0:
            print("  WARNING: Report content is empty!")
            # Add minimal content for debugging
            if output_format.lower() == "html":
                report_content = f"""<!DOCTYPE html>
<html>
<head><title>Debug Report for {model_tag}</title></head>
<body>
<h1>Debug Report for {model_tag}</h1>
<p>Generated at: {datetime.now()}</p>
<p>Data sources found: {list(data_sources.keys())}</p>
<p>This is a debug report - the main generation failed.</p>
</body>
</html>"""
            else:
                report_content = f"""# Debug Report for {model_tag}

Generated at: {datetime.now()}

Data sources found: {list(data_sources.keys())}

This is a debug report - the main generation failed.
"""

        # Save report
        report_path = reports_dir / report_filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✓ Comprehensive report generated: {report_path}")

        return str(report_path)

    except Exception as e:
        print(f"✗ Error generating comprehensive report: {e}")
        import traceback

        traceback.print_exc()

        # Create error report
        if output_format.lower() == "html":
            error_content = f"""<!DOCTYPE html>
<html>
<head><title>Error Report for {model_tag}</title></head>
<body>
<h1>Error Report for {model_tag}</h1>
<p>Generated at: {datetime.now()}</p>

<h2>Error Details</h2>
<p>Error: {str(e)}</p>

<h2>Data Sources Check</h2>
<ul>
<li>results directory exists: {results_dir.exists()}</li>
<li>analysis directory exists: {analysis_dir.exists()}</li>
<li>plots directory exists: {plots_dir.exists()}</li>
</ul>

<h2>Directory Contents</h2>
"""

            if results_dir.exists():
                error_content += "<ul>\n"
                for item in results_dir.rglob("*"):
                    if item.is_file():
                        error_content += f"<li>{item.relative_to(results_dir)}</li>\n"
                error_content += "</ul>\n"

            error_content += f"""
<h2>Full Traceback</h2>
<pre>{traceback.format_exc()}</pre>
</body>
</html>"""
            error_path = reports_dir / f"{model_tag}_error_report.html"
        else:
            error_content = f"""# Error Report for {model_tag}

Generated at: {datetime.now()}

## Error Details
Error: {str(e)}

## Data Sources Check
- results directory exists: {results_dir.exists()}
- analysis directory exists: {analysis_dir.exists()}
- plots directory exists: {plots_dir.exists()}

## Directory Contents
"""

            if results_dir.exists():
                error_content += f"\nResults directory contents:\n"
                for item in results_dir.rglob("*"):
                    if item.is_file():
                        error_content += f"- {item.relative_to(results_dir)}\n"

            error_content += f"\n## Full Traceback\n```\n{traceback.format_exc()}\n```"

            # Save error report
            error_path = reports_dir / f"{model_tag}_error_report.md"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_content)

        print(f"✓ Error report generated: {error_path}")
        return str(error_path)


def generate_additional_charts(
    data_sources: Dict, model_tag: str, plots_dir: Path
) -> Dict:
    """
    Generate charts that are missing from current analysis.
    """
    additional_charts = {}

    if "parsed_data" not in data_sources:
        return additional_charts

    parsed_df = data_sources["parsed_data"]

    # Import chart generation functions from reporting module
    try:
        from ..decision_analysis import analyze_indicator_specific_performance
        from ..reporting import (
            create_category_performance_plot,
            create_indicator_performance_plot,
            create_risk_analysis_chart,
            create_rolling_performance_chart,
            create_technical_indicators_timeline,
        )

        # Extract LLM metrics from statistical validation (needed for category plots)
        llm_metrics = {}
        if "statistical_validation" in data_sources:
            stat_validation = data_sources["statistical_validation"]
            if "dataset_info" in stat_validation:
                dataset_info = stat_validation["dataset_info"]
                llm_metrics = {
                    "total_return": dataset_info.get("total_strategy_return", 0),
                    "sharpe": dataset_info.get("sharpe_ratio", 0),
                    "win_rate": dataset_info.get("win_rate", 0),
                }

        # Rolling performance charts
        rolling_chart_path = plots_dir / f"{model_tag}_rolling_performance.png"
        create_rolling_performance_chart(parsed_df, model_tag, str(rolling_chart_path))
        additional_charts["rolling_performance_plots"] = {
            "rolling_performance": f"../plots/{rolling_chart_path.name}"
        }

        # Risk analysis charts
        risk_chart_path = plots_dir / f"{model_tag}_risk_analysis.png"
        create_risk_analysis_chart(parsed_df, model_tag, str(risk_chart_path))
        additional_charts["risk_analysis_plots"] = {
            "risk_analysis": f"../plots/{risk_chart_path.name}"
        }

        # Technical indicators timeline (if features data available)
        if "features_data" in data_sources:
            features_df = data_sources["features_data"]
            timeline_chart_path = plots_dir / f"{model_tag}_technical_timeline.png"
            create_technical_indicators_timeline(
                features_df, parsed_df, model_tag, str(timeline_chart_path)
            )
            additional_charts["technical_plots"] = {
                "technical_timeline": f"../plots/{timeline_chart_path.name}"
            }

        # Strategy category performance analysis (if baseline data available)
        if "baseline_comparison" in data_sources:
            baseline_df = data_sources["baseline_comparison"]
            category_stats = calculate_category_performance(baseline_df)

            category_plot_path = plots_dir / f"{model_tag}_category_performance.png"
            create_category_performance_plot(
                category_stats, llm_metrics, str(category_plot_path)
            )
            additional_charts["category_performance_plots"] = {
                "category_performance": f"../plots/{category_plot_path.name}"
            }

        # Indicator-specific performance analysis (if features data available)
        if "features_data" in data_sources:
            features_df = data_sources["features_data"]
            indicator_perf = analyze_indicator_specific_performance(
                parsed_df, features_df
            )

            indicator_plot_path = plots_dir / f"{model_tag}_indicator_performance.png"
            create_indicator_performance_plot(indicator_perf, str(indicator_plot_path))
            additional_charts["indicator_performance_plots"] = {
                "indicator_performance": f"../plots/{indicator_plot_path.name}"
            }

    except ImportError as e:
        print(f"Warning: Could not import chart functions: {e}")

    # Statistical visualizations (if validation data available)
    if "statistical_validation" in data_sources:
        additional_charts["statistical_plots"] = create_statistical_visualizations(
            data_sources["statistical_validation"], model_tag, plots_dir
        )

    # Chain of thought visualizations (if chain of thought data available)
    if (
        "parsed_data" in data_sources
        and "chain_of_thought" in data_sources["parsed_data"].columns
    ):
        try:
            from ..chain_of_thought_analysis import (
                create_reasoning_quality_visualizations,
            )

            create_reasoning_quality_visualizations(
                data_sources["parsed_data"], model_tag, plots_dir
            )
            chart_path = plots_dir / f"{model_tag}_chain_of_thought_quality.png"
            additional_charts["chain_of_thought_plots"] = {
                "chain_of_thought_quality": f"../plots/{chart_path.name}"
            }
            print("  ✓ Chain of thought visualizations generated")
        except Exception as e:
            print(f"  ⚠ Chain of thought visualization failed: {e}")

    return additional_charts
