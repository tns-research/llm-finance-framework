# src/report_generator.py
"""
Comprehensive Experiment Report Generator.

Creates a single, showable document consolidating all analysis results:
- Statistical validation
- Baseline comparisons
- Calibration analysis
- Decision patterns
- Risk analysis
- Performance metrics

Generates missing charts and provides executive summary.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import existing analysis modules
try:
    # Try relative imports (when imported as module)
    from .baselines import calculate_baseline_metrics
    from .statistical_validation import print_validation_report
except ImportError:
    # Fall back to absolute imports (when run as script)
    from baselines import calculate_baseline_metrics
    from statistical_validation import print_validation_report


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
        base_dir = Path(__file__).parent.parent

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
            from .statistical_validation import comprehensive_statistical_validation

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

    # Collect plot files - generate paths relative to reports directory
    sources["plots"] = {}
    plot_extensions = [".png", ".jpg", ".jpeg"]

    for ext in plot_extensions:
        for plot_file in plots_path.glob(f"{model_tag}*{ext}"):
            plot_name = plot_file.stem.replace(f"{model_tag}_", "")
            # Path from reports/ to plots/ is ../plots/filename.png
            sources["plots"][plot_name] = f"../plots/{plot_file.name}"

    return sources


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
        from .reporting import (
            create_risk_analysis_chart,
            create_rolling_performance_chart,
        )

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

    except ImportError as e:
        print(f"Warning: Could not import chart functions: {e}")

    # Statistical visualizations (if validation data available)
    if "statistical_validation" in data_sources:
        additional_charts["statistical_plots"] = create_statistical_visualizations(
            data_sources["statistical_validation"], model_tag, plots_dir
        )

    return additional_charts


def create_rolling_performance_charts(
    parsed_df: pd.DataFrame, model_tag: str, plots_dir: Path
) -> Dict[str, str]:
    """
    Create rolling performance charts (Sharpe, returns, drawdowns).
    """
    charts = {}

    # Ensure we have the necessary columns
    required_cols = ["date", "strategy_return", "next_return_1d", "cumulative_return"]
    if not all(col in parsed_df.columns for col in required_cols):
        return charts

    # Sort by date
    df = parsed_df.sort_values("date").copy()

    # Calculate rolling metrics
    window_sizes = [63, 126, 252]  # ~3, 6, 12 months

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Rolling Performance Analysis - {model_tag}", fontsize=16, fontweight="bold"
    )

    # 1. Rolling Sharpe Ratio
    ax1 = axes[0, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_sharpe = (
                df["strategy_return"]
                .rolling(window)
                .apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                )
            )
            ax1.plot(
                df["date"], rolling_sharpe, label=f"{window//21}M Window", alpha=0.8
            )

    ax1.set_title("Rolling Sharpe Ratio", fontweight="bold")
    ax1.set_ylabel("Annualized Sharpe Ratio")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Rolling Returns
    ax2 = axes[0, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_returns = df["strategy_return"].rolling(window).sum()
            ax2.plot(
                df["date"], rolling_returns, label=f"{window//21}M Window", alpha=0.8
            )

    ax2.set_title("Rolling Total Returns", fontweight="bold")
    ax2.set_ylabel("Total Return (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Rolling Maximum Drawdown
    ax3 = axes[1, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_cumulative = df["strategy_return"].rolling(window).sum()
            rolling_max = rolling_cumulative.rolling(window, min_periods=1).max()
            rolling_dd = rolling_cumulative - rolling_max
            ax3.plot(df["date"], rolling_dd, label=f"{window//21}M Window", alpha=0.8)

    ax3.set_title("Rolling Maximum Drawdown", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Rolling Win Rate
    ax4 = axes[1, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_wins = (df["strategy_return"] > 0).rolling(window).sum()
            rolling_win_rate = rolling_wins / window * 100
            ax4.plot(
                df["date"], rolling_win_rate, label=f"{window//21}M Window", alpha=0.8
            )

    ax4.set_title("Rolling Win Rate", fontweight="bold")
    ax4.set_ylabel("Win Rate (%)")
    ax4.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50% Line")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    chart_path = plots_dir / f"{model_tag}_rolling_performance.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    charts["rolling_performance"] = f"../plots/{chart_path.name}"
    print(f"✓ Rolling performance chart saved: {chart_path}")

    return charts


def create_risk_analysis_charts(
    parsed_df: pd.DataFrame, model_tag: str, plots_dir: Path
) -> Dict[str, str]:
    """
    Create risk analysis charts (VaR, CVaR, stress tests).
    """
    charts = {}

    if "strategy_return" not in parsed_df.columns:
        return charts

    returns = parsed_df["strategy_return"].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Risk Analysis - {model_tag}", fontsize=16, fontweight="bold")

    # 1. Returns Distribution
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=50, alpha=0.7, density=True, label="Strategy Returns")
    ax1.axvline(
        np.mean(returns),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(returns):.3f}%",
    )
    ax1.set_title("Returns Distribution", fontweight="bold")
    ax1.set_xlabel("Daily Return (%)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Value at Risk (VaR) Over Time
    ax2 = axes[0, 1]
    rolling_window = 63  # ~3 months

    if len(returns) > rolling_window:
        var_95 = []
        var_99 = []
        dates = []

        for i in range(rolling_window, len(returns)):
            window_returns = returns[i - rolling_window : i]
            var_95.append(np.percentile(window_returns, 5))
            var_99.append(np.percentile(window_returns, 1))
            dates.append(parsed_df.iloc[i]["date"])

        ax2.plot(dates, var_95, label="VaR 95%", color="orange", alpha=0.8)
        ax2.plot(dates, var_99, label="VaR 99%", color="red", alpha=0.8)
        ax2.fill_between(
            dates, var_99, var_95, alpha=0.2, color="red", label="Tail Risk Zone"
        )

    ax2.set_title("Rolling Value at Risk", fontweight="bold")
    ax2.set_ylabel("VaR (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Drawdown Analysis
    ax3 = axes[1, 0]
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max

    ax3.fill_between(range(len(drawdowns)), 0, drawdowns, color="red", alpha=0.3)
    ax3.plot(drawdowns, color="red", linewidth=1)
    ax3.set_title("Drawdown Analysis", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(alpha=0.3)

    # 4. Stress Test Scenarios
    ax4 = axes[1, 1]

    # Different stress scenarios
    scenarios = {
        "Base Case": returns,
        "High Volatility": returns * 2,
        "Crash Scenario": np.where(
            returns < np.percentile(returns, 10), returns * 3, returns
        ),
        "Bull Market": np.where(returns > 0, returns * 1.5, returns * 0.5),
    }

    for scenario_name, scenario_returns in scenarios.items():
        cumulative_scenario = np.cumsum(scenario_returns)
        ax4.plot(cumulative_scenario, label=scenario_name, alpha=0.8)

    ax4.set_title("Stress Test Scenarios", fontweight="bold")
    ax4.set_ylabel("Cumulative Return (%)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    chart_path = plots_dir / f"{model_tag}_risk_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    charts["risk_analysis"] = f"../plots/{chart_path.name}"
    print(f"✓ Risk analysis chart saved: {chart_path}")

    return charts


def create_statistical_visualizations(
    validation_results: Dict, model_tag: str, plots_dir: Path
) -> Dict[str, str]:
    """
    Create visualizations of statistical validation results.
    """
    charts = {}

    # Bootstrap distribution plot
    if "bootstrap_vs_index" in validation_results:
        bootstrap_data = validation_results["bootstrap_vs_index"]

        if "all_results" in bootstrap_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Statistical Validation Visualizations - {model_tag}",
                fontsize=14,
                fontweight="bold",
            )

            # 1. Bootstrap Distribution
            ax1 = axes[0]
            bootstrap_diffs = np.array(bootstrap_data["all_results"])
            ax1.hist(
                bootstrap_diffs,
                bins=50,
                alpha=0.7,
                density=True,
                label="Bootstrap Distribution",
            )
            ax1.axvline(
                bootstrap_data["sharpe_difference"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}',
            )
            ax1.axvline(
                np.mean(bootstrap_diffs),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(bootstrap_diffs):.3f}",
            )
            ax1.set_title("Bootstrap Sharpe Difference Distribution", fontweight="bold")
            ax1.set_xlabel("Sharpe Ratio Difference")
            ax1.set_ylabel("Density")
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. Confidence Interval
            ax2 = axes[1]
            ax2.hist(bootstrap_diffs, bins=30, alpha=0.7, density=True)
            ci_lower, ci_upper = bootstrap_data["ci_95_bootstrap"]
            ax2.axvline(
                ci_lower,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"95% CI Lower: {ci_lower:.3f}",
            )
            ax2.axvline(
                ci_upper,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"95% CI Upper: {ci_upper:.3f}",
            )
            ax2.axvline(
                bootstrap_data["sharpe_difference"],
                color="red",
                linestyle="-",
                linewidth=2,
                label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}',
            )
            ax2.set_title("Confidence Interval Analysis", fontweight="bold")
            ax2.set_xlabel("Sharpe Ratio Difference")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()

            chart_path = plots_dir / f"{model_tag}_statistical_validation.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            charts["statistical_validation"] = f"../plots/{chart_path.name}"
            print(f"✓ Statistical validation visualization saved: {chart_path}")

    return charts


def generate_master_report(
    model_tag: str, data_sources: Dict, analysis_dir: Path, plots_dir: Path
) -> str:
    """
    Generate the master markdown report combining all analyses.
    """
    report_lines = []

    # Header
    report_lines.extend(
        [
            f"# LLM Trading Strategy Experiment Report",
            f"## Model: {model_tag} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]
    )

    # Executive Summary - Start with key takeaways
    report_lines.extend(generate_executive_summary(data_sources, model_tag))

    # Performance Overview - Visual summary of results
    report_lines.extend(generate_performance_overview(data_sources, model_tag))

    # Comprehensive Risk Analysis - Combine risk attribution and risk analysis
    report_lines.extend(generate_comprehensive_risk_analysis(data_sources, model_tag))

    # Market Environment Analysis - How strategy performs in different conditions
    report_lines.extend(generate_market_regime_analysis(data_sources, model_tag))

    # Statistical Rigor - Validation and confidence assessment
    report_lines.extend(
        generate_statistical_validation_section(data_sources, model_tag)
    )

    # Decision Behavior Analysis - LLM decision-making patterns
    report_lines.extend(generate_decision_behavior_analysis(data_sources, model_tag))

    # Practical Implementation - Real-world deployment considerations
    report_lines.extend(generate_practical_considerations(data_sources, model_tag))

    # Key Insights & Strategic Recommendations
    report_lines.extend(generate_insights_recommendations(data_sources, model_tag))

    # Technical Appendix - Data sources and methodology
    report_lines.extend(generate_technical_details(data_sources, model_tag))

    return "\n".join(report_lines)


def generate_master_report_html(
    model_tag: str, data_sources: Dict, analysis_dir: Path, plots_dir: Path
) -> str:
    """
    Generate the master HTML report combining all analyses with beautiful styling.
    """
    html_parts = []

    # HTML Header with CSS
    html_parts.append(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Trading Strategy Report - {model_tag}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}

        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
            padding: 30px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #4f46e5;
        }}

        .section h2 {{
            color: #1e293b;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section h3 {{
            color: #334155;
            font-size: 1.4rem;
            font-weight: 500;
            margin: 25px 0 15px 0;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }}

        .metric-card .label {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 5px;
        }}

        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
        }}

        .metric-card .value.positive {{
            color: #10b981;
        }}

        .metric-card .value.negative {{
            color: #ef4444;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }}

        .chart-caption {{
            font-style: italic;
            color: #64748b;
            font-size: 0.9rem;
        }}

        .data-sources {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .data-source-item {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .data-source-item::before {{
            content: "📊";
            font-size: 1.2rem;
        }}

        .code-block {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 20px 0;
            white-space: pre-wrap;
        }}

        .insights-list {{
            list-style: none;
            padding: 0;
        }}

        .insights-list li {{
            padding: 12px 0;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}

        .insights-list li::before {{
            content: "💡";
            font-size: 1.1rem;
            flex-shrink: 0;
        }}

        .insights-list li:last-child {{
            border-bottom: none;
        }}

        .footer {{
            background: #f1f5f9;
            padding: 30px;
            text-align: center;
            color: #64748b;
            border-top: 1px solid #e2e8f0;
        }}

        .footer p {{
            margin: 5px 0;
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .header {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 2rem;
            }}

            .content {{
                padding: 20px;
            }}

            .section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LLM Trading Strategy Experiment Report</h1>
            <div class="subtitle">Model: {model_tag} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        <div class="content">
"""
    )

    # Executive Summary Section
    html_parts.append(generate_executive_summary_html(data_sources, model_tag))

    # Performance Overview Section
    html_parts.append(generate_performance_overview_html(data_sources, model_tag))

    # Comprehensive Risk Analysis Section
    html_parts.append(
        generate_comprehensive_risk_analysis_html(data_sources, model_tag)
    )

    # Market Environment Analysis Section
    html_parts.append(generate_market_regime_analysis_html(data_sources, model_tag))

    # Statistical Rigor Section
    html_parts.append(
        generate_statistical_validation_section_html(data_sources, model_tag)
    )

    # Decision Behavior Analysis Section
    html_parts.append(generate_decision_behavior_analysis_html(data_sources, model_tag))

    # Practical Implementation Section
    html_parts.append(generate_practical_considerations_html(data_sources, model_tag))

    # Key Insights & Strategic Recommendations Section
    html_parts.append(generate_insights_recommendations_html(data_sources, model_tag))

    # Technical Appendix Section
    html_parts.append(generate_technical_details_html(data_sources, model_tag))

    # Close HTML
    html_parts.append(
        """
        </div>
        <div class="footer">
            <p><strong>LLM Finance Experiment Framework</strong></p>
            <p>This report was automatically generated. For questions about methodology or results, refer to the technical documentation.</p>
            <p>Generated on """
        + datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        + """</p>
        </div>
    </div>
</body>
</html>"""
    )

    return "".join(html_parts)


def generate_executive_summary(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate comprehensive executive summary with key takeaways."""
    lines = [
        "## 📈 Executive Summary",
        "",
        "### Key Performance Metrics",
        "",
    ]

    # Extract and analyze key metrics
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        dataset = sv.get("dataset_info", {})

        total_return = dataset.get("total_strategy_return", 0)
        index_return = dataset.get("total_index_return", 0)
        n_periods = dataset.get("n_periods", 0)

        # Performance assessment
        performance_rating = (
            "Excellent"
            if total_return > index_return + 5
            else (
                "Good"
                if total_return > index_return
                else "Underperforming" if total_return < index_return - 5 else "Neutral"
            )
        )

        lines.extend(
            [
                f"| Metric | Strategy | Index | Difference |",
                "|--------|----------|-------|------------|",
                f"| Total Return | {total_return:.2f}% | {index_return:.2f}% | {total_return - index_return:+.2f}% |",
                f"| Sharpe Ratio | {sv.get('bootstrap_vs_index', {}).get('strategy_sharpe', 'N/A'):.3f} | {sv.get('bootstrap_vs_index', {}).get('benchmark_sharpe', 'N/A'):.3f} | {sv.get('bootstrap_vs_index', {}).get('sharpe_difference', 0):+.3f} |",
                f"| Trading Days | {n_periods} | {n_periods} | - |",
                "",
                f"**Overall Assessment**: {performance_rating} performance vs market index",
                "",
            ]
        )

        # Statistical significance
        if "bootstrap_vs_index" in sv:
            bs = sv["bootstrap_vs_index"]
            sig_status = (
                "✅ Statistically Significant"
                if bs.get("significant_difference_5pct")
                else "❌ Not Statistically Significant"
            )
            effect_size = bs.get("effect_size", 0)
            effect_magnitude = (
                "Large"
                if abs(effect_size) > 0.8
                else (
                    "Medium"
                    if abs(effect_size) > 0.5
                    else "Small" if abs(effect_size) > 0.2 else "Negligible"
                )
            )

            lines.extend(
                [
                    "### Statistical Confidence",
                    "",
                    f"- **Significance vs Index**: {sig_status} (p = {bs.get('p_value_two_sided', 'N/A'):.4f})",
                    f"- **Effect Size**: {effect_size:.3f} ({effect_magnitude})",
                    f"- **Confidence Interval**: [{bs.get('ci_95_bootstrap', [0, 0])[0]:+.3f}, {bs.get('ci_95_bootstrap', [0, 0])[1]:+.3f}] Sharpe ratio difference",
                ]
            )

        # Out-of-sample validation
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if "error" not in oos:
                overfitting_detected = oos.get("overfitting_detection", {}).get(
                    "overall_overfitting_detected", False
                )
                overfitting_status = (
                    "🚨 Overfitting Detected"
                    if overfitting_detected
                    else "✅ No Overfitting Detected"
                )

                lines.extend(
                    [
                        "",
                        "### Validation Results",
                        "",
                        f"- **Out-of-Sample Test**: {overfitting_status}",
                    ]
                )

                if overfitting_detected:
                    decay = oos.get("overfitting_detection", {}).get(
                        "sharpe_decay_pct", 0
                    )
                    lines.append(
                        f"- **Performance Decay**: {decay:.1f}% reduction in Sharpe ratio out-of-sample"
                    )

        # Decision quality assessment
        if "hold_decision_analysis" in sv:
            hold_data = sv["hold_decision_analysis"]
            if "combined_assessment" in hold_data:
                combined = hold_data["combined_assessment"]
                hold_score = combined.get("overall_score", 0)
                hold_rating = (
                    "Excellent"
                    if hold_score > 0.6
                    else "Good" if hold_score > 0.4 else "Poor"
                )

                lines.extend(
                    [
                        "",
                        "### Decision Quality",
                        "",
                        f"- **HOLD Decision Success**: {hold_score:.1%} ({hold_rating})",
                        f"- **Contextual Accuracy**: {hold_data.get('contextual_correctness', {}).get('context_success_rate', 0):.1%}",
                    ]
                )

    # Key takeaways and implications
    lines.extend(
        [
            "",
            "### Key Takeaways & Implications",
            "",
            "**For This LLM Configuration:**",
        ]
    )

    # Dynamic takeaways based on performance
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        bs = sv.get("bootstrap_vs_index", {})

        if bs.get("significant_difference_5pct", False):
            if bs.get("sharpe_difference", 0) > 0:
                lines.append(
                    "- ✅ **Outperforms market index** with statistical significance"
                )
                lines.append("- 🎯 Shows potential for AI-driven alpha generation")
            else:
                lines.append("- ❌ **Underperforms market index** significantly")
                lines.append(
                    "- ⚠️ May require strategy refinement or different LLM approach"
                )
        else:
            lines.append(
                "- ❓ **Performance not significantly different** from market index"
            )
            lines.append(
                "- 🔄 Results may vary with different market conditions or time periods"
            )

        # Overfitting assessment
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if oos.get("overfitting_detection", {}).get("overall_overfitting_detected"):
                lines.append(
                    "- 🚨 **Overfitting risk detected** - strategy may not generalize"
                )
                lines.append(
                    "- 🧪 Requires additional testing across different market regimes"
                )

        # Decision quality insights
        if "hold_decision_analysis" in sv:
            hold_data = sv["hold_decision_analysis"]
            if hold_data.get("combined_assessment", {}).get("overall_score", 0) > 0.5:
                lines.append("- ✅ **Strong decision-making** in HOLD scenarios")
            else:
                lines.append("- ⚠️ **Conservative HOLD usage** - may miss opportunities")

    lines.extend(
        [
            "",
            "**Research Implications:**",
            "- 🤖 Demonstrates LLM capability for financial decision-making",
            "- 📊 Provides baseline for comparing different AI approaches",
            "- 🔬 Highlights importance of rigorous statistical validation",
            "",
            "---",
            "",
        ]
    )

    return lines


def generate_market_regime_analysis(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate market regime analysis section."""
    lines = [
        "## 📊 Market Regime Analysis",
        "",
        "Performance breakdown by market conditions reveals how the strategy adapts to different environments.",
        "",
    ]

    if "parsed_data" not in data_sources:
        lines.extend(
            ["Market regime analysis requires parsed trading data.", "", "---", ""]
        )
        return lines

    parsed_df = data_sources["parsed_data"]

    # Calculate market regimes based on volatility and returns
    if "next_return_1d" in parsed_df.columns and "strategy_return" in parsed_df.columns:
        # Calculate rolling volatility (20-day window)
        market_returns = parsed_df["next_return_1d"]
        rolling_vol = market_returns.rolling(20).std() * np.sqrt(252)  # Annualized

        # Define regimes based on volatility percentiles
        vol_median = rolling_vol.median()
        vol_high = rolling_vol.quantile(0.75)

        # Create regime labels
        regimes = []
        for vol in rolling_vol:
            if pd.isna(vol):
                regimes.append("Unknown")
            elif vol > vol_high:
                regimes.append("High Volatility")
            elif vol > vol_median:
                regimes.append("Moderate Volatility")
            else:
                regimes.append("Low Volatility")

        parsed_df = parsed_df.copy()
        parsed_df["regime"] = regimes

        # Performance by regime
        regime_performance = []
        for regime in ["Low Volatility", "Moderate Volatility", "High Volatility"]:
            regime_data = parsed_df[parsed_df["regime"] == regime]
            if len(regime_data) > 0:
                strategy_return = (
                    regime_data["strategy_return"].mean() * 252
                )  # Annualized
                market_return = regime_data["next_return_1d"].mean() * 252
                win_rate = (regime_data["strategy_return"] > 0).mean() * 100
                days = len(regime_data)

                regime_performance.append(
                    {
                        "regime": regime,
                        "strategy_return": strategy_return,
                        "market_return": market_return,
                        "excess_return": strategy_return - market_return,
                        "win_rate": win_rate,
                        "days": days,
                    }
                )

        if regime_performance:
            lines.extend(
                [
                    "| Market Regime | Strategy Return | Market Return | Excess Return | Win Rate | Days |",
                    "|---------------|-----------------|---------------|---------------|----------|------|",
                ]
            )

            for perf in regime_performance:
                lines.append(
                    f"| {perf['regime']} | {perf['strategy_return']:.2f}% | {perf['market_return']:.2f}% | {perf['excess_return']:+.2f}% | {perf['win_rate']:.1f}% | {perf['days']} |"
                )

            lines.extend(
                [
                    "",
                    "### Key Regime Insights",
                    "",
                ]
            )

            # Analyze regime performance
            best_regime = max(regime_performance, key=lambda x: x["excess_return"])
            worst_regime = min(regime_performance, key=lambda x: x["excess_return"])

            lines.extend(
                [
                    f"- **Best Performance**: {best_regime['regime']} regime ({best_regime['excess_return']:+.2f}% excess return)",
                    f"- **Worst Performance**: {worst_regime['regime']} regime ({worst_regime['excess_return']:+.2f}% excess return)",
                    f"- **Strategy Adaptation**: {'✅ Adapts well to changing conditions' if abs(best_regime['excess_return'] - worst_regime['excess_return']) < 5 else '⚠️ Performance varies significantly by regime'}",
                    "",
                    "### Practical Implications",
                    "",
                    "- **Portfolio Integration**: Consider regime-based allocation adjustments",
                    "- **Risk Management**: Higher volatility periods may require position size reduction",
                    "- **Strategy Optimization**: Focus improvement efforts on worst-performing regimes",
                    "",
                ]
            )

    lines.extend(["---", ""])

    return lines


def generate_risk_attribution(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate risk attribution and decomposition section."""
    lines = [
        "## 🔍 Risk Attribution Analysis",
        "",
        "Understanding the sources of returns and risk helps identify strategy strengths and weaknesses.",
        "",
    ]

    if "parsed_data" not in data_sources:
        lines.extend(
            ["Risk attribution analysis requires parsed trading data.", "", "---", ""]
        )
        return lines

    parsed_df = data_sources["parsed_data"]

    if "strategy_return" in parsed_df.columns and "next_return_1d" in parsed_df.columns:
        strategy_returns = parsed_df["strategy_return"]
        market_returns = parsed_df["next_return_1d"]

        # Basic risk metrics
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        market_vol = market_returns.std() * np.sqrt(252)
        correlation = strategy_returns.corr(market_returns)

        # Calculate beta and alpha (simplified CAPM)
        if market_vol > 0:
            beta = correlation * (strategy_vol / market_vol)
            alpha = (strategy_returns.mean() - beta * market_returns.mean()) * 252

            # Risk decomposition
            systematic_risk = beta**2 * market_vol**2
            idiosyncratic_risk = strategy_vol**2 - systematic_risk

            lines.extend(
                [
                    "### Risk Decomposition",
                    "",
                    "| Risk Component | Annualized Value | % of Total Risk |",
                    "|---------------|------------------|-----------------|",
                    f"| Total Strategy Risk | {strategy_vol:.2f}% | 100.0% |",
                    f"| Systematic Risk (Beta-related) | {np.sqrt(systematic_risk):.2f}% | {systematic_risk/strategy_vol**2*100:.1f}% |",
                    f"| Idiosyncratic Risk | {np.sqrt(idiosyncratic_risk):.2f}% | {idiosyncratic_risk/strategy_vol**2*100:.1f}% |",
                    "",
                    "### Risk-Adjusted Performance",
                    "",
                    f"- **Beta**: {beta:.3f} ({'High systematic risk' if beta > 1.2 else 'Moderate systematic risk' if beta > 0.8 else 'Low systematic risk'})",
                    f"- **Alpha**: {alpha:.2f}% ({'Positive alpha generated' if alpha > 0 else 'Negative alpha'})",
                    f"- **Correlation to Market**: {correlation:.3f} ({'Highly correlated' if abs(correlation) > 0.7 else 'Moderately correlated' if abs(correlation) > 0.3 else 'Low correlation'})",
                    "",
                ]
            )

            # Win/loss analysis
            winning_days = strategy_returns > 0
            losing_days = strategy_returns < 0

            avg_win = strategy_returns[winning_days].mean()
            avg_loss = strategy_returns[losing_days].mean()
            win_rate = winning_days.mean()

            # Profitability assessment
            profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            lines.extend(
                [
                    "### Profitability Analysis",
                    "",
                    "| Metric | Value | Interpretation |",
                    "|--------|-------|----------------|",
                    f"| Win Rate | {win_rate:.1%} | {'Good' if win_rate > 0.55 else 'Fair' if win_rate > 0.45 else 'Poor'} |",
                    f"| Average Win | {avg_win:.3f}% | Daily return when profitable |",
                    f"| Average Loss | {avg_loss:.3f}% | Daily return when unprofitable |",
                    f"| Profit/Loss Ratio | {profit_ratio:.2f} | {'Good' if profit_ratio > 1.5 else 'Fair' if profit_ratio > 1.0 else 'Poor'} |",
                    f"| Expectancy | {expectancy:.3f}% | Expected daily return |",
                    "",
                    "### Risk Management Insights",
                    "",
                    "- **Return Distribution**: Strategy shows "
                    + (
                        "favorable skew"
                        if expectancy > 0 and profit_ratio > 1.2
                        else "mixed characteristics"
                    ),
                    "- **Risk Profile**: "
                    + (
                        "Aggressive"
                        if strategy_vol > market_vol * 1.5
                        else (
                            "Conservative"
                            if strategy_vol < market_vol * 0.7
                            else "Market-like"
                        )
                    ),
                    "- **Diversification**: "
                    + (
                        "Low correlation provides diversification benefits"
                        if abs(correlation) < 0.5
                        else "High correlation suggests limited diversification"
                    ),
                    "",
                ]
            )

    lines.extend(["---", ""])

    return lines


def generate_practical_considerations(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate practical implementation considerations section."""
    lines = [
        "## 🛠️ Practical Implementation Considerations",
        "",
        "Real-world deployment requires addressing transaction costs, liquidity, and operational factors.",
        "",
    ]

    # Transaction costs analysis
    if "parsed_data" in data_sources:
        parsed_df = data_sources["parsed_data"]

        # Analyze trading frequency (assuming position changes indicate trades)
        if "decision" in parsed_df.columns:
            decisions = parsed_df["decision"]

            # Count position changes (simplified trade detection)
            position_changes = 0
            prev_decision = None

            for decision in decisions:
                if prev_decision is not None and decision != prev_decision:
                    position_changes += 1
                prev_decision = decision

            trading_frequency = position_changes / len(parsed_df) * 100
            annual_trades = position_changes * (252 / len(parsed_df))  # Approximate

            lines.extend(
                [
                    "### Transaction Costs Impact",
                    "",
                    f"- **Trading Frequency**: {trading_frequency:.1f}% of days involve position changes",
                    f"- **Estimated Annual Trades**: {annual_trades:.0f} round trips",
                ]
            )

            # Estimate costs (rough assumptions)
            avg_commission = 0.001  # 0.1% per trade
            avg_spread = 0.0005  # 0.05% spread cost
            total_cost_per_trade = avg_commission + avg_spread

            annual_cost_bps = (
                annual_trades * total_cost_per_trade * 10000
            )  # Convert to bps

            lines.extend(
                [
                    f"- **Estimated Trading Costs**: {annual_cost_bps:.0f} basis points annually",
                    f"- **Cost Impact**: {'Significant' if annual_cost_bps > 50 else 'Moderate' if annual_cost_bps > 20 else 'Minimal'} impact on performance",
                    "",
                ]
            )

    # Operational considerations
    lines.extend(
        [
            "### Operational Considerations",
            "",
            "#### Technical Infrastructure",
            "- **API Reliability**: LLM responses must be consistent and available during market hours",
            "- **Response Time**: Decision latency should be under 100ms for real-time trading",
            "- **Fallback Mechanisms**: Alternative decision rules when LLM unavailable",
            "- **Monitoring**: Real-time performance tracking and automated alerts",
            "",
            "#### Risk Management",
            "- **Position Limits**: Maximum exposure per asset/sector",
            "- **Drawdown Controls**: Automatic reduction during losing streaks",
            "- **Liquidity Checks**: Ensure sufficient volume for position sizing",
            "- **Market Impact**: Consider price impact of larger orders",
            "",
            "#### Regulatory & Compliance",
            "- **Audit Trail**: Complete record of decision-making process",
            "- **Explainability**: Ability to explain AI-driven trades to regulators",
            "- **Bias Monitoring**: Regular checks for systematic biases",
            "- **Testing Requirements**: Validation across multiple market scenarios",
            "",
            "### Scaling Considerations",
            "",
            "- **Cost Efficiency**: LLM API costs vs traditional strategy development",
            "- **Performance Consistency**: Stability across different market conditions",
            "- **Portfolio Size**: Impact of strategy capacity and market impact",
            "- **Multi-Asset Extension**: Applicability beyond single-asset strategies",
            "",
        ]
    )

    # Performance expectations
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        bs = sv.get("bootstrap_vs_index", {})

        if bs.get("significant_difference_5pct", False):
            if bs.get("sharpe_difference", 0) > 0:
                lines.extend(
                    [
                        "### Deployment Recommendations",
                        "",
                        "✅ **Recommended for live deployment** with proper risk controls",
                        "- Implement position sizing based on confidence scores",
                        "- Monitor for overfitting in live performance",
                        "- Consider hybrid approach combining AI with traditional rules",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "### Deployment Recommendations",
                        "",
                        "⚠️ **Not recommended for live deployment** in current form",
                        "- Requires significant strategy refinement",
                        "- Consider as research baseline rather than production strategy",
                        "- May be suitable for specialized market conditions",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "### Deployment Recommendations",
                    "",
                    "🔄 **Further testing required** before deployment decision",
                    "- Results not statistically significant from market index",
                    "- Additional validation across different time periods needed",
                    "- Consider as experimental approach rather than primary strategy",
                    "",
                ]
            )

    lines.extend(["---", ""])

    return lines


def generate_comprehensive_risk_analysis(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate comprehensive risk analysis combining attribution and risk metrics."""
    lines = [
        "## 📊 Comprehensive Risk Analysis",
        "",
        "Complete assessment of strategy risk profile, including attribution, VaR, and stress testing.",
        "",
    ]

    # Risk Attribution Analysis
    if "parsed_data" in data_sources:
        parsed_df = data_sources["parsed_data"]

        try:
            from .statistical_validation import calculate_risk_attribution

            risk_metrics = calculate_risk_attribution(
                parsed_df["strategy_return"].values, parsed_df["next_return_1d"].values
            )

            lines.extend(
                [
                    "### Risk Attribution & Decomposition",
                    "",
                    "| Risk Component | Value | Interpretation |",
                    "|---------------|-------|----------------|",
                    f"| Beta (Market Sensitivity) | {risk_metrics['beta']:.3f} | {'High' if abs(risk_metrics['beta']) > 1.2 else 'Moderate' if abs(risk_metrics['beta']) > 0.8 else 'Low'} systematic risk |",
                    f"| Alpha (Excess Return) | {risk_metrics['alpha']:.2f}% | {'Positive' if risk_metrics['alpha'] > 0 else 'Negative'} risk-adjusted performance |",
                    f"| Correlation to Market | {risk_metrics['correlation']:.3f} | {'Highly' if abs(risk_metrics['correlation']) > 0.7 else 'Moderately' if abs(risk_metrics['correlation']) > 0.3 else 'Low'} correlated |",
                    f"| Total Volatility | {risk_metrics['total_risk']:.2f}% | Annualized strategy volatility |",
                    "",
                    f"**Risk Decomposition**: {risk_metrics['systematic_risk_pct']:.1f}% systematic risk, {risk_metrics['idiosyncratic_risk_pct']:.1f}% idiosyncratic risk",
                    "",
                ]
            )

        except ImportError as e:
            lines.extend(
                [
                    "Risk attribution analysis not available.",
                    "",
                ]
            )

    # Include Risk Analysis Chart
    if (
        "risk_analysis_plots" in data_sources
        and "risk_analysis" in data_sources["risk_analysis_plots"]
    ):
        lines.extend(
            [
                "### Risk Metrics Visualization",
                "",
                f"![Risk Analysis]({data_sources['risk_analysis_plots']['risk_analysis']})",
                "*Figure: Comprehensive risk analysis including VaR, drawdowns, and stress tests*",
                "",
            ]
        )

    # Rolling Performance Analysis
    if (
        "rolling_performance_plots" in data_sources
        and "rolling_performance" in data_sources["rolling_performance_plots"]
    ):
        lines.extend(
            [
                "### Rolling Performance Analysis",
                "",
                f"![Rolling Performance]({data_sources['rolling_performance_plots']['rolling_performance']})",
                "*Figure: Rolling Sharpe ratio, returns, drawdowns, and win rates over time*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_decision_behavior_analysis(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate comprehensive decision behavior analysis combining calibration and HOLD analysis."""
    lines = [
        "## 🎯 Decision Behavior Analysis",
        "",
        "Analysis of LLM decision-making patterns, calibration quality, and behavioral biases.",
        "",
    ]

    # Calibration Analysis
    if "plots" in data_sources:
        if "calibration" in data_sources["plots"]:
            lines.extend(
                [
                    "### Prediction Calibration",
                    "",
                    f"![Calibration Plot]({data_sources['plots']['calibration']})",
                    "*Figure: How well predicted confidence matches actual performance*",
                    "",
                ]
            )

        if "calibration_by_decision" in data_sources["plots"]:
            lines.extend(
                [
                    f"![Calibration by Decision]({data_sources['plots']['calibration_by_decision']})",
                    "*Figure: Calibration analysis by decision type (BUY/HOLD/SELL)*",
                    "",
                ]
            )

    # Include calibration analysis text
    if "calibration_analysis" in data_sources:
        cal_text = data_sources["calibration_analysis"]
        # Extract key insights from calibration analysis
        lines.extend(
            [
                "### Calibration Insights",
                "",
            ]
        )

        # Look for key metrics in the calibration text
        if "Overall Win Rate:" in cal_text:
            lines.append(
                "**Overall Performance**: "
                + cal_text.split("Overall Win Rate:")[1].split("\n")[0].strip()
            )

        if "Mean Predicted Probability:" in cal_text:
            lines.append(
                "**Average Confidence**: "
                + cal_text.split("Mean Predicted Probability:")[1]
                .split("\n")[0]
                .strip()
            )

        lines.append("")

    # RSI Technical Analysis
    lines.extend(generate_rsi_analysis_section(data_sources, model_tag))

    # Decision Patterns
    if "plots" in data_sources and "decision_patterns" in data_sources["plots"]:
        lines.extend(
            [
                "### Decision Pattern Analysis",
                "",
                f"![Decision Patterns]({data_sources['plots']['decision_patterns']})",
                "*Figure: Decision changes after wins vs losses - evidence of learning/adaptation*",
                "",
            ]
        )

    # Comprehensive Decision Analysis
    if (
        "statistical_validation" in data_sources
        and "decision_effectiveness" in data_sources["statistical_validation"]
    ):
        decision_data = data_sources["statistical_validation"]["decision_effectiveness"]

        # Decision Distribution
        if decision_data.get("decision_distribution"):
            lines.extend(
                [
                    "### Decision Distribution",
                    "",
                    "| Decision | Count | Percentage |",
                    "|----------|-------|------------|",
                ]
            )

            for decision, stats in decision_data["decision_distribution"].items():
                lines.append(
                    f"| {decision} | {stats['count']} | {stats['percentage']:.1f}% |"
                )

            lines.append("")

        # Overall Effectiveness
        if decision_data.get("overall_effectiveness"):
            overall = decision_data["overall_effectiveness"]
            lines.extend(
                [
                    "### Overall Decision Effectiveness",
                    "",
                    f"**Total Decisions**: {overall['total_decisions']}",
                    f"**Overall Win Rate**: {overall['overall_win_rate']:.1f}%",
                    f"**Average Daily Return**: {overall['overall_avg_return']:.3f}%",
                    f"**Total Return**: {overall['overall_total_return']:.2f}%",
                    f"**Annualized Volatility**: {overall['overall_volatility']:.2f}%",
                    f"**Sharpe Ratio**: {overall['overall_sharpe']:.3f}",
                    f"**Maximum Drawdown**: {overall['overall_max_drawdown']:.2f}%",
                    "",
                ]
            )

        # Individual Decision Performance
        if decision_data.get("decision_performance"):
            lines.extend(
                [
                    "### Performance by Decision Type",
                    "",
                    "| Decision | Win Rate | Avg Return | Excess Return | Sharpe | Volatility | Frequency |",
                    "|----------|----------|------------|---------------|--------|------------|-----------|",
                ]
            )

            for decision, perf in decision_data["decision_performance"].items():
                lines.append(
                    f"| {decision} | {perf['win_rate']:.1f}% | {perf['avg_daily_return']:.3f}% | "
                    f"{perf['excess_return_annualized']:+.1f}% | {perf['sharpe_ratio']:.2f} | "
                    f"{perf['volatility_annualized']:.1f}% | {perf['decision_frequency_pct']:.1f}% |"
                )

            lines.append("")

        # Risk-Adjusted Analysis Summary
        if decision_data.get("risk_adjusted_analysis"):
            risk_adj = decision_data["risk_adjusted_analysis"]
            lines.extend(
                [
                    "### Decision Strategy Insights",
                    "",
                    f"- **Best Performing Decision**: {risk_adj['best_decision']} "
                    f"({risk_adj['best_excess_return']:+.1f}% annualized excess return)",
                    f"- **Worst Performing Decision**: {risk_adj['worst_decision']} "
                    f"({risk_adj['worst_excess_return']:+.1f}% annualized excess return)",
                    f"- **Decision Consistency**: {risk_adj['decision_consistency'].title()} performance across decision types",
                    "",
                ]
            )

    # Legacy HOLD Analysis (for detailed context)
    if (
        "statistical_validation" in data_sources
        and "hold_decision_analysis" in data_sources["statistical_validation"]
    ):
        hold_data = data_sources["statistical_validation"]["hold_decision_analysis"]

        if "combined_assessment" in hold_data and "note" not in hold_data:
            combined = hold_data["combined_assessment"]
            hold_score = combined.get("overall_score", 0)
            hold_rating = (
                "Excellent"
                if hold_score > 0.6
                else "Good" if hold_score > 0.4 else "Poor"
            )

            lines.extend(
                [
                    "### Detailed HOLD Analysis",
                    "",
                    f"**HOLD Success Rate**: {hold_score:.1%} ({hold_rating})",
                    "",
                    "#### Quiet Market Performance",
                    f"- **Success Rate**: {hold_data.get('quiet_market_success', {}).get('success_rate', 0):.1%}",
                    f"- **Assessment**: {hold_data.get('quiet_market_success', {}).get('interpretation', 'N/A')[:60]}...",
                    "",
                    "#### Enhanced HOLD Analysis",
                    f"- **Relative Performance**: {hold_data.get('relative_performance', {}).get('avg_score', 0):.1%}",
                    f"- **Risk Avoidance**: {hold_data.get('risk_avoidance', {}).get('avoidance_rate', 0):.1%}",
                    "",
                ]
            )

    lines.extend(["---", ""])

    return lines


def generate_performance_overview(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate performance overview section."""
    lines = [
        "## 🎯 Performance Overview",
        "",
    ]

    # Include baseline comparison plot if available
    if "plots" in data_sources and "baseline_comparison" in data_sources["plots"]:
        lines.extend(
            [
                f"![Baseline Comparison]({data_sources['plots']['baseline_comparison']})",
                "*Figure 1: Strategy performance vs baseline strategies*",
                "",
            ]
        )

    # Include equity curves if available
    if "plots" in data_sources and "equity_curves" in data_sources["plots"]:
        lines.extend(
            [
                f"![Equity Curves]({data_sources['plots']['equity_curves']})",
                "*Figure 2: Equity curves over time*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_statistical_validation_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate statistical validation section."""
    lines = [
        "## 📊 Statistical Validation",
        "",
    ]

    if "statistical_validation" in data_sources:
        # Use the existing print function but capture output
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            print_validation_report(data_sources["statistical_validation"], model_tag)
            validation_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        # Convert the console output to markdown
        lines.extend(validation_text.split("\n"))
        lines.append("")

    # Include statistical visualization if available
    if (
        "statistical_plots" in data_sources
        and "statistical_validation" in data_sources["statistical_plots"]
    ):
        lines.extend(
            [
                f"![Statistical Validation Visualization]({data_sources['statistical_plots']['statistical_validation']})",
                "*Figure 3: Bootstrap distribution and confidence intervals*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_decision_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate decision analysis section."""
    lines = [
        "## 🎪 Decision Analysis",
        "",
    ]

    # Calibration plots
    if "plots" in data_sources:
        if "calibration" in data_sources["plots"]:
            lines.extend(
                [
                    f"![Calibration Plot]({data_sources['plots']['calibration']})",
                    "*Figure 4: Prediction confidence vs actual performance*",
                    "",
                ]
            )

        if "calibration_by_decision" in data_sources["plots"]:
            lines.extend(
                [
                    f"![Calibration by Decision]({data_sources['plots']['calibration_by_decision']})",
                    "*Figure 5: Calibration analysis by decision type (BUY/HOLD/SELL)*",
                    "",
                ]
            )

    # Include calibration analysis text
    if "calibration_analysis" in data_sources:
        lines.extend(["### Detailed Calibration Analysis", "", "```markdown"])
        # Extract key sections from the calibration analysis
        cal_text = data_sources["calibration_analysis"]
        lines.extend(cal_text.split("\n"))
        lines.extend(
            [
                "```",
                "",
            ]
        )

    # Decision patterns
    if "plots" in data_sources and "decision_patterns" in data_sources["plots"]:
        lines.extend(
            [
                f"![Decision Patterns]({data_sources['plots']['decision_patterns']})",
                "*Figure 6: Decision patterns after wins vs losses*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_risk_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate risk analysis section."""
    lines = [
        "## 📉 Risk Analysis",
        "",
    ]

    # Include risk analysis chart if available
    if (
        "risk_analysis_plots" in data_sources
        and "risk_analysis" in data_sources["risk_analysis_plots"]
    ):
        lines.extend(
            [
                f"![Risk Analysis]({data_sources['risk_analysis_plots']['risk_analysis']})",
                "*Figure 7: Comprehensive risk analysis including VaR, drawdowns, and stress tests*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_hold_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate HOLD decision analysis section."""
    lines = [
        "## 🛡️ HOLD Decision Analysis",
        "",
    ]

    if (
        "statistical_validation" in data_sources
        and "hold_decision_analysis" in data_sources["statistical_validation"]
    ):
        hold_data = data_sources["statistical_validation"]["hold_decision_analysis"]

        if "note" in hold_data:
            lines.extend(
                [
                    f"{hold_data['note']}",
                    "",
                ]
            )
        else:
            # Overall assessment
            overall_rate = hold_data["overall_hold_success_rate"]
            category = hold_data["combined_assessment"]["performance_category"]

            lines.extend(
                [
                    f"**Overall HOLD Success Rate:** {overall_rate:.1%} ({category.upper()})",
                    "",
                    "### Quiet Market Success (<0.2% Daily Moves)",
                    "",
                ]
            )

            quiet = hold_data["quiet_market_success"]
            lines.extend(
                [
                    f"- **Success Rate:** {quiet['success_rate']:.1%}",
                    f"- **Successful HOLDs:** {quiet['successful_holds']}/{quiet['total_holds']}",
                    f"- **{quiet['interpretation']}**",
                    "",
                ]
            )

            # Contextual correctness
            lines.extend(
                [
                    "### Contextual Decision Correctness",
                    "",
                ]
            )

            context = hold_data["contextual_correctness"]
            lines.extend(
                [
                    f"- **Average Context Score:** {context['avg_context_score']:.2f}",
                    f"- **Context Success Rate:** {context['context_success_rate']:.1%}",
                    f"- **{context['interpretation']}**",
                    "",
                ]
            )

            # Context reasons breakdown
            if context["reason_breakdown"]:
                lines.extend(
                    [
                        "**Top Context Reasons:**",
                        "",
                    ]
                )
                for reason, count in sorted(
                    context["reason_breakdown"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    reason_name = reason.replace("_", " ").title()
                    lines.extend(
                        [
                            f"- {reason_name}: {count} decisions",
                        ]
                    )
                lines.append("")

            # Combined assessment
            combined = hold_data["combined_assessment"]
            lines.extend(
                [
                    "### Combined Assessment",
                    "",
                    f"- **Quiet Market Weight:** {combined['quiet_weight']:.0%}",
                    f"- **Context Weight:** {combined['context_weight']:.0%}",
                    f"- **Overall Score:** {combined['overall_score']:.1%}",
                    "",
                ]
            )

            # HOLD statistics
            stats = hold_data["hold_statistics"]
            lines.extend(
                [
                    "### HOLD Usage Statistics",
                    "",
                    f"- **Total HOLD Decisions:** {stats['total_hold_decisions']}",
                    f"- **HOLD Usage Rate:** {stats['hold_percentage']:.1%}",
                    f"- **Avg Market Move During HOLD:** {stats['avg_market_move_during_hold']:.2%}",
                    f"- **HOLD During Quiet Markets:** {stats['hold_during_quiet_pct']:.1%}",
                    "",
                ]
            )

            # Summary insight
            if "summary" in hold_data:
                lines.extend(
                    [
                        f"**Key Insight:** {hold_data['summary']['key_insight']}",
                        "",
                    ]
                )

    else:
        lines.extend(
            [
                "HOLD decision analysis not available. Run statistical validation first.",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_additional_charts_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate additional charts section."""
    lines = [
        "## 📈 Additional Performance Analysis",
        "",
    ]

    # Rolling performance charts
    if (
        "rolling_performance_plots" in data_sources
        and "rolling_performance" in data_sources["rolling_performance_plots"]
    ):
        lines.extend(
            [
                f"![Rolling Performance]({data_sources['rolling_performance_plots']['rolling_performance']})",
                "*Figure 8: Rolling performance metrics over different time windows*",
                "",
            ]
        )

    lines.extend(["---", ""])

    return lines


def generate_insights_recommendations(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate insights and recommendations section."""
    lines = [
        "## 💡 Key Insights & Recommendations",
        "",
    ]

    # Extract insights from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]

        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]

            lines.append("### Key Findings")
            for finding in sa.get("key_findings", []):
                lines.append(f"- {finding}")

            lines.append("")
            lines.append("### Recommendations")
            for rec in sa.get("recommendations", []):
                lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "### Overall Assessment",
        ]
    )

    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]
            lines.extend(
                [
                    f"- **Assessment**: {sa.get('overall_assessment', 'Unknown').upper()}",
                    f"- **Confidence Level**: {sa.get('confidence_level', 'Unknown').upper()}",
                ]
            )

    lines.extend(["", "---", ""])

    return lines


def generate_technical_details(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate technical details section."""
    lines = [
        "## 📁 Technical Details",
        "",
        f"- **Model Tag**: {model_tag}",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### Data Sources",
    ]

    # List all data sources used
    for source_name, source_data in data_sources.items():
        if source_name == "plots":
            lines.append(f"- **Plots**: {len(source_data)} chart files")
        elif source_name == "statistical_validation":
            lines.append(
                "- **Statistical Validation**: Bootstrap analysis and out-of-sample testing"
            )
        elif source_name == "baseline_comparison":
            lines.append(
                "- **Baseline Comparison**: Performance vs multiple strategies"
            )
        elif source_name == "parsed_data":
            n_periods = len(source_data) if hasattr(source_data, "__len__") else "N/A"
            lines.append(f"- **Parsed Data**: {n_periods} trading periods")
        else:
            lines.append(f"- **{source_name.replace('_', ' ').title()}**: Available")

    lines.extend(
        [
            "",
            "### Analysis Components",
            "- Statistical significance testing (bootstrap)",
            "- Out-of-sample validation",
            "- Risk-adjusted performance metrics",
            "- Decision calibration analysis",
            "- Baseline strategy comparisons",
            "- Rolling performance analysis",
            "",
            "---",
            "",
            "*This report was automatically generated by the LLM Finance Experiment framework.*",
            "*For questions about methodology or results, refer to the technical documentation.*",
        ]
    )

    return lines


def generate_rsi_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate RSI analysis section for markdown reports."""
    # Check if any technical indicator plots are available
    has_technical_plots = ("plots" in data_sources and
                          ("technical_indicators" in data_sources["plots"] or
                           "rsi_performance" in data_sources["plots"]))

    if not has_technical_plots:
        return [
            "### Technical Indicators Analysis",
            "",
            "*Technical indicators were disabled for this experiment.*",
            "",
        ]

    section = [
        "### Technical Indicators Analysis",
        "",
        "This section analyzes how the model utilized technical indicators.",
        "",
    ]

    # Technical indicators overview plot
    if "plots" in data_sources and "technical_indicators" in data_sources["plots"]:
        section.extend([
            "#### Technical Indicator Overview",
            "",
            f"![Technical Indicators]({data_sources['plots']['technical_indicators']})",
            "*Figure: Price action with available technical indicators and trading signals*",
            "",
        ])

    # RSI performance analysis plot (only if RSI-specific analysis was generated)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        section.extend([
            "#### RSI Performance Analysis",
            "",
            f"![RSI Analysis]({data_sources['plots']['rsi_performance']})",
            "*Figure: RSI distribution by decision type and performance correlation*",
            "",
        ])

    # Key insights (conditional based on what analysis was performed)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        section.extend([
        "#### Key RSI Insights",
        "",
        "- **Decision Distribution**: How BUY/HOLD/SELL decisions correlate with RSI levels",
        "- **Performance by RSI Range**: Win rates across different RSI ranges (0-30, 30-70, 70-100)",
        "- **Winning vs Losing Trades**: RSI distribution comparison between profitable and unprofitable trades",
        "- **RSI Momentum**: Performance based on RSI directional changes and momentum",
        "",
        "**RSI Strategy Effectiveness**: RSI-based strategies provide momentum signals that complement trend and volatility indicators.",
        "",
    ])
    elif "plots" in data_sources and "technical_indicators" in data_sources["plots"]:
        section.extend([
            "#### Technical Indicator Analysis",
            "",
            "Technical indicators were included in this experiment. The overview plot above shows available indicators alongside price action and trading decisions.",
            "",
        ])

    return section


def generate_rsi_analysis_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate RSI analysis section for HTML reports."""
    # Check if any technical indicator plots are available
    has_technical_plots = ("plots" in data_sources and
                          ("technical_indicators" in data_sources["plots"] or
                           "rsi_performance" in data_sources["plots"]))

    if not has_technical_plots:
        return """
                <h3>Technical Indicators Analysis</h3>
                <p><em>Technical indicators were disabled for this experiment.</em></p>
"""

    html = """
                <h3>Technical Indicators Analysis</h3>
                <p>This section analyzes how the model utilized technical indicators.</p>
"""

    # Technical indicators overview plot
    if "plots" in data_sources and "technical_indicators" in data_sources["plots"]:
        html += f"""
                <h4>Technical Indicator Overview</h4>
                <div class="chart-container">
                    <img src="{data_sources['plots']['technical_indicators']}" alt="Technical Indicators">
                    <div class="chart-caption">Figure: Price action with available technical indicators and trading signals</div>
                </div>
"""

    # RSI performance analysis plot (only if RSI-specific analysis was generated)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        html += f"""
                <h4>RSI Performance Analysis</h4>
                <div class="chart-container">
                    <img src="{data_sources['plots']['rsi_performance']}" alt="RSI Analysis">
                    <div class="chart-caption">Figure: RSI distribution by decision type and performance correlation</div>
                </div>
"""

    # Key insights (only if RSI-specific analysis was generated)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        html += """
                <h4>Key RSI Insights</h4>
                <ul>
                    <li><strong>Decision Distribution</strong>: How BUY/HOLD/SELL decisions correlate with RSI levels</li>
                    <li><strong>Performance by RSI Range</strong>: Win rates across different RSI ranges (0-30, 30-70, 70-100)</li>
                    <li><strong>Winning vs Losing Trades</strong>: RSI distribution comparison between profitable and unprofitable trades</li>
                    <li><strong>RSI Momentum</strong>: Performance based on RSI directional changes and momentum</li>
                </ul>
                <p><strong>RSI Strategy Effectiveness</strong>: RSI-based strategies provide momentum signals that complement trend and volatility indicators.</p>
"""
    elif "plots" in data_sources and "technical_indicators" in data_sources["plots"]:
        html += """
                <h4>Technical Indicator Analysis</h4>
                <p>Technical indicators were included in this experiment. The overview plot above shows available indicators alongside price action and trading decisions.</p>
"""

    return html


# HTML Section Generation Functions


def generate_executive_summary_html(data_sources: Dict, model_tag: str) -> str:
    """Generate executive summary section in HTML."""
    html = """            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="metric-grid">
"""

    # Extract key metrics from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        dataset = sv.get("dataset_info", {})

        total_return = dataset.get("total_strategy_return", "N/A")
        index_return = dataset.get("total_index_return", "N/A")
        n_periods = dataset.get("n_periods", "N/A")

        html += f"""
                    <div class="metric-card">
                        <div class="label">Total Return</div>
                        <div class="value{' positive' if isinstance(total_return, (int, float)) and total_return > 0 else ''}">{total_return:.2f}%</div>
                        <small>vs Index: {index_return:.2f}%</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Trading Period</div>
                        <div class="value">{n_periods} days</div>
                    </div>
"""

        # Bootstrap results
        if "bootstrap_vs_index" in sv:
            bs = sv["bootstrap_vs_index"]
            sig_status = (
                "✓ Significant"
                if bs.get("significant_difference_5pct")
                else "❌ Not significant"
            )
            p_value = bs.get("p_value_two_sided", "N/A")
            effect_size = bs.get("effect_size", "N/A")

            html += f"""
                    <div class="metric-card">
                        <div class="label">Statistical Significance</div>
                        <div class="value">{sig_status}</div>
                        <small>p-value: {p_value:.4f}</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Effect Size</div>
                        <div class="value">{effect_size:.3f}</div>
                        <small>Cohen's d</small>
                    </div>
"""

        # Out-of-sample validation
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if "error" not in oos:
                overfitting = (
                    "🚨 Detected"
                    if oos.get("overfitting_detection", {}).get(
                        "overall_overfitting_detected"
                    )
                    else "✅ None detected"
                )
                html += f"""
                    <div class="metric-card">
                        <div class="label">Overfitting</div>
                        <div class="value">{overfitting}</div>
                    </div>
"""

    html += """
                </div>
            </div>
"""

    return html


def generate_comprehensive_risk_analysis_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate comprehensive risk analysis section in HTML."""
    html = """            <div class="section">
                <h2>📊 Comprehensive Risk Analysis</h2>
                <p>Complete assessment of strategy risk profile, including attribution, VaR, and stress testing.</p>
"""

    # Risk Attribution Analysis
    if "parsed_data" in data_sources:
        parsed_df = data_sources["parsed_data"]

        try:
            from .statistical_validation import calculate_risk_attribution

            risk_metrics = calculate_risk_attribution(
                parsed_df["strategy_return"].values, parsed_df["next_return_1d"].values
            )

            html += f"""
                <h3>Risk Attribution & Decomposition</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Beta (Market Sensitivity)</div>
                        <div class="value">{risk_metrics['beta']:.3f}</div>
                        <small>{'High' if abs(risk_metrics['beta']) > 1.2 else 'Moderate' if abs(risk_metrics['beta']) > 0.8 else 'Low'} systematic risk</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Alpha (Excess Return)</div>
                        <div class="value{' positive' if risk_metrics['alpha'] > 0 else ' negative'}">{risk_metrics['alpha']:.2f}%</div>
                        <small>Risk-adjusted performance</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Market Correlation</div>
                        <div class="value">{risk_metrics['correlation']:.3f}</div>
                        <small>{'Highly' if abs(risk_metrics['correlation']) > 0.7 else 'Moderately' if abs(risk_metrics['correlation']) > 0.3 else 'Low'} correlated</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Total Volatility</div>
                        <div class="value">{risk_metrics['total_risk']:.2f}%</div>
                        <small>Annualized strategy volatility</small>
                    </div>
                </div>

                <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <strong>Risk Decomposition:</strong> {risk_metrics['systematic_risk_pct']:.1f}% systematic risk, {risk_metrics['idiosyncratic_risk_pct']:.1f}% idiosyncratic risk
                </div>
"""

        except ImportError as e:
            html += """
                <p>Risk attribution analysis not available.</p>
"""

    # Include Risk Analysis Chart
    if (
        "risk_analysis_plots" in data_sources
        and "risk_analysis" in data_sources["risk_analysis_plots"]
    ):
        html += f"""
                <h3>Risk Metrics Visualization</h3>
                <div class="chart-container">
                    <img src="{data_sources['risk_analysis_plots']['risk_analysis']}" alt="Risk Analysis">
                    <div class="chart-caption">Figure: Comprehensive risk analysis including VaR, drawdowns, and stress tests</div>
                </div>
"""

    # Rolling Performance Analysis
    if (
        "rolling_performance_plots" in data_sources
        and "rolling_performance" in data_sources["rolling_performance_plots"]
    ):
        html += f"""
                <h3>Rolling Performance Analysis</h3>
                <div class="chart-container">
                    <img src="{data_sources['rolling_performance_plots']['rolling_performance']}" alt="Rolling Performance">
                    <div class="chart-caption">Figure: Rolling Sharpe ratio, returns, drawdowns, and win rates over time</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_decision_behavior_analysis_html(data_sources: Dict, model_tag: str) -> str:
    """Generate comprehensive decision behavior analysis section in HTML."""
    html = """            <div class="section">
                <h2>🎯 Decision Behavior Analysis</h2>
                <p>Analysis of LLM decision-making patterns, calibration quality, and behavioral biases.</p>
"""

    # Calibration Analysis
    if "plots" in data_sources:
        if "calibration" in data_sources["plots"]:
            html += f"""
                <h3>Prediction Calibration</h3>
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration']}" alt="Calibration Plot">
                    <div class="chart-caption">Figure: How well predicted confidence matches actual performance</div>
                </div>
"""

        if "calibration_by_decision" in data_sources["plots"]:
            html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration_by_decision']}" alt="Calibration by Decision">
                    <div class="chart-caption">Figure: Calibration analysis by decision type (BUY/HOLD/SELL)</div>
                </div>
"""

    # RSI Technical Analysis
    html += generate_rsi_analysis_section_html(data_sources, model_tag)

    # Decision Patterns
    if "plots" in data_sources and "decision_patterns" in data_sources["plots"]:
        html += f"""
                <h3>Decision Pattern Analysis</h3>
                <div class="chart-container">
                    <img src="{data_sources['plots']['decision_patterns']}" alt="Decision Patterns">
                    <div class="chart-caption">Figure: Decision changes after wins vs losses - evidence of learning/adaptation</div>
                </div>
"""

    # Comprehensive Decision Analysis
    if (
        "statistical_validation" in data_sources
        and "decision_effectiveness" in data_sources["statistical_validation"]
    ):
        decision_data = data_sources["statistical_validation"]["decision_effectiveness"]

        # Decision Distribution
        if decision_data.get("decision_distribution"):
            html += """
                <h3>Decision Distribution</h3>
                <div class="metric-grid">
"""
            for decision, stats in decision_data["decision_distribution"].items():
                html += f"""
                    <div class="metric-card">
                        <div class="label">{decision} Decisions</div>
                        <div class="value">{stats['count']}</div>
                        <small>{stats['percentage']:.1f}% of total</small>
                    </div>
"""
            html += """
                </div>
"""

        # Overall Effectiveness
        if decision_data.get("overall_effectiveness"):
            overall = decision_data["overall_effectiveness"]
            html += f"""
                <h3>Overall Decision Effectiveness</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Total Decisions</div>
                        <div class="value">{overall['total_decisions']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Overall Win Rate</div>
                        <div class="value{' positive' if overall['overall_win_rate'] > 50 else ' negative'}">{overall['overall_win_rate']:.1f}%</div>
                        <small>Profitable decisions</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Avg Daily Return</div>
                        <div class="value{' positive' if overall['overall_avg_return'] > 0 else ' negative'}">{overall['overall_avg_return']:.3f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value{' positive' if overall['overall_sharpe'] > 0 else ' negative'}">{overall['overall_sharpe']:.2f}</div>
                        <small>Risk-adjusted return</small>
                    </div>
                </div>
"""

        # Individual Decision Performance
        if decision_data.get("decision_performance"):
            html += """
                <h3>Performance by Decision Type</h3>
                <div class="metric-grid">
"""
            for decision, perf in decision_data["decision_performance"].items():
                excess_class = (
                    " positive" if perf["excess_return_annualized"] > 0 else " negative"
                )
                html += f"""
                    <div class="metric-card">
                        <div class="label">{decision} Performance</div>
                        <div class="value{excess_class}">{perf['excess_return_annualized']:+.1f}%</div>
                        <small>Annual excess return</small>
                    </div>
"""
            html += """
                </div>

                <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                    <thead>
                        <tr style="background: #f8fafc;">
                            <th style="padding: 10px; text-align: left; border: 1px solid #e2e8f0;">Decision</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">Win Rate</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">Avg Return</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">Sharpe</th>
                            <th style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">Frequency</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for decision, perf in decision_data["decision_performance"].items():
                html += f"""
                        <tr>
                            <td style="padding: 10px; border: 1px solid #e2e8f0; font-weight: bold;">{decision}</td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">{perf['win_rate']:.1f}%</td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">{perf['avg_daily_return']:.3f}%</td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">{perf['sharpe_ratio']:.2f}</td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #e2e8f0;">{perf['decision_frequency_pct']:.1f}%</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
"""

        # Risk-Adjusted Analysis Summary
        if decision_data.get("risk_adjusted_analysis"):
            risk_adj = decision_data["risk_adjusted_analysis"]
            html += f"""
                <h3>Decision Strategy Insights</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Best Decision</div>
                        <div class="value">{risk_adj['best_decision']}</div>
                        <small>{risk_adj['best_excess_return']:+.1f}% excess return</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Worst Decision</div>
                        <div class="value">{risk_adj['worst_decision']}</div>
                        <small>{risk_adj['worst_excess_return']:+.1f}% excess return</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Consistency</div>
                        <div class="value">{risk_adj['decision_consistency'].title()}</div>
                        <small>Performance across decisions</small>
                    </div>
                </div>
"""

    # Legacy HOLD Analysis (for detailed context)
    if (
        "statistical_validation" in data_sources
        and "hold_decision_analysis" in data_sources["statistical_validation"]
    ):
        hold_data = data_sources["statistical_validation"]["hold_decision_analysis"]

        if "combined_assessment" in hold_data and "note" not in hold_data:
            combined = hold_data["combined_assessment"]
            hold_score = combined.get("overall_score", 0)
            hold_rating = (
                "Excellent"
                if hold_score > 0.6
                else "Good" if hold_score > 0.4 else "Poor"
            )

            html += f"""
                <h3>Detailed HOLD Analysis</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">HOLD Success Rate</div>
                        <div class="value{' positive' if hold_score > 0.4 else ' negative'}">{hold_score:.1%}</div>
                        <small>{hold_rating}</small>
                    </div>
                </div>

                <h4>Quiet Market Performance</h4>
                <ul class="insights-list">
                    <li><strong>Success Rate:</strong> {hold_data.get('quiet_market_success', {}).get('success_rate', 0):.1%}</li>
                    <li><strong>Assessment:</strong> {hold_data.get('quiet_market_success', {}).get('interpretation', 'N/A')[:60]}...</li>
                </ul>
"""

    html += "            </div>\n"
    return html


def generate_performance_overview_html(data_sources: Dict, model_tag: str) -> str:
    """Generate performance overview section in HTML."""
    html = """            <div class="section">
                <h2>🎯 Performance Overview</h2>
"""

    # Include baseline comparison plot if available
    if "plots" in data_sources and "baseline_comparison" in data_sources["plots"]:
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['baseline_comparison']}" alt="Baseline Comparison">
                    <div class="chart-caption">Figure 1: Strategy performance vs baseline strategies</div>
                </div>
"""

    # Include equity curves if available
    if "plots" in data_sources and "equity_curves" in data_sources["plots"]:
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['equity_curves']}" alt="Equity Curves">
                    <div class="chart-caption">Figure 2: Equity curves over time</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_statistical_validation_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate statistical validation section in HTML."""
    html = """            <div class="section">
                <h2>📊 Statistical Validation</h2>
"""

    if "statistical_validation" in data_sources:
        # Use the existing print function but capture output
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            print_validation_report(data_sources["statistical_validation"], model_tag)
            validation_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        # Convert the console output to HTML
        html += f"""
                <div class="code-block">{validation_text}</div>
"""

    # Include statistical visualization if available
    if (
        "statistical_plots" in data_sources
        and "statistical_validation" in data_sources["statistical_plots"]
    ):
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['statistical_plots']['statistical_validation']}" alt="Statistical Validation Visualization">
                    <div class="chart-caption">Figure 3: Bootstrap distribution and confidence intervals</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_decision_analysis_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate decision analysis section in HTML."""
    html = """            <div class="section">
                <h2>🎪 Decision Analysis</h2>
"""

    # Calibration plots
    if "plots" in data_sources:
        if "calibration" in data_sources["plots"]:
            html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration']}" alt="Calibration Plot">
                    <div class="chart-caption">Figure 4: Prediction confidence vs actual performance</div>
                </div>
"""

        if "calibration_by_decision" in data_sources["plots"]:
            html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration_by_decision']}" alt="Calibration by Decision">
                    <div class="chart-caption">Figure 5: Calibration analysis by decision type (BUY/HOLD/SELL)</div>
                </div>
"""

    # Include calibration analysis text
    if "calibration_analysis" in data_sources:
        cal_text = data_sources["calibration_analysis"]
        html += f"""
                <h3>Detailed Calibration Analysis</h3>
                <div class="code-block">{cal_text}</div>
"""

    # Decision patterns
    if "plots" in data_sources and "decision_patterns" in data_sources["plots"]:
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['decision_patterns']}" alt="Decision Patterns">
                    <div class="chart-caption">Figure 6: Decision patterns after wins vs losses</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_risk_analysis_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate risk analysis section in HTML."""
    html = """            <div class="section">
                <h2>📉 Risk Analysis</h2>
"""

    # Include risk analysis chart if available
    if (
        "risk_analysis_plots" in data_sources
        and "risk_analysis" in data_sources["risk_analysis_plots"]
    ):
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['risk_analysis_plots']['risk_analysis']}" alt="Risk Analysis">
                    <div class="chart-caption">Figure 7: Comprehensive risk analysis including VaR, drawdowns, and stress tests</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_hold_analysis_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate HOLD decision analysis section in HTML."""
    html = """            <div class="section">
                <h2>🛡️ HOLD Decision Analysis</h2>
"""

    if (
        "statistical_validation" in data_sources
        and "hold_decision_analysis" in data_sources["statistical_validation"]
    ):
        hold_data = data_sources["statistical_validation"]["hold_decision_analysis"]

        if "note" in hold_data:
            html += f"""
                <p>{hold_data['note']}</p>
"""
        else:
            # Overall assessment
            overall_rate = hold_data["overall_hold_success_rate"]
            category = hold_data["combined_assessment"]["performance_category"]

            html += f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Overall HOLD Success Rate</div>
                        <div class="value{' positive' if overall_rate > 0.5 else ' negative'}">{overall_rate:.1%}</div>
                        <small>({category.upper()})</small>
                    </div>
                </div>

                <h3>Quiet Market Success (&lt;0.2% Daily Moves)</h3>
"""

            quiet = hold_data["quiet_market_success"]
            html += f"""
                <ul class="insights-list">
                    <li><strong>Success Rate:</strong> {quiet['success_rate']:.1%}</li>
                    <li><strong>Successful HOLDs:</strong> {quiet['successful_holds']}/{quiet['total_holds']}</li>
                    <li><strong>{quiet['interpretation']}</strong></li>
                </ul>
"""

            # Contextual correctness
            html += """
                <h3>Contextual Decision Correctness</h3>
"""

            context = hold_data["contextual_correctness"]
            html += f"""
                <ul class="insights-list">
                    <li><strong>Average Context Score:</strong> {context['avg_context_score']:.2f}</li>
                    <li><strong>Context Success Rate:</strong> {context['context_success_rate']:.1%}</li>
                    <li><strong>{context['interpretation']}</strong></li>
                </ul>
"""

            # Context reasons breakdown
            if context["reason_breakdown"]:
                html += """
                <h4>Top Context Reasons:</h4>
                <ul class="insights-list">
"""
                for reason, count in sorted(
                    context["reason_breakdown"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    reason_name = reason.replace("_", " ").title()
                    html += f"                    <li>{reason_name}: {count} decisions</li>\n"
                html += "                </ul>\n"

            # Combined assessment
            combined = hold_data["combined_assessment"]
            html += """
                <h3>Combined Assessment</h3>
                <ul class="insights-list">
"""
            html += f"""
                    <li><strong>Quiet Market Weight:</strong> {combined['quiet_weight']:.0%}</li>
                    <li><strong>Context Weight:</strong> {combined['context_weight']:.0%}</li>
                    <li><strong>Overall Score:</strong> {combined['overall_score']:.1%}</li>
                </ul>
"""

            # HOLD statistics
            stats = hold_data["hold_statistics"]
            html += """
                <h3>HOLD Usage Statistics</h3>
                <ul class="insights-list">
"""
            html += f"""
                    <li><strong>Total HOLD Decisions:</strong> {stats['total_hold_decisions']}</li>
                    <li><strong>HOLD Usage Rate:</strong> {stats['hold_percentage']:.1%}</li>
                    <li><strong>Avg Market Move During HOLD:</strong> {stats['avg_market_move_during_hold']:.2%}</li>
                    <li><strong>HOLD During Quiet Markets:</strong> {stats['hold_during_quiet_pct']:.1%}</li>
                </ul>
"""

            # Summary insight
            if "summary" in hold_data:
                html += f"""
                <div style="background: #f0f9ff; padding: 20px; border-radius: 8px; border-left: 4px solid #0ea5e9; margin-top: 20px;">
                    <strong>💡 Key Insight:</strong> {hold_data['summary']['key_insight']}
                </div>
"""

    else:
        html += """
                <p>HOLD decision analysis not available. Run statistical validation first.</p>
"""

    html += "            </div>\n"
    return html


def generate_additional_charts_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate additional charts section in HTML."""
    html = """            <div class="section">
                <h2>📈 Additional Performance Analysis</h2>
"""

    # Rolling performance charts
    if (
        "rolling_performance_plots" in data_sources
        and "rolling_performance" in data_sources["rolling_performance_plots"]
    ):
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['rolling_performance_plots']['rolling_performance']}" alt="Rolling Performance">
                    <div class="chart-caption">Figure 8: Rolling performance metrics over different time windows</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_insights_recommendations_html(data_sources: Dict, model_tag: str) -> str:
    """Generate insights and recommendations section in HTML."""
    html = """            <div class="section">
                <h2>💡 Key Insights & Recommendations</h2>
"""

    # Extract insights from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]

        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]

            html += """
                <h3>Key Findings</h3>
                <ul class="insights-list">
"""
            for finding in sa.get("key_findings", []):
                html += f"                    <li>{finding}</li>\n"
            html += "                </ul>\n"

            html += """
                <h3>Recommendations</h3>
                <ul class="insights-list">
"""
            for rec in sa.get("recommendations", []):
                html += f"                    <li>{rec}</li>\n"
            html += "                </ul>\n"

    html += """
                <h3>Overall Assessment</h3>
                <div class="metric-grid">
"""

    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        if "summary_assessment" in sv:
            sa = sv["summary_assessment"]
            html += f"""
                    <div class="metric-card">
                        <div class="label">Assessment</div>
                        <div class="value">{sa.get('overall_assessment', 'Unknown').upper()}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Confidence Level</div>
                        <div class="value">{sa.get('confidence_level', 'Unknown').upper()}</div>
                    </div>
"""

    html += """
                </div>
            </div>
"""

    return html


def generate_market_regime_analysis_html(data_sources: Dict, model_tag: str) -> str:
    """Generate market regime analysis section in HTML."""
    html = """            <div class="section">
                <h2>📊 Market Regime Analysis</h2>
                <p>Performance breakdown by market conditions reveals how the strategy adapts to different environments.</p>
"""

    if "parsed_data" not in data_sources:
        html += """
                <p>Market regime analysis requires parsed trading data.</p>
"""
        html += "            </div>\n"
        return html

    parsed_df = data_sources["parsed_data"]

    if "next_return_1d" in parsed_df.columns and "strategy_return" in parsed_df.columns:
        market_returns = parsed_df["next_return_1d"]
        rolling_vol = market_returns.rolling(20).std() * np.sqrt(252)

        vol_median = rolling_vol.median()
        vol_high = rolling_vol.quantile(0.75)

        regimes = []
        for vol in rolling_vol:
            if pd.isna(vol):
                regimes.append("Unknown")
            elif vol > vol_high:
                regimes.append("High Volatility")
            elif vol > vol_median:
                regimes.append("Moderate Volatility")
            else:
                regimes.append("Low Volatility")

        parsed_df = parsed_df.copy()
        parsed_df["regime"] = regimes

        regime_performance = []
        for regime in ["Low Volatility", "Moderate Volatility", "High Volatility"]:
            regime_data = parsed_df[parsed_df["regime"] == regime]
            if len(regime_data) > 0:
                strategy_return = regime_data["strategy_return"].mean() * 252
                market_return = regime_data["next_return_1d"].mean() * 252
                win_rate = (regime_data["strategy_return"] > 0).mean() * 100
                days = len(regime_data)

                regime_performance.append(
                    {
                        "regime": regime,
                        "strategy_return": strategy_return,
                        "market_return": market_return,
                        "excess_return": strategy_return - market_return,
                        "win_rate": win_rate,
                        "days": days,
                    }
                )

        if regime_performance:
            html += """
                <div class="metric-grid">
"""
            for perf in regime_performance:
                html += f"""
                    <div class="metric-card">
                        <div class="label">{perf['regime']}</div>
                        <div class="value{' positive' if perf['excess_return'] > 0 else ' negative'}">{perf['excess_return']:+.2f}%</div>
                        <small>Excess Return</small>
                    </div>
"""

            html += """
                </div>

                <h3>Key Regime Insights</h3>
                <ul class="insights-list">
"""

            best_regime = max(regime_performance, key=lambda x: x["excess_return"])
            worst_regime = min(regime_performance, key=lambda x: x["excess_return"])

            adaptation_quality = (
                "✅ Adapts well to changing conditions"
                if abs(best_regime["excess_return"] - worst_regime["excess_return"]) < 5
                else "⚠️ Performance varies significantly by regime"
            )

            html += f"""
                    <li><strong>Best Performance:</strong> {best_regime['regime']} regime ({best_regime['excess_return']:+.2f}% excess return)</li>
                    <li><strong>Worst Performance:</strong> {worst_regime['regime']} regime ({worst_regime['excess_return']:+.2f}% excess return)</li>
                    <li><strong>Strategy Adaptation:</strong> {adaptation_quality}</li>
                </ul>

                <h3>Practical Implications</h3>
                <ul class="insights-list">
                    <li><strong>Portfolio Integration:</strong> Consider regime-based allocation adjustments</li>
                    <li><strong>Risk Management:</strong> Higher volatility periods may require position size reduction</li>
                    <li><strong>Strategy Optimization:</strong> Focus improvement efforts on worst-performing regimes</li>
                </ul>
"""

    html += "            </div>\n"
    return html


def generate_risk_attribution_html(data_sources: Dict, model_tag: str) -> str:
    """Generate risk attribution analysis section in HTML."""
    html = """            <div class="section">
                <h2>🔍 Risk Attribution Analysis</h2>
                <p>Understanding the sources of returns and risk helps identify strategy strengths and weaknesses.</p>
"""

    if "parsed_data" not in data_sources:
        html += """
                <p>Risk attribution analysis requires parsed trading data.</p>
"""
        html += "            </div>\n"
        return html

    parsed_df = data_sources["parsed_data"]

    if "strategy_return" in parsed_df.columns and "next_return_1d" in parsed_df.columns:
        strategy_returns = parsed_df["strategy_return"]
        market_returns = parsed_df["next_return_1d"]

        strategy_vol = strategy_returns.std() * np.sqrt(252)
        market_vol = market_returns.std() * np.sqrt(252)
        correlation = strategy_returns.corr(market_returns)

        if market_vol > 0:
            beta = correlation * (strategy_vol / market_vol)
            alpha = (strategy_returns.mean() - beta * market_returns.mean()) * 252

            systematic_risk = beta**2 * market_vol**2
            idiosyncratic_risk = strategy_vol**2 - systematic_risk

            html += f"""
                <h3>Risk Decomposition</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Total Strategy Risk</div>
                        <div class="value">{strategy_vol:.2f}%</div>
                        <small>Annualized Volatility</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Systematic Risk</div>
                        <div class="value">{np.sqrt(systematic_risk):.2f}%</div>
                        <small>Beta-related</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Idiosyncratic Risk</div>
                        <div class="value">{np.sqrt(idiosyncratic_risk):.2f}%</div>
                        <small>Strategy-specific</small>
                    </div>
                </div>

                <h3>Risk-Adjusted Performance</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Beta</div>
                        <div class="value">{beta:.3f}</div>
                        <small>{'High systematic risk' if beta > 1.2 else 'Moderate systematic risk' if beta > 0.8 else 'Low systematic risk'}</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Alpha</div>
                        <div class="value{' positive' if alpha > 0 else ' negative'}">{alpha:.2f}%</div>
                        <small>Annualized excess return</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Market Correlation</div>
                        <div class="value">{correlation:.3f}</div>
                        <small>{'Highly correlated' if abs(correlation) > 0.7 else 'Moderately correlated' if abs(correlation) > 0.3 else 'Low correlation'}</small>
                    </div>
                </div>
"""

            winning_days = strategy_returns > 0
            losing_days = strategy_returns < 0

            avg_win = strategy_returns[winning_days].mean()
            avg_loss = strategy_returns[losing_days].mean()
            win_rate = winning_days.mean()

            profit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

            win_rate_desc = (
                "Good" if win_rate > 0.55 else "Fair" if win_rate > 0.45 else "Poor"
            )
            profit_ratio_desc = (
                "Good"
                if profit_ratio > 1.5
                else "Fair" if profit_ratio > 1.0 else "Poor"
            )

            html += f"""
                <h3>Profitability Analysis</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Win Rate</div>
                        <div class="value">{win_rate:.1%}</div>
                        <small>{win_rate_desc}</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Profit/Loss Ratio</div>
                        <div class="value">{profit_ratio:.2f}</div>
                        <small>{profit_ratio_desc}</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Expectancy</div>
                        <div class="value{' positive' if expectancy > 0 else ' negative'}">{expectancy:.3f}%</div>
                        <small>Expected daily return</small>
                    </div>
                </div>

                <h3>Risk Management Insights</h3>
                <ul class="insights-list">
"""

            return_skew = (
                "favorable skew"
                if expectancy > 0 and profit_ratio > 1.2
                else "mixed characteristics"
            )
            risk_profile = (
                "Aggressive"
                if strategy_vol > market_vol * 1.5
                else (
                    "Conservative" if strategy_vol < market_vol * 0.7 else "Market-like"
                )
            )
            diversification = (
                "Low correlation provides diversification benefits"
                if abs(correlation) < 0.5
                else "High correlation suggests limited diversification"
            )

            html += f"""
                    <li><strong>Return Distribution:</strong> Strategy shows {return_skew}</li>
                    <li><strong>Risk Profile:</strong> {risk_profile}</li>
                    <li><strong>Diversification:</strong> {diversification}</li>
                </ul>
"""

    html += "            </div>\n"
    return html


def generate_practical_considerations_html(data_sources: Dict, model_tag: str) -> str:
    """Generate practical implementation considerations section in HTML."""
    html = """            <div class="section">
                <h2>🛠️ Practical Implementation Considerations</h2>
                <p>Real-world deployment requires addressing transaction costs, liquidity, and operational factors.</p>
"""

    # Transaction costs analysis
    if "parsed_data" in data_sources:
        parsed_df = data_sources["parsed_data"]

        if "decision" in parsed_df.columns:
            decisions = parsed_df["decision"]

            position_changes = 0
            prev_decision = None

            for decision in decisions:
                if prev_decision is not None and decision != prev_decision:
                    position_changes += 1
                prev_decision = decision

            trading_frequency = position_changes / len(parsed_df) * 100
            annual_trades = position_changes * (252 / len(parsed_df))

            avg_commission = 0.001
            avg_spread = 0.0005
            total_cost_per_trade = avg_commission + avg_spread

            annual_cost_bps = annual_trades * total_cost_per_trade * 10000

            cost_impact = (
                "Significant"
                if annual_cost_bps > 50
                else "Moderate" if annual_cost_bps > 20 else "Minimal"
            )

            html += f"""
                <h3>Transaction Costs Impact</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Trading Frequency</div>
                        <div class="value">{trading_frequency:.1f}%</div>
                        <small>Days with position changes</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Annual Trades</div>
                        <div class="value">{annual_trades:.0f}</div>
                        <small>Round trip trades</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Trading Costs</div>
                        <div class="value">{annual_cost_bps:.0f} bps</div>
                        <small>{cost_impact} impact</small>
                    </div>
                </div>
"""

    html += """
                <h3>Operational Considerations</h3>

                <h4>Technical Infrastructure</h4>
                <ul class="insights-list">
                    <li><strong>API Reliability:</strong> LLM responses must be consistent and available during market hours</li>
                    <li><strong>Response Time:</strong> Decision latency should be under 100ms for real-time trading</li>
                    <li><strong>Fallback Mechanisms:</strong> Alternative decision rules when LLM unavailable</li>
                    <li><strong>Monitoring:</strong> Real-time performance tracking and automated alerts</li>
                </ul>

                <h4>Risk Management</h4>
                <ul class="insights-list">
                    <li><strong>Position Limits:</strong> Maximum exposure per asset/sector</li>
                    <li><strong>Drawdown Controls:</strong> Automatic reduction during losing streaks</li>
                    <li><strong>Liquidity Checks:</strong> Ensure sufficient volume for position sizing</li>
                    <li><strong>Market Impact:</strong> Consider price impact of larger orders</li>
                </ul>

                <h4>Regulatory & Compliance</h4>
                <ul class="insights-list">
                    <li><strong>Audit Trail:</strong> Complete record of decision-making process</li>
                    <li><strong>Explainability:</strong> Ability to explain AI-driven trades to regulators</li>
                    <li><strong>Bias Monitoring:</strong> Regular checks for systematic biases</li>
                    <li><strong>Testing Requirements:</strong> Validation across multiple market scenarios</li>
                </ul>

                <h3>Scaling Considerations</h3>
                <ul class="insights-list">
                    <li><strong>Cost Efficiency:</strong> LLM API costs vs traditional strategy development</li>
                    <li><strong>Performance Consistency:</strong> Stability across different market conditions</li>
                    <li><strong>Portfolio Size:</strong> Impact of strategy capacity and market impact</li>
                    <li><strong>Multi-Asset Extension:</strong> Applicability beyond single-asset strategies</li>
                </ul>
"""

    # Performance expectations and deployment recommendations
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        bs = sv.get("bootstrap_vs_index", {})

        html += """
                <h3>Deployment Recommendations</h3>
"""

        if bs.get("significant_difference_5pct", False):
            if bs.get("sharpe_difference", 0) > 0:
                html += """
                <div style="background: #dcfce7; padding: 20px; border-radius: 8px; border-left: 4px solid #16a34a; margin: 20px 0;">
                    <strong>✅ Recommended for live deployment</strong> with proper risk controls
                    <ul style="margin-top: 10px;">
                        <li>Implement position sizing based on confidence scores</li>
                        <li>Monitor for overfitting in live performance</li>
                        <li>Consider hybrid approach combining AI with traditional rules</li>
                    </ul>
                </div>
"""
            else:
                html += """
                <div style="background: #fef3c7; padding: 20px; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                    <strong>⚠️ Not recommended for live deployment</strong> in current form
                    <ul style="margin-top: 10px;">
                        <li>Requires significant strategy refinement</li>
                        <li>Consider as research baseline rather than production strategy</li>
                        <li>May be suitable for specialized market conditions</li>
                    </ul>
                </div>
"""
        else:
            html += """
                <div style="background: #e0f2fe; padding: 20px; border-radius: 8px; border-left: 4px solid #0284c7; margin: 20px 0;">
                    <strong>🔄 Further testing required</strong> before deployment decision
                    <ul style="margin-top: 10px;">
                        <li>Results not statistically significant from market index</li>
                        <li>Additional validation across different time periods needed</li>
                        <li>Consider as experimental approach rather than primary strategy</li>
                    </ul>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_technical_details_html(data_sources: Dict, model_tag: str) -> str:
    """Generate technical details section in HTML."""
    html = f"""            <div class="section">
                <h2>📁 Technical Details</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Model Tag</div>
                        <div class="value">{model_tag}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Generated</div>
                        <div class="value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                    </div>
                </div>

                <h3>Data Sources</h3>
                <div class="data-sources">
"""

    # List all data sources used
    for source_name, source_data in data_sources.items():
        if source_name == "plots":
            html += f"""                    <div class="data-source-item">Plots: {len(source_data)} chart files</div>
"""
        elif source_name == "statistical_validation":
            html += """                    <div class="data-source-item">Statistical Validation: Bootstrap analysis and out-of-sample testing</div>
"""
        elif source_name == "baseline_comparison":
            html += """                    <div class="data-source-item">Baseline Comparison: Performance vs multiple strategies</div>
"""
        elif source_name == "parsed_data":
            n_periods = len(source_data) if hasattr(source_data, "__len__") else "N/A"
            html += f"""                    <div class="data-source-item">Parsed Data: {n_periods} trading periods</div>
"""
        else:
            html += f"""                    <div class="data-source-item">{source_name.replace('_', ' ').title()}: Available</div>
"""

    html += """                </div>

                <h3>Analysis Components</h3>
                <ul class="insights-list">
                    <li>Statistical significance testing (bootstrap)</li>
                    <li>Out-of-sample validation</li>
                    <li>Risk-adjusted performance metrics</li>
                    <li>Decision calibration analysis</li>
                    <li>Baseline strategy comparisons</li>
                    <li>Rolling performance analysis</li>
                </ul>
            </div>
"""

    return html


# Standalone execution
if __name__ == "__main__":
    # Example usage - handle imports when run as script
    import argparse
    import sys

    # Add current directory to path for imports when run as script
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(
        description="Generate comprehensive experiment report"
    )
    parser.add_argument(
        "model_tag",
        nargs="?",
        default="dummy_model_memory_only",
        help="Model identifier (default: dummy_model_memory_only)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip generating additional charts"
    )

    args = parser.parse_args()

    try:
        report_path = generate_comprehensive_report(
            model_tag=args.model_tag,
            include_additional_charts=not args.no_charts,
            output_format=args.format,
        )
        print(f"Comprehensive report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback

        traceback.print_exc()
