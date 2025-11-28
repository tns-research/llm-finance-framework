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

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import existing analysis modules
try:
    # Try relative imports (when imported as module)
    from .statistical_validation import print_validation_report
    from .baselines import calculate_baseline_metrics
except ImportError:
    # Fall back to absolute imports (when run as script)
    from statistical_validation import print_validation_report
    from baselines import calculate_baseline_metrics


def generate_comprehensive_report(
    model_tag: str,
    base_dir: str = None,
    include_additional_charts: bool = True,
    output_format: str = "markdown"
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
            if key == 'plots':
                print(f"    {key}: {len(value)} plot files")
            elif hasattr(value, '__len__') and not isinstance(value, str):
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

        with open(report_path, 'w', encoding='utf-8') as f:
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
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(error_content)

        print(f"✓ Error report generated: {error_path}")
        return str(error_path)


def collect_data_sources(model_tag: str, analysis_dir: Path, plots_dir: Path) -> Dict:
    """
    Collect all available data sources for the model.
    """
    sources = {}

    # Statistical validation JSON
    stat_validation_file = analysis_dir / f"{model_tag}_statistical_validation.json"
    if stat_validation_file.exists():
        with open(stat_validation_file, 'r') as f:
            sources['statistical_validation'] = json.load(f)

    # Baseline comparison CSV
    baseline_csv = analysis_dir / f"{model_tag}_baseline_comparison.csv"
    if baseline_csv.exists():
        sources['baseline_comparison'] = pd.read_csv(baseline_csv)

    # Calibration analysis markdown
    calibration_md = analysis_dir / f"{model_tag}_calibration_analysis.md"
    if calibration_md.exists():
        with open(calibration_md, 'r') as f:
            sources['calibration_analysis'] = f.read()

    # Pattern analysis markdown
    pattern_md = analysis_dir / f"{model_tag}_pattern_analysis.md"
    if pattern_md.exists():
        with open(pattern_md, 'r') as f:
            sources['pattern_analysis'] = f.read()

    # Parsed results CSV (for additional analysis)
    parsed_dir = analysis_dir.parent.parent / "parsed"  # Go up to results, then to parsed
    parsed_csv = parsed_dir / f"{model_tag}_parsed.csv"
    if parsed_csv.exists():
        sources['parsed_data'] = pd.read_csv(parsed_csv, parse_dates=['date'])

    # Collect plot files - generate paths relative to reports directory
    sources['plots'] = {}
    plot_extensions = ['.png', '.jpg', '.jpeg']

    for ext in plot_extensions:
        for plot_file in plots_dir.glob(f"{model_tag}*{ext}"):
            plot_name = plot_file.stem.replace(f"{model_tag}_", "")
            # Path from reports/ to plots/ is ../plots/filename.png
            sources['plots'][plot_name] = f"../plots/{plot_file.name}"

    return sources


def generate_additional_charts(data_sources: Dict, model_tag: str, plots_dir: Path) -> Dict:
    """
    Generate charts that are missing from current analysis.
    """
    additional_charts = {}

    if 'parsed_data' not in data_sources:
        return additional_charts

    parsed_df = data_sources['parsed_data']

    # Rolling performance charts
    additional_charts['rolling_performance_plots'] = create_rolling_performance_charts(
        parsed_df, model_tag, plots_dir
    )

    # Risk analysis charts
    additional_charts['risk_analysis_plots'] = create_risk_analysis_charts(
        parsed_df, model_tag, plots_dir
    )

    # Statistical visualizations (if validation data available)
    if 'statistical_validation' in data_sources:
        additional_charts['statistical_plots'] = create_statistical_visualizations(
            data_sources['statistical_validation'], model_tag, plots_dir
        )

    return additional_charts


def create_rolling_performance_charts(parsed_df: pd.DataFrame, model_tag: str, plots_dir: Path) -> Dict[str, str]:
    """
    Create rolling performance charts (Sharpe, returns, drawdowns).
    """
    charts = {}

    # Ensure we have the necessary columns
    required_cols = ['date', 'strategy_return', 'next_return_1d', 'cumulative_return']
    if not all(col in parsed_df.columns for col in required_cols):
        return charts

    # Sort by date
    df = parsed_df.sort_values('date').copy()

    # Calculate rolling metrics
    window_sizes = [63, 126, 252]  # ~3, 6, 12 months

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Rolling Performance Analysis - {model_tag}', fontsize=16, fontweight='bold')

    # 1. Rolling Sharpe Ratio
    ax1 = axes[0, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_sharpe = df['strategy_return'].rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            ax1.plot(df['date'], rolling_sharpe, label=f'{window//21}M Window', alpha=0.8)

    ax1.set_title('Rolling Sharpe Ratio', fontweight='bold')
    ax1.set_ylabel('Annualized Sharpe Ratio')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Rolling Returns
    ax2 = axes[0, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_returns = df['strategy_return'].rolling(window).sum()
            ax2.plot(df['date'], rolling_returns, label=f'{window//21}M Window', alpha=0.8)

    ax2.set_title('Rolling Total Returns', fontweight='bold')
    ax2.set_ylabel('Total Return (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Rolling Maximum Drawdown
    ax3 = axes[1, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_cumulative = df['strategy_return'].rolling(window).sum()
            rolling_max = rolling_cumulative.rolling(window, min_periods=1).max()
            rolling_dd = rolling_cumulative - rolling_max
            ax3.plot(df['date'], rolling_dd, label=f'{window//21}M Window', alpha=0.8)

    ax3.set_title('Rolling Maximum Drawdown', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Rolling Win Rate
    ax4 = axes[1, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_wins = (df['strategy_return'] > 0).rolling(window).sum()
            rolling_win_rate = rolling_wins / window * 100
            ax4.plot(df['date'], rolling_win_rate, label=f'{window//21}M Window', alpha=0.8)

    ax4.set_title('Rolling Win Rate', fontweight='bold')
    ax4.set_ylabel('Win Rate (%)')
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% Line')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    chart_path = plots_dir / f"{model_tag}_rolling_performance.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    charts['rolling_performance'] = str(chart_path.relative_to(plots_dir.parent))
    print(f"✓ Rolling performance chart saved: {chart_path}")

    return charts


def create_risk_analysis_charts(parsed_df: pd.DataFrame, model_tag: str, plots_dir: Path) -> Dict[str, str]:
    """
    Create risk analysis charts (VaR, CVaR, stress tests).
    """
    charts = {}

    if 'strategy_return' not in parsed_df.columns:
        return charts

    returns = parsed_df['strategy_return'].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Risk Analysis - {model_tag}', fontsize=16, fontweight='bold')

    # 1. Returns Distribution
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=50, alpha=0.7, density=True, label='Strategy Returns')
    ax1.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.3f}%')
    ax1.set_title('Returns Distribution', fontweight='bold')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
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
            window_returns = returns[i-rolling_window:i]
            var_95.append(np.percentile(window_returns, 5))
            var_99.append(np.percentile(window_returns, 1))
            dates.append(parsed_df.iloc[i]['date'])

        ax2.plot(dates, var_95, label='VaR 95%', color='orange', alpha=0.8)
        ax2.plot(dates, var_99, label='VaR 99%', color='red', alpha=0.8)
        ax2.fill_between(dates, var_99, var_95, alpha=0.2, color='red', label='Tail Risk Zone')

    ax2.set_title('Rolling Value at Risk', fontweight='bold')
    ax2.set_ylabel('VaR (%)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Drawdown Analysis
    ax3 = axes[1, 0]
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max

    ax3.fill_between(range(len(drawdowns)), 0, drawdowns, color='red', alpha=0.3)
    ax3.plot(drawdowns, color='red', linewidth=1)
    ax3.set_title('Drawdown Analysis', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(alpha=0.3)

    # 4. Stress Test Scenarios
    ax4 = axes[1, 1]

    # Different stress scenarios
    scenarios = {
        'Base Case': returns,
        'High Volatility': returns * 2,
        'Crash Scenario': np.where(returns < np.percentile(returns, 10),
                                 returns * 3, returns),
        'Bull Market': np.where(returns > 0, returns * 1.5, returns * 0.5)
    }

    for scenario_name, scenario_returns in scenarios.items():
        cumulative_scenario = np.cumsum(scenario_returns)
        ax4.plot(cumulative_scenario, label=scenario_name, alpha=0.8)

    ax4.set_title('Stress Test Scenarios', fontweight='bold')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    chart_path = plots_dir / f"{model_tag}_risk_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    charts['risk_analysis'] = str(chart_path.relative_to(plots_dir.parent))
    print(f"✓ Risk analysis chart saved: {chart_path}")

    return charts


def create_statistical_visualizations(validation_results: Dict, model_tag: str, plots_dir: Path) -> Dict[str, str]:
    """
    Create visualizations of statistical validation results.
    """
    charts = {}

    # Bootstrap distribution plot
    if 'bootstrap_vs_index' in validation_results:
        bootstrap_data = validation_results['bootstrap_vs_index']

        if 'all_results' in bootstrap_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Statistical Validation Visualizations - {model_tag}', fontsize=14, fontweight='bold')

            # 1. Bootstrap Distribution
            ax1 = axes[0]
            bootstrap_diffs = np.array(bootstrap_data['all_results'])
            ax1.hist(bootstrap_diffs, bins=50, alpha=0.7, density=True, label='Bootstrap Distribution')
            ax1.axvline(bootstrap_data['sharpe_difference'], color='red', linestyle='--',
                       linewidth=2, label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}')
            ax1.axvline(np.mean(bootstrap_diffs), color='blue', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(bootstrap_diffs):.3f}')
            ax1.set_title('Bootstrap Sharpe Difference Distribution', fontweight='bold')
            ax1.set_xlabel('Sharpe Ratio Difference')
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. Confidence Interval
            ax2 = axes[1]
            ax2.hist(bootstrap_diffs, bins=30, alpha=0.7, density=True)
            ci_lower, ci_upper = bootstrap_data['ci_95_bootstrap']
            ax2.axvline(ci_lower, color='orange', linestyle='--', linewidth=2,
                       label=f'95% CI Lower: {ci_lower:.3f}')
            ax2.axvline(ci_upper, color='orange', linestyle='--', linewidth=2,
                       label=f'95% CI Upper: {ci_upper:.3f}')
            ax2.axvline(bootstrap_data['sharpe_difference'], color='red', linestyle='-',
                       linewidth=2, label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}')
            ax2.set_title('Confidence Interval Analysis', fontweight='bold')
            ax2.set_xlabel('Sharpe Ratio Difference')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()

            chart_path = plots_dir / f"{model_tag}_statistical_validation.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            charts['statistical_validation'] = str(chart_path.relative_to(plots_dir.parent))
            print(f"✓ Statistical validation visualization saved: {chart_path}")

    return charts


def generate_master_report(
    model_tag: str,
    data_sources: Dict,
    analysis_dir: Path,
    plots_dir: Path
) -> str:
    """
    Generate the master markdown report combining all analyses.
    """
    report_lines = []

    # Header
    report_lines.extend([
        f"# LLM Trading Strategy Experiment Report",
        f"## Model: {model_tag} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        ""
    ])

    # Debug: Add data sources summary
    report_lines.extend([
        f"## Data Sources Summary",
        f"",
        f"- Total data sources collected: {len(data_sources)}",
    ])

    for key, value in data_sources.items():
        if key == 'plots':
            report_lines.append(f"- {key}: {len(value)} plot files")
        elif hasattr(value, '__len__') and not isinstance(value, str):
            report_lines.append(f"- {key}: {len(value)} items")
        else:
            report_lines.append(f"- {key}: available")

    report_lines.extend([
        "",
        "---",
        ""
    ])

    # Executive Summary
    report_lines.extend(generate_executive_summary(data_sources, model_tag))

    # Performance Overview
    report_lines.extend(generate_performance_overview(data_sources, model_tag))

    # Statistical Validation
    report_lines.extend(generate_statistical_validation_section(data_sources, model_tag))

    # Decision Analysis
    report_lines.extend(generate_decision_analysis_section(data_sources, model_tag))

    # Risk Analysis
    report_lines.extend(generate_risk_analysis_section(data_sources, model_tag))

    # HOLD Decision Analysis
    report_lines.extend(generate_hold_analysis_section(data_sources, model_tag))

    # Additional Charts
    report_lines.extend(generate_additional_charts_section(data_sources, model_tag))

    # Key Insights & Recommendations
    report_lines.extend(generate_insights_recommendations(data_sources, model_tag))

    # Technical Details
    report_lines.extend(generate_technical_details(data_sources, model_tag))

    return "\n".join(report_lines)


def generate_master_report_html(
    model_tag: str,
    data_sources: Dict,
    analysis_dir: Path,
    plots_dir: Path
) -> str:
    """
    Generate the master HTML report combining all analyses with beautiful styling.
    """
    html_parts = []

    # HTML Header with CSS
    html_parts.append(f"""<!DOCTYPE html>
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
""")

    # Executive Summary Section
    html_parts.append(generate_executive_summary_html(data_sources, model_tag))

    # Performance Overview Section
    html_parts.append(generate_performance_overview_html(data_sources, model_tag))

    # Statistical Validation Section
    html_parts.append(generate_statistical_validation_section_html(data_sources, model_tag))

    # Decision Analysis Section
    html_parts.append(generate_decision_analysis_section_html(data_sources, model_tag))

    # Risk Analysis Section
    html_parts.append(generate_risk_analysis_section_html(data_sources, model_tag))

    # HOLD Decision Analysis Section
    html_parts.append(generate_hold_analysis_section_html(data_sources, model_tag))

    # Additional Charts Section
    html_parts.append(generate_additional_charts_section_html(data_sources, model_tag))

    # Key Insights & Recommendations Section
    html_parts.append(generate_insights_recommendations_html(data_sources, model_tag))

    # Technical Details Section
    html_parts.append(generate_technical_details_html(data_sources, model_tag))

    # Close HTML
    html_parts.append("""
        </div>
        <div class="footer">
            <p><strong>LLM Finance Experiment Framework</strong></p>
            <p>This report was automatically generated. For questions about methodology or results, refer to the technical documentation.</p>
            <p>Generated on """ + datetime.now().strftime('%Y-%m-%d at %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>""")

    return "".join(html_parts)


def generate_executive_summary(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate executive summary section."""
    lines = [
        "## 📈 Executive Summary",
        "",
    ]

    # Extract key metrics from statistical validation
    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']
        dataset = sv.get('dataset_info', {})

        lines.extend([
            f"- **Total Return**: {dataset.get('total_strategy_return', 'N/A'):.2f}% "
            f"(vs Index: {dataset.get('total_index_return', 'N/A'):.2f}%)",
            f"- **Period**: {dataset.get('n_periods', 'N/A')} trading days",
        ])

        # Bootstrap results
        if 'bootstrap_vs_index' in sv:
            bs = sv['bootstrap_vs_index']
            sig_status = "✓ Significant" if bs.get('significant_difference_5pct') else "❌ Not significant"
            lines.extend([
                f"- **Statistical Significance**: {sig_status} vs index (p={bs.get('p_value_two_sided', 'N/A'):.4f})",
                f"- **Effect Size**: {bs.get('effect_size', 'N/A'):.3f} (Cohen's d)",
            ])

        # Out-of-sample validation
        if 'out_of_sample_validation' in sv:
            oos = sv['out_of_sample_validation']
            if 'error' not in oos:
                overfitting = "🚨 Detected" if oos.get('overfitting_detection', {}).get('overall_overfitting_detected') else "✅ None detected"
                lines.append(f"- **Overfitting**: {overfitting}")

    lines.extend([
        "",
        "---",
        ""
    ])

    return lines


def generate_performance_overview(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate performance overview section."""
    lines = [
        "## 🎯 Performance Overview",
        "",
    ]

    # Include baseline comparison plot if available
    if 'plots' in data_sources and 'baseline_comparison' in data_sources['plots']:
        lines.extend([
            f"![Baseline Comparison]({data_sources['plots']['baseline_comparison']})",
            "*Figure 1: Strategy performance vs baseline strategies*",
            "",
        ])

    # Include equity curves if available
    if 'plots' in data_sources and 'equity_curves' in data_sources['plots']:
        lines.extend([
            f"![Equity Curves]({data_sources['plots']['equity_curves']})",
            "*Figure 2: Equity curves over time*",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_statistical_validation_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate statistical validation section."""
    lines = [
        "## 📊 Statistical Validation",
        "",
    ]

    if 'statistical_validation' in data_sources:
        # Use the existing print function but capture output
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            print_validation_report(data_sources['statistical_validation'], model_tag)
            validation_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        # Convert the console output to markdown
        lines.extend(validation_text.split('\n'))
        lines.append("")

    # Include statistical visualization if available
    if 'statistical_plots' in data_sources and 'statistical_validation' in data_sources['statistical_plots']:
        lines.extend([
            f"![Statistical Validation Visualization]({data_sources['statistical_plots']['statistical_validation']})",
            "*Figure 3: Bootstrap distribution and confidence intervals*",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_decision_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate decision analysis section."""
    lines = [
        "## 🎪 Decision Analysis",
        "",
    ]

    # Calibration plots
    if 'plots' in data_sources:
        if 'calibration' in data_sources['plots']:
            lines.extend([
                f"![Calibration Plot]({data_sources['plots']['calibration']})",
                "*Figure 4: Prediction confidence vs actual performance*",
                "",
            ])

        if 'calibration_by_decision' in data_sources['plots']:
            lines.extend([
                f"![Calibration by Decision]({data_sources['plots']['calibration_by_decision']})",
                "*Figure 5: Calibration analysis by decision type (BUY/HOLD/SELL)*",
                "",
            ])

    # Include calibration analysis text
    if 'calibration_analysis' in data_sources:
        lines.extend([
            "### Detailed Calibration Analysis",
            "",
            "```markdown"
        ])
        # Extract key sections from the calibration analysis
        cal_text = data_sources['calibration_analysis']
        lines.extend(cal_text.split('\n'))
        lines.extend([
            "```",
            "",
        ])

    # Decision patterns
    if 'plots' in data_sources and 'decision_patterns' in data_sources['plots']:
        lines.extend([
            f"![Decision Patterns]({data_sources['plots']['decision_patterns']})",
            "*Figure 6: Decision patterns after wins vs losses*",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_risk_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate risk analysis section."""
    lines = [
        "## 📉 Risk Analysis",
        "",
    ]

    # Include risk analysis chart if available
    if 'risk_analysis_plots' in data_sources and 'risk_analysis' in data_sources['risk_analysis_plots']:
        lines.extend([
            f"![Risk Analysis]({data_sources['risk_analysis_plots']['risk_analysis']})",
            "*Figure 7: Comprehensive risk analysis including VaR, drawdowns, and stress tests*",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_hold_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate HOLD decision analysis section."""
    lines = [
        "## 🛡️ HOLD Decision Analysis",
        "",
    ]

    if 'statistical_validation' in data_sources and 'hold_decision_analysis' in data_sources['statistical_validation']:
        hold_data = data_sources['statistical_validation']['hold_decision_analysis']

        if 'note' in hold_data:
            lines.extend([
                f"{hold_data['note']}",
                "",
            ])
        else:
            # Overall assessment
            overall_rate = hold_data['overall_hold_success_rate']
            category = hold_data['combined_assessment']['performance_category']

            lines.extend([
                f"**Overall HOLD Success Rate:** {overall_rate:.1%} ({category.upper()})",
                "",
                "### Quiet Market Success (<0.2% Daily Moves)",
                "",
            ])

            quiet = hold_data['quiet_market_success']
            lines.extend([
                f"- **Success Rate:** {quiet['success_rate']:.1%}",
                f"- **Successful HOLDs:** {quiet['successful_holds']}/{quiet['total_holds']}",
                f"- **{quiet['interpretation']}**",
                "",
            ])

            # Contextual correctness
            lines.extend([
                "### Contextual Decision Correctness",
                "",
            ])

            context = hold_data['contextual_correctness']
            lines.extend([
                f"- **Average Context Score:** {context['avg_context_score']:.2f}",
                f"- **Context Success Rate:** {context['context_success_rate']:.1%}",
                f"- **{context['interpretation']}**",
                "",
            ])

            # Context reasons breakdown
            if context['reason_breakdown']:
                lines.extend([
                    "**Top Context Reasons:**",
                    "",
                ])
                for reason, count in sorted(context['reason_breakdown'].items(), key=lambda x: x[1], reverse=True):
                    reason_name = reason.replace('_', ' ').title()
                    lines.extend([
                        f"- {reason_name}: {count} decisions",
                    ])
                lines.append("")

            # Combined assessment
            combined = hold_data['combined_assessment']
            lines.extend([
                "### Combined Assessment",
                "",
                f"- **Quiet Market Weight:** {combined['quiet_weight']:.0%}",
                f"- **Context Weight:** {combined['context_weight']:.0%}",
                f"- **Overall Score:** {combined['overall_score']:.1%}",
                "",
            ])

            # HOLD statistics
            stats = hold_data['hold_statistics']
            lines.extend([
                "### HOLD Usage Statistics",
                "",
                f"- **Total HOLD Decisions:** {stats['total_hold_decisions']}",
                f"- **HOLD Usage Rate:** {stats['hold_percentage']:.1%}",
                f"- **Avg Market Move During HOLD:** {stats['avg_market_move_during_hold']:.2%}",
                f"- **HOLD During Quiet Markets:** {stats['hold_during_quiet_pct']:.1%}",
                "",
            ])

            # Summary insight
            if 'summary' in hold_data:
                lines.extend([
                    f"**Key Insight:** {hold_data['summary']['key_insight']}",
                    "",
                ])

    else:
        lines.extend([
            "HOLD decision analysis not available. Run statistical validation first.",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_additional_charts_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate additional charts section."""
    lines = [
        "## 📈 Additional Performance Analysis",
        "",
    ]

    # Rolling performance charts
    if 'rolling_performance_plots' in data_sources and 'rolling_performance' in data_sources['rolling_performance_plots']:
        lines.extend([
            f"![Rolling Performance]({data_sources['rolling_performance_plots']['rolling_performance']})",
            "*Figure 8: Rolling performance metrics over different time windows*",
            "",
        ])

    lines.extend([
        "---",
        ""
    ])

    return lines


def generate_insights_recommendations(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate insights and recommendations section."""
    lines = [
        "## 💡 Key Insights & Recommendations",
        "",
    ]

    # Extract insights from statistical validation
    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']

        if 'summary_assessment' in sv:
            sa = sv['summary_assessment']

            lines.append("### Key Findings")
            for finding in sa.get('key_findings', []):
                lines.append(f"- {finding}")

            lines.append("")
            lines.append("### Recommendations")
            for rec in sa.get('recommendations', []):
                lines.append(f"- {rec}")

    lines.extend([
        "",
        "### Overall Assessment",
    ])

    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']
        if 'summary_assessment' in sv:
            sa = sv['summary_assessment']
            lines.extend([
                f"- **Assessment**: {sa.get('overall_assessment', 'Unknown').upper()}",
                f"- **Confidence Level**: {sa.get('confidence_level', 'Unknown').upper()}",
            ])

    lines.extend([
        "",
        "---",
        ""
    ])

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
        if source_name == 'plots':
            lines.append(f"- **Plots**: {len(source_data)} chart files")
        elif source_name == 'statistical_validation':
            lines.append("- **Statistical Validation**: Bootstrap analysis and out-of-sample testing")
        elif source_name == 'baseline_comparison':
            lines.append("- **Baseline Comparison**: Performance vs multiple strategies")
        elif source_name == 'parsed_data':
            n_periods = len(source_data) if hasattr(source_data, '__len__') else 'N/A'
            lines.append(f"- **Parsed Data**: {n_periods} trading periods")
        else:
            lines.append(f"- **{source_name.replace('_', ' ').title()}**: Available")

    lines.extend([
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
        "*For questions about methodology or results, refer to the technical documentation.*"
    ])

    return lines


# HTML Section Generation Functions

def generate_executive_summary_html(data_sources: Dict, model_tag: str) -> str:
    """Generate executive summary section in HTML."""
    html = """            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="metric-grid">
"""

    # Extract key metrics from statistical validation
    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']
        dataset = sv.get('dataset_info', {})

        total_return = dataset.get('total_strategy_return', 'N/A')
        index_return = dataset.get('total_index_return', 'N/A')
        n_periods = dataset.get('n_periods', 'N/A')

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
        if 'bootstrap_vs_index' in sv:
            bs = sv['bootstrap_vs_index']
            sig_status = "✓ Significant" if bs.get('significant_difference_5pct') else "❌ Not significant"
            p_value = bs.get('p_value_two_sided', 'N/A')
            effect_size = bs.get('effect_size', 'N/A')

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
        if 'out_of_sample_validation' in sv:
            oos = sv['out_of_sample_validation']
            if 'error' not in oos:
                overfitting = "🚨 Detected" if oos.get('overfitting_detection', {}).get('overall_overfitting_detected') else "✅ None detected"
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


def generate_performance_overview_html(data_sources: Dict, model_tag: str) -> str:
    """Generate performance overview section in HTML."""
    html = """            <div class="section">
                <h2>🎯 Performance Overview</h2>
"""

    # Include baseline comparison plot if available
    if 'plots' in data_sources and 'baseline_comparison' in data_sources['plots']:
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['baseline_comparison']}" alt="Baseline Comparison">
                    <div class="chart-caption">Figure 1: Strategy performance vs baseline strategies</div>
                </div>
"""

    # Include equity curves if available
    if 'plots' in data_sources and 'equity_curves' in data_sources['plots']:
        html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['equity_curves']}" alt="Equity Curves">
                    <div class="chart-caption">Figure 2: Equity curves over time</div>
                </div>
"""

    html += "            </div>\n"
    return html


def generate_statistical_validation_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate statistical validation section in HTML."""
    html = """            <div class="section">
                <h2>📊 Statistical Validation</h2>
"""

    if 'statistical_validation' in data_sources:
        # Use the existing print function but capture output
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            print_validation_report(data_sources['statistical_validation'], model_tag)
            validation_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        # Convert the console output to HTML
        html += f"""
                <div class="code-block">{validation_text}</div>
"""

    # Include statistical visualization if available
    if 'statistical_plots' in data_sources and 'statistical_validation' in data_sources['statistical_plots']:
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
    if 'plots' in data_sources:
        if 'calibration' in data_sources['plots']:
            html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration']}" alt="Calibration Plot">
                    <div class="chart-caption">Figure 4: Prediction confidence vs actual performance</div>
                </div>
"""

        if 'calibration_by_decision' in data_sources['plots']:
            html += f"""
                <div class="chart-container">
                    <img src="{data_sources['plots']['calibration_by_decision']}" alt="Calibration by Decision">
                    <div class="chart-caption">Figure 5: Calibration analysis by decision type (BUY/HOLD/SELL)</div>
                </div>
"""

    # Include calibration analysis text
    if 'calibration_analysis' in data_sources:
        cal_text = data_sources['calibration_analysis']
        html += f"""
                <h3>Detailed Calibration Analysis</h3>
                <div class="code-block">{cal_text}</div>
"""

    # Decision patterns
    if 'plots' in data_sources and 'decision_patterns' in data_sources['plots']:
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
    if 'risk_analysis_plots' in data_sources and 'risk_analysis' in data_sources['risk_analysis_plots']:
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

    if 'statistical_validation' in data_sources and 'hold_decision_analysis' in data_sources['statistical_validation']:
        hold_data = data_sources['statistical_validation']['hold_decision_analysis']

        if 'note' in hold_data:
            html += f"""
                <p>{hold_data['note']}</p>
"""
        else:
            # Overall assessment
            overall_rate = hold_data['overall_hold_success_rate']
            category = hold_data['combined_assessment']['performance_category']

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

            quiet = hold_data['quiet_market_success']
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

            context = hold_data['contextual_correctness']
            html += f"""
                <ul class="insights-list">
                    <li><strong>Average Context Score:</strong> {context['avg_context_score']:.2f}</li>
                    <li><strong>Context Success Rate:</strong> {context['context_success_rate']:.1%}</li>
                    <li><strong>{context['interpretation']}</strong></li>
                </ul>
"""

            # Context reasons breakdown
            if context['reason_breakdown']:
                html += """
                <h4>Top Context Reasons:</h4>
                <ul class="insights-list">
"""
                for reason, count in sorted(context['reason_breakdown'].items(), key=lambda x: x[1], reverse=True):
                    reason_name = reason.replace('_', ' ').title()
                    html += f"                    <li>{reason_name}: {count} decisions</li>\n"
                html += "                </ul>\n"

            # Combined assessment
            combined = hold_data['combined_assessment']
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
            stats = hold_data['hold_statistics']
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
            if 'summary' in hold_data:
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
    if 'rolling_performance_plots' in data_sources and 'rolling_performance' in data_sources['rolling_performance_plots']:
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
    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']

        if 'summary_assessment' in sv:
            sa = sv['summary_assessment']

            html += """
                <h3>Key Findings</h3>
                <ul class="insights-list">
"""
            for finding in sa.get('key_findings', []):
                html += f"                    <li>{finding}</li>\n"
            html += "                </ul>\n"

            html += """
                <h3>Recommendations</h3>
                <ul class="insights-list">
"""
            for rec in sa.get('recommendations', []):
                html += f"                    <li>{rec}</li>\n"
            html += "                </ul>\n"

    html += """
                <h3>Overall Assessment</h3>
                <div class="metric-grid">
"""

    if 'statistical_validation' in data_sources:
        sv = data_sources['statistical_validation']
        if 'summary_assessment' in sv:
            sa = sv['summary_assessment']
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
        if source_name == 'plots':
            html += f"""                    <div class="data-source-item">Plots: {len(source_data)} chart files</div>
"""
        elif source_name == 'statistical_validation':
            html += """                    <div class="data-source-item">Statistical Validation: Bootstrap analysis and out-of-sample testing</div>
"""
        elif source_name == 'baseline_comparison':
            html += """                    <div class="data-source-item">Baseline Comparison: Performance vs multiple strategies</div>
"""
        elif source_name == 'parsed_data':
            n_periods = len(source_data) if hasattr(source_data, '__len__') else 'N/A'
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
    import sys
    import argparse

    # Add current directory to path for imports when run as script
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description='Generate comprehensive experiment report')
    parser.add_argument('model_tag', nargs='?', default='dummy_model_memory_only',
                       help='Model identifier (default: dummy_model_memory_only)')
    parser.add_argument('--format', '-f', choices=['markdown', 'html'], default='markdown',
                       help='Output format (default: markdown)')
    parser.add_argument('--no-charts', action='store_true',
                       help='Skip generating additional charts')

    args = parser.parse_args()

    try:
        report_path = generate_comprehensive_report(
            model_tag=args.model_tag,
            include_additional_charts=not args.no_charts,
            output_format=args.format
        )
        print(f"Comprehensive report generated: {report_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
