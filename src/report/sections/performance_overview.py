# src/report/sections/performance_overview.py
"""Performance overview section: markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


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
