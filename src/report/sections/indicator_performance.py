"""Indicator-specific performance section.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


def generate_indicator_performance_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate indicator-specific performance analysis section."""
    lines = [
        "## 🎯 Indicator-Specific Performance Analysis",
        "",
        "This analysis examines LLM performance when technical indicators give specific signals,",
        "revealing which market conditions the model handles most effectively.",
        "",
    ]

    # Indicator performance plot
    if (
        "indicator_performance_plots" in data_sources
        and "indicator_performance" in data_sources["indicator_performance_plots"]
    ):
        lines.extend(
            [
                f"![Indicator Performance Analysis]({data_sources['indicator_performance_plots']['indicator_performance']})",
                "*Figure: LLM returns, win rates, and performance differentials when indicators signal vs no signal*",
                "",
            ]
        )

        # Add key insights
        lines.extend(
            [
                "### Key Insights:",
                "",
            ]
        )

        # Extract indicator performance insights
        if "parsed_data" in data_sources and "features_data" in data_sources:
            from ...decision_analysis import analyze_indicator_specific_performance

            indicator_perf = analyze_indicator_specific_performance(
                data_sources["parsed_data"], data_sources["features_data"]
            )

            if indicator_perf:
                # Find indicators where LLM performs significantly better/worse
                insights = []
                for indicator, data in indicator_perf.items():
                    if "error" not in data:
                        perf_diff = data.get("performance_differential", 0)
                        if abs(perf_diff) > 0.02:  # More than 2% difference
                            direction = "better" if perf_diff > 0 else "worse"
                            insights.append(
                                {
                                    "indicator": indicator,
                                    "performance": perf_diff,
                                    "direction": direction,
                                }
                            )

                if insights:
                    # Sort by performance differential
                    insights.sort(key=lambda x: x["performance"], reverse=True)

                    for insight in insights[:3]:  # Top 3 insights
                        perf_pct = insight["performance"] * 100
                        lines.append(
                            f"- **{insight['indicator']}**: LLM performs {insight['direction']} by "
                            f"{abs(perf_pct):+.1f}% when indicator signals"
                        )

                    lines.append("")

                    # Market condition analysis
                    best_conditions = [
                        i for i in insights if i["direction"] == "better"
                    ]
                    if best_conditions:
                        best_indicator = best_conditions[0]["indicator"]
                        lines.append(
                            f"- **Optimal Conditions**: LLM excels most in {best_indicator} signal environments"
                        )

                    lines.append("")

    lines.extend(["---", ""])

    return lines


def generate_indicator_performance_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate indicator-specific performance analysis section in HTML."""
    html = """
    <section class="analysis-section">
        <div class="section-header">
            <h2>🎯 Indicator-Specific Performance Analysis</h2>
            <p>This analysis examines LLM performance when technical indicators give specific signals,
            revealing which market conditions the model handles most effectively.</p>
        </div>
    """

    # Indicator performance plot
    if (
        "indicator_performance_plots" in data_sources
        and "indicator_performance" in data_sources["indicator_performance_plots"]
    ):
        plot_path = data_sources["indicator_performance_plots"]["indicator_performance"]
        html += f"""
        <div class="chart-container">
            <img src="{plot_path}" alt="Indicator Performance Analysis" class="responsive-chart">
            <p class="chart-caption">Figure: LLM returns, win rates, and performance differentials when indicators signal vs no signal</p>
        </div>
        """

        # Add insights section
        if "parsed_data" in data_sources and "features_data" in data_sources:
            from ...decision_analysis import analyze_indicator_specific_performance

            indicator_perf = analyze_indicator_specific_performance(
                data_sources["parsed_data"], data_sources["features_data"]
            )

            if indicator_perf:
                # Find indicators where LLM performs significantly better/worse
                insights = []
                for indicator, data in indicator_perf.items():
                    if "error" not in data:
                        perf_diff = data.get("performance_differential", 0)
                        if abs(perf_diff) > 0.02:  # More than 2% difference
                            direction = "better" if perf_diff > 0 else "worse"
                            insights.append(
                                {
                                    "indicator": indicator,
                                    "performance": perf_diff,
                                    "direction": direction,
                                }
                            )

                if insights:
                    # Sort by performance differential
                    insights.sort(key=lambda x: x["performance"], reverse=True)

                    html += '<div class="insights-grid">'

                    for i, insight in enumerate(insights[:3]):  # Top 3 insights
                        perf_pct = insight["performance"] * 100
                        card_class = (
                            "positive"
                            if insight["direction"] == "better"
                            else "negative"
                        )

                        html += f"""
        <div class="insight-card {card_class}">
            <h4>{insight['indicator']} Signals</h4>
            <p class="metric">{abs(perf_pct):+.1f}%</p>
            <p class="value">LLM performs {insight['direction']} when indicator signals</p>
        </div>
                        """

                    # Market condition analysis
                    best_conditions = [
                        i for i in insights if i["direction"] == "better"
                    ]
                    if best_conditions:
                        best_indicator = best_conditions[0]["indicator"]
                        html += f"""
        <div class="insight-card neutral">
            <h4>🎯 Optimal Conditions</h4>
            <p>LLM excels most in {best_indicator} signal environments</p>
        </div>
                        """

                    html += "</div>"

    html += "    </section>"
    return html
