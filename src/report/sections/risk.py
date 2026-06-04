"""Comprehensive risk analysis section.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


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
            from ...statistical_validation import calculate_risk_attribution

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

        except ImportError:
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
            from ...statistical_validation import calculate_risk_attribution

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

        except ImportError:
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
