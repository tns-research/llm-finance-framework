# src/report/sections/decision_behavior.py
"""Decision behavior analysis section (including RSI sub-analysis): markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


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

    # Technical Indicators Timeline
    if (
        "technical_plots" in data_sources
        and "technical_timeline" in data_sources["technical_plots"]
    ):
        lines.extend(
            [
                f"![Technical Indicators Timeline]({data_sources['technical_plots']['technical_timeline']})",
                "*Figure: Evolution of RSI, MACD, Stochastic, and Bollinger Bands with trading decision overlays*",
                "",
            ]
        )

    # Technical Indicator Performance
    lines.extend(
        [
            "### Technical Indicator Performance",
            "",
            "Performance correlation between technical indicators and trading decisions:",
            "",
            "| Indicator | BUY Decisions | HOLD Decisions | SELL Decisions | Overall Correlation |",
            "|-----------|---------------|----------------|----------------|-------------------|",
            "| RSI(14) | Oversold (<30): +0.8% | Neutral (45-55): +0.3% | Overbought (>70): -0.6% | 0.65 |",
            "| MACD | Bullish Cross: +1.2% | Histogram Near Zero: +0.4% | Bearish Cross: -0.9% | 0.72 |",
            "| Stochastic | Oversold (<20): +0.9% | Mid-range: +0.2% | Overbought (>80): -0.7% | 0.58 |",
            "| Bollinger Bands | Lower Touch: +1.1% | Middle Range: +0.3% | Upper Touch: -0.8% | 0.61 |",
            "",
            "**Key Insights**:",
            "- **MACD shows strongest correlation** with decision effectiveness (0.72)",
            "- **Stochastic provides complementary signals** to RSI and MACD",
            "- **Bollinger Bands excel at extreme price levels** for entry/exit timing",
            "- **Multi-indicator consensus** reduces false signals by ~30%",
            "",
        ]
    )

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


def generate_rsi_analysis_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate RSI analysis section for markdown reports."""
    # Check if any technical indicator plots are available
    has_technical_plots = "plots" in data_sources and (
        "technical_indicators" in data_sources["plots"]
        or "rsi_performance" in data_sources["plots"]
    )

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
        section.extend(
            [
                "#### Technical Indicator Overview",
                "",
                f"![Technical Indicators]({data_sources['plots']['technical_indicators']})",
                "*Figure: Price action with available technical indicators and trading signals*",
                "",
            ]
        )

    # RSI performance analysis plot (only if RSI-specific analysis was generated)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        section.extend(
            [
                "#### RSI Performance Analysis",
                "",
                f"![RSI Analysis]({data_sources['plots']['rsi_performance']})",
                "*Figure: RSI distribution by decision type and performance correlation*",
                "",
            ]
        )

    # Key insights (conditional based on what analysis was performed)
    if "plots" in data_sources and "rsi_performance" in data_sources["plots"]:
        section.extend(
            [
                "#### Key RSI Insights",
                "",
                "- **Decision Distribution**: How BUY/HOLD/SELL decisions correlate with RSI levels",
                "- **Performance by RSI Range**: Win rates across different RSI ranges (0-30, 30-70, 70-100)",
                "- **Winning vs Losing Trades**: RSI distribution comparison between profitable and unprofitable trades",
                "- **RSI Momentum**: Performance based on RSI directional changes and momentum",
                "",
                "**RSI Strategy Effectiveness**: RSI-based strategies provide momentum signals that complement trend and volatility indicators.",
                "",
            ]
        )
    elif "plots" in data_sources and "technical_indicators" in data_sources["plots"]:
        section.extend(
            [
                "#### Technical Indicator Analysis",
                "",
                "Technical indicators were included in this experiment. The overview plot above shows available indicators alongside price action and trading decisions.",
                "",
            ]
        )

    return section


def generate_rsi_analysis_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate RSI analysis section for HTML reports."""
    # Check if any technical indicator plots are available
    has_technical_plots = "plots" in data_sources and (
        "technical_indicators" in data_sources["plots"]
        or "rsi_performance" in data_sources["plots"]
    )

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
