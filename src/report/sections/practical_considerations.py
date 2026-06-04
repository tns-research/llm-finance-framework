"""Practical implementation considerations section.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

from ...constants import TRADING_DAYS_PER_YEAR


def _compute_transaction_costs(parsed_df):
    """Estimate trading frequency and annualized cost from decision changes.

    Returns None when the decision column is absent. Shared by the markdown
    and HTML practical-considerations renderers.
    """
    if "decision" not in parsed_df.columns:
        return None

    decisions = parsed_df["decision"]

    # Count position changes (simplified trade detection)
    position_changes = 0
    prev_decision = None
    for decision in decisions:
        if prev_decision is not None and decision != prev_decision:
            position_changes += 1
        prev_decision = decision

    trading_frequency = position_changes / len(parsed_df) * 100
    annual_trades = position_changes * (
        TRADING_DAYS_PER_YEAR / len(parsed_df)
    )  # Approximate

    # Estimate costs (rough assumptions)
    avg_commission = 0.001  # 0.1% per trade
    avg_spread = 0.0005  # 0.05% spread cost
    total_cost_per_trade = avg_commission + avg_spread
    annual_cost_bps = annual_trades * total_cost_per_trade * 10000  # Convert to bps

    cost_impact = (
        "Significant"
        if annual_cost_bps > 50
        else "Moderate" if annual_cost_bps > 20 else "Minimal"
    )

    return {
        "trading_frequency": trading_frequency,
        "annual_trades": annual_trades,
        "annual_cost_bps": annual_cost_bps,
        "cost_impact": cost_impact,
    }


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

        costs = _compute_transaction_costs(parsed_df)
        if costs is not None:
            lines.extend(
                [
                    "### Transaction Costs Impact",
                    "",
                    f"- **Trading Frequency**: {costs['trading_frequency']:.1f}% of days involve position changes",
                    f"- **Estimated Annual Trades**: {costs['annual_trades']:.0f} round trips",
                    f"- **Estimated Trading Costs**: {costs['annual_cost_bps']:.0f} basis points annually",
                    f"- **Cost Impact**: {costs['cost_impact']} impact on performance",
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


def generate_practical_considerations_html(data_sources: Dict, model_tag: str) -> str:
    """Generate practical implementation considerations section in HTML."""
    html = """            <div class="section">
                <h2>🛠️ Practical Implementation Considerations</h2>
                <p>Real-world deployment requires addressing transaction costs, liquidity, and operational factors.</p>
"""

    # Transaction costs analysis
    if "parsed_data" in data_sources:
        parsed_df = data_sources["parsed_data"]

        costs = _compute_transaction_costs(parsed_df)
        if costs is not None:
            html += f"""
                <h3>Transaction Costs Impact</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Trading Frequency</div>
                        <div class="value">{costs['trading_frequency']:.1f}%</div>
                        <small>Days with position changes</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Annual Trades</div>
                        <div class="value">{costs['annual_trades']:.0f}</div>
                        <small>Round trip trades</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Trading Costs</div>
                        <div class="value">{costs['annual_cost_bps']:.0f} bps</div>
                        <small>{costs['cost_impact']} impact</small>
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
