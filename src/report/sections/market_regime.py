# src/report/sections/market_regime.py
"""Market regime analysis section: shared regime compute plus markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from ...constants import TRADING_DAYS_PER_YEAR, VOL_WINDOW


def _compute_regime_performance(parsed_df):
    """Classify days into volatility regimes and aggregate per-regime stats.

    Returns a list of regime performance dicts (empty when the required
    columns are absent or no regime has data). Shared by the markdown and HTML
    market-regime renderers.
    """
    if not (
        "next_return_1d" in parsed_df.columns and "strategy_return" in parsed_df.columns
    ):
        return []

    # Calculate rolling volatility (20-day window)
    market_returns = parsed_df["next_return_1d"]
    rolling_vol = market_returns.rolling(VOL_WINDOW).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )  # Annualized

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
                regime_data["strategy_return"].mean() * TRADING_DAYS_PER_YEAR
            )  # Annualized
            market_return = regime_data["next_return_1d"].mean() * TRADING_DAYS_PER_YEAR
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

    return regime_performance


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

    regime_performance = _compute_regime_performance(parsed_df)
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

    regime_performance = _compute_regime_performance(parsed_df)
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
