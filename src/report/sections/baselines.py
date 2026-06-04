# src/report/sections/baselines.py
"""Baseline strategy suite section: shared compute plus markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

from ...baselines import STRATEGY_METADATA

_BASELINE_CATEGORY_DESCRIPTIONS = {
    "passive": "Market benchmark and passive strategies",
    "trend_following": "Strategies that follow market trends",
    "mean_reversion": "Strategies betting on price return to mean",
    "momentum": "Acceleration and momentum-based strategies",
    "risk_management": "Volatility and risk-based approaches",
    "multi_factor": "Combined factor strategies",
    "confirmation": "Dual-indicator confirmation strategies",
    "noise": "Random and statistical control baselines",
}


def _compute_baseline_strategy_breakdown(comparison_df):
    """Group baseline rows by category; return (category_stats, strategy_details).

    Shared by the markdown and HTML baseline-strategies renderers so the
    aggregation lives in one place. Each renderer formats the result in its
    own way (the twins use divergent format specs).
    """
    category_stats = {}
    strategy_details = []

    for _, row in comparison_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]
            metadata = STRATEGY_METADATA[baseline]

            if category not in category_stats:
                category_stats[category] = {
                    "strategies": [],
                    "best_return": -999,
                    "worst_return": 999,
                    "total_return": 0,
                    "total_sharpe": 0,
                    "total_win_rate": 0,
                    "count": 0,
                }

            stats = category_stats[category]
            stats["strategies"].append(row)
            stats["best_return"] = max(stats["best_return"], row["total_return"])
            stats["worst_return"] = min(stats["worst_return"], row["total_return"])
            stats["total_return"] += row["total_return"]
            stats["total_sharpe"] += row["sharpe_annualized"]
            stats["total_win_rate"] += row["win_rate"]
            stats["count"] += 1

            strategy_details.append(
                {
                    "name": baseline,
                    "category": category,
                    "return": row["total_return"],
                    "sharpe": row["sharpe_annualized"],
                    "win_rate": row["win_rate"],
                    "indicators": (
                        ", ".join(metadata["indicators"])
                        if metadata["indicators"]
                        else "None"
                    ),
                    "description": metadata["description"],
                }
            )

    return category_stats, strategy_details


def generate_baseline_strategies_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate baseline strategies section in HTML."""
    html = """            <div class="section">
                <h2>📈 Enhanced Baseline Strategy Suite</h2>
                <p>Comprehensive analysis of our 15 baseline strategies across 8 categories, providing research-grade comparisons for LLM performance evaluation.</p>
"""

    if "baseline_comparison" not in data_sources:
        html += """                <p><strong>Note:</strong> Baseline comparison data not available for this run.</p>
            </div>
"""
        return html

    comparison_df = data_sources["baseline_comparison"]
    category_stats, strategy_details = _compute_baseline_strategy_breakdown(
        comparison_df
    )

    # Generate category summary table
    if category_stats:
        html += """
                <h3>Strategy Categories Overview</h3>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Strategies</th>
                                <th>Best Return</th>
                                <th>Avg Return</th>
                                <th>Avg Sharpe</th>
                                <th>Avg Win Rate</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        category_descriptions = _BASELINE_CATEGORY_DESCRIPTIONS

        for category, stats in category_stats.items():
            count = stats["count"]
            best_return = stats["best_return"]
            avg_return = stats["total_return"] / count
            avg_sharpe = stats["total_sharpe"] / count
            avg_win_rate = stats["total_win_rate"] / count
            description = category_descriptions.get(
                category, category.replace("_", " ").title()
            )

            return_class = "positive" if avg_return > 0 else "negative"
            win_rate_class = "positive" if avg_win_rate > 0.5 else "neutral"

            html += f"""
                            <tr>
                                <td><strong>{category.replace('_', ' ').title()}</strong></td>
                                <td>{count}</td>
                                <td class="{return_class}">{best_return:+.1f}%</td>
                                <td class="{return_class}">{avg_return:+.1f}%</td>
                                <td>{avg_sharpe:.2f}</td>
                                <td class="{win_rate_class}">{avg_win_rate:.1f}%</td>
                                <td>{description}</td>
                            </tr>
"""

        html += """
                        </tbody>
                    </table>
                </div>
"""

    # Generate individual strategy details table
    if strategy_details:
        html += """
                <h3>Individual Strategy Performance</h3>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Category</th>
                                <th>Total Return</th>
                                <th>Sharpe</th>
                                <th>Win Rate</th>
                                <th>Indicators</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        # Sort by category, then by return
        strategy_details.sort(key=lambda x: (x["category"], -x["return"]))

        for strategy in strategy_details:
            return_class = "positive" if strategy["return"] > 0 else "negative"
            win_rate_class = "positive" if strategy["win_rate"] > 0.5 else "neutral"

            html += f"""
                            <tr>
                                <td><strong>{strategy['name'].replace('_', ' ').title()}</strong></td>
                                <td>{strategy['category'].replace('_', ' ').title()}</td>
                                <td class="{return_class}">{strategy['return']:+.1f}%</td>
                                <td>{strategy['sharpe']:.2f}</td>
                                <td class="{win_rate_class}">{strategy['win_rate']:.1f}%</td>
                                <td>{strategy['indicators']}</td>
                                <td>{strategy['description']}</td>
                            </tr>
"""

        html += """
                        </tbody>
                    </table>
                </div>
"""

    # Add summary statistics
    total_strategies = len(strategy_details)
    categories_count = len(category_stats)

    html += f"""
                <h3>Suite Summary</h3>
                <ul>
                    <li><strong>Total Strategies:</strong> {total_strategies} across {categories_count} categories</li>
                    <li><strong>Categories:</strong> Passive, Trend Following, Mean Reversion, Momentum, Risk Management, Multi-Factor, Confirmation, Noise</li>
                    <li><strong>Indicators Covered:</strong> RSI, MACD, Stochastic, Bollinger Bands, Moving Averages, Volatility</li>
                    <li><strong>Research Purpose:</strong> Statistical control and performance benchmarking for LLM strategies</li>
                </ul>
            </div>
"""

    return html


def generate_baseline_strategies_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate comprehensive baseline strategies showcase with categorization."""
    lines = [
        "## 📈 Enhanced Baseline Strategy Suite",
        "",
        "Comprehensive analysis of our 15 baseline strategies across 8 categories,",
        "providing research-grade comparisons for LLM performance evaluation.",
        "",
    ]

    if "baseline_comparison" not in data_sources:
        lines.extend(
            [
                "**Note:** Baseline comparison data not available for this run.",
                "",
            ]
        )
        return lines

    comparison_df = data_sources["baseline_comparison"]
    category_stats, strategy_details = _compute_baseline_strategy_breakdown(
        comparison_df
    )

    # Generate category summary table
    if category_stats:
        lines.extend(
            [
                "### Strategy Categories Overview",
                "",
                "| Category | Strategies | Best Return | Avg Return | Avg Sharpe | Avg Win Rate | Description |",
                "|----------|------------|-------------|------------|------------|--------------|-------------|",
            ]
        )

        category_descriptions = _BASELINE_CATEGORY_DESCRIPTIONS

        for category, stats in category_stats.items():
            count = stats["count"]
            best_return = stats["best_return"]
            avg_return = stats["total_return"] / count
            avg_sharpe = stats["total_sharpe"] / count
            avg_win_rate = stats["total_win_rate"] / count
            description = category_descriptions.get(
                category, category.replace("_", " ").title()
            )

            lines.append(
                f"| {category.replace('_', ' ').title()} | {count} | {best_return:+.1f}% | {avg_return:+.1f}% | {avg_sharpe:.2f} | {avg_win_rate:.1%} | {description} |"
            )

        lines.extend(
            [
                "",
            ]
        )

    # Generate individual strategy details table
    if strategy_details:
        lines.extend(
            [
                "### Individual Strategy Performance",
                "",
                "| Strategy | Category | Total Return | Sharpe | Win Rate | Indicators | Description |",
                "|----------|----------|--------------|--------|-----------|------------|-------------|",
            ]
        )

        # Sort by category, then by return
        strategy_details.sort(key=lambda x: (x["category"], -x["return"]))

        for strategy in strategy_details:
            lines.append(
                f"| {strategy['name'].replace('_', ' ').title()} | {strategy['category'].replace('_', ' ').title()} | {strategy['return']:+.1f}% | {strategy['sharpe']:.2f} | {strategy['win_rate']:.1%} | {strategy['indicators']} | {strategy['description']} |"
            )

        lines.extend(
            [
                "",
            ]
        )

    # Add summary statistics
    total_strategies = len(strategy_details)
    categories_count = len(category_stats)

    lines.extend(
        [
            "### Suite Summary",
            f"- **Total Strategies**: {total_strategies} across {categories_count} categories",
            "- **Categories**: Passive, Trend Following, Mean Reversion, Momentum, Risk Management, Multi-Factor, Confirmation, Noise",
            "- **Indicators Covered**: RSI, MACD, Stochastic, Bollinger Bands, Moving Averages, Volatility",
            "- **Research Purpose**: Statistical control and performance benchmarking for LLM strategies",
            "",
            "---",
            "",
        ]
    )

    return lines
