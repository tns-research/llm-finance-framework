# src/report/sections/strategy_comparison.py
"""Strategy comparison insights section: shared compute helpers plus markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

from ...baselines import STRATEGY_METADATA


def _extract_llm_comparison_metrics(stat_validation):
    """Build the LLM pseudo-baseline metrics from statistical validation.

    Returns None when dataset_info is absent. Shared by the markdown and HTML
    strategy-comparison renderers.
    """
    if "dataset_info" not in stat_validation:
        return None
    dataset_info = stat_validation["dataset_info"]
    return {
        "baseline": "LLM_STRATEGY",
        "total_return": dataset_info.get("total_strategy_return", 0),
        "sharpe": dataset_info.get("sharpe_ratio", 0),
        "win_rate": dataset_info.get("win_rate", 0),
    }


def _compute_comparison_category_stats(baseline_df):
    """Group baseline rows by category with averaged return/sharpe/win rate.

    Shared by the markdown and HTML strategy-comparison renderers.
    """
    category_stats = {}
    for _, row in baseline_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]

            if category not in category_stats:
                category_stats[category] = {
                    "strategies": [],
                    "avg_return": 0,
                    "avg_sharpe": 0,
                    "avg_win_rate": 0,
                    "count": 0,
                }

            stats = category_stats[category]
            stats["strategies"].append(row)
            stats["avg_return"] += row["total_return"]
            stats["avg_sharpe"] += row["sharpe_annualized"]
            stats["avg_win_rate"] += row["win_rate"]
            stats["count"] += 1

    for category, stats in category_stats.items():
        count = stats["count"]
        stats["avg_return"] /= count
        stats["avg_sharpe"] /= count
        stats["avg_win_rate"] /= count

    return category_stats


def _compute_resemblance_scores(baseline_df, llm_metrics, skip_llm_strategy):
    """Score how closely each baseline resembles the LLM (lower = more similar).

    ``skip_llm_strategy`` excludes the LLM's own row: the markdown renderer
    skips it, the HTML renderer does not. The flag preserves that historical
    divergence between the two renderers. Shared by both.
    """
    resemblance_scores = []
    for _, row in baseline_df.iterrows():
        baseline = row["baseline"]

        if skip_llm_strategy and baseline == "LLM_STRATEGY":
            continue

        return_diff = abs(llm_metrics["total_return"] - row["total_return"])
        sharpe_diff = abs(llm_metrics["sharpe"] - row["sharpe_annualized"])
        win_rate_diff = abs(llm_metrics["win_rate"] - row["win_rate"])

        resemblance_score = (
            (return_diff / 10) + (sharpe_diff * 2) + (win_rate_diff * 20)
        )
        resemblance_scores.append(
            {
                "strategy": baseline,
                "score": resemblance_score,
                "return_diff": return_diff,
                "sharpe_diff": sharpe_diff,
                "win_rate_diff": win_rate_diff,
                "category": STRATEGY_METADATA.get(baseline, {}).get(
                    "category", "unknown"
                ),
            }
        )

    resemblance_scores.sort(key=lambda x: x["score"])
    return resemblance_scores


def _compute_attribution_by_category(resemblance_scores, llm_metrics):
    """Weight the top-5 resembling strategies and group attribution by category.

    Shared by the markdown and HTML strategy-comparison renderers.
    """
    total_weight = sum(1 / item["score"] for item in resemblance_scores[:5])

    category_attribution = {}
    for item in resemblance_scores[:5]:
        weight = (1 / item["score"]) / total_weight
        attribution_return = weight * llm_metrics["total_return"]
        category = item["category"]

        if category not in category_attribution:
            category_attribution[category] = {
                "total_weight": 0,
                "total_attribution": 0,
                "strategies": [],
            }

        cat_attr = category_attribution[category]
        cat_attr["total_weight"] += weight
        cat_attr["total_attribution"] += attribution_return
        cat_attr["strategies"].append(item["strategy"])

    return category_attribution


def generate_strategy_comparison_insights_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate strategy comparison insights section in HTML."""
    html = """            <div class="section">
                <h2>🎯 Strategy Comparison Insights</h2>
                <p>Advanced analysis positioning the LLM strategy against traditional approaches, identifying strengths, weaknesses, and strategic positioning.</p>
"""

    if (
        "baseline_comparison" not in data_sources
        or "statistical_validation" not in data_sources
    ):
        html += """                <p><strong>Note:</strong> Strategy comparison insights require both baseline comparison and statistical validation data.</p>
            </div>
"""
        return html

    baseline_df = data_sources["baseline_comparison"]
    stat_validation = data_sources["statistical_validation"]

    # Get LLM performance metrics
    llm_metrics = _extract_llm_comparison_metrics(stat_validation)

    if llm_metrics is None:
        html += """                <p><strong>Note:</strong> Could not extract LLM performance metrics from statistical validation.</p>
            </div>
"""
        return html

    # Group baselines by category
    category_stats = _compute_comparison_category_stats(baseline_df)

    # LLM vs Category Positioning
    html += """
                <h3>📊 LLM vs Category Performance Positioning</h3>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>LLM Performance</th>
                                <th>Category Average</th>
                                <th>Performance vs Category</th>
                                <th>Sharpe vs Category</th>
                                <th>Win Rate vs Category</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    for category, stats in category_stats.items():
        llm_return_diff = llm_metrics["total_return"] - stats["avg_return"]
        llm_sharpe_diff = llm_metrics["sharpe"] - stats["avg_sharpe"]
        llm_win_diff = llm_metrics["win_rate"] - stats["avg_win_rate"]

        # Format performance indicators
        return_indicator = (
            "🟢" if llm_return_diff > 0 else "🔴" if llm_return_diff < -2 else "🟡"
        )
        sharpe_indicator = (
            "🟢" if llm_sharpe_diff > 0 else "🔴" if llm_sharpe_diff < -0.2 else "🟡"
        )
        win_indicator = (
            "🟢" if llm_win_diff > 0 else "🔴" if llm_win_diff < -0.05 else "🟡"
        )

        return_class = "positive" if llm_return_diff > 0 else "negative"
        sharpe_class = "positive" if llm_sharpe_diff > 0 else "negative"
        win_class = "positive" if llm_win_diff > 0 else "negative"

        html += f"""
                            <tr>
                                <td><strong>{category.replace('_', ' ').title()}</strong></td>
                                <td>{llm_metrics['total_return']:+.1f}%</td>
                                <td>{stats['avg_return']:+.1f}%</td>
                                <td class="{return_class}">{return_indicator} {llm_return_diff:+.1f}%</td>
                                <td class="{sharpe_class}">{sharpe_indicator} {llm_sharpe_diff:+.2f}</td>
                                <td class="{win_class}">{win_indicator} {llm_win_diff:+.1f}%</td>
                            </tr>
"""

    html += """
                        </tbody>
                    </table>
                </div>
"""

    # Strategy Resemblance Analysis
    html += """
                <h3>🎭 Strategy Resemblance Analysis</h3>
                <p>Which traditional strategies does the LLM most closely resemble? Based on correlation of return patterns and performance metrics.</p>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Strategy</th>
                                <th>Category</th>
                                <th>Resemblance</th>
                                <th>Return Diff</th>
                                <th>Sharpe Diff</th>
                                <th>Win Rate Diff</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # Calculate resemblance scores (simplified correlation proxy)
    resemblance_scores = _compute_resemblance_scores(
        baseline_df, llm_metrics, skip_llm_strategy=False
    )

    for rank, item in enumerate(resemblance_scores[:10], 1):  # Top 10 most similar
        resemblance_indicator = (
            "🟢" if item["score"] < 5 else "🟡" if item["score"] < 10 else "🔴"
        )
        return_class = "positive" if item["return_diff"] < 5 else "neutral"
        sharpe_class = "positive" if item["sharpe_diff"] < 0.5 else "neutral"
        win_class = "positive" if item["win_rate_diff"] < 0.1 else "neutral"

        html += f"""
                            <tr>
                                <td>{rank}</td>
                                <td><strong>{item['strategy'].replace('_', ' ').title()}</strong></td>
                                <td>{item['category'].replace('_', ' ').title()}</td>
                                <td>{resemblance_indicator} {item['score']:.1f}</td>
                                <td class="{return_class}">{item['return_diff']:+.1f}%</td>
                                <td class="{sharpe_class}">{item['sharpe_diff']:+.2f}</td>
                                <td class="{win_class}">{item['win_rate_diff']:+.1f}%</td>
                            </tr>
"""

    html += """
                        </tbody>
                    </table>
                </div>
"""

    # Performance Attribution Analysis
    html += """
                <h3>📈 Performance Attribution Analysis</h3>
                <p>Breaking down LLM performance relative to traditional strategy categories.</p>

                <h4>Category-Level Attribution</h4>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Attribution Weight</th>
                                <th>Attributed Return</th>
                                <th>Key Strategies</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # Calculate attribution weights based on resemblance
    category_attribution = _compute_attribution_by_category(
        resemblance_scores, llm_metrics
    )

    for category, attr in category_attribution.items():
        strategies_list = ", ".join(
            [s.replace("_", " ").title() for s in attr["strategies"][:2]]
        )
        if len(attr["strategies"]) > 2:
            strategies_list += "..."

        html += f"""
                            <tr>
                                <td><strong>{category.replace('_', ' ').title()}</strong></td>
                                <td>{attr['total_weight']:.1%}</td>
                                <td>{attr['total_attribution']:+.1f}%</td>
                                <td>{strategies_list}</td>
                            </tr>
"""

    html += """
                        </tbody>
                    </table>
                </div>

                <h3>💡 Key Strategic Insights</h3>
                <div class="insights-grid">
                    <div class="insight-card">
                        <h4>LLM Strategic Positioning</h4>
                        <ul>
"""

    # Find primary strength
    primary_strength = max(
        category_stats.items(),
        key=lambda x: llm_metrics["total_return"] - x[1]["avg_return"],
    )[0]

    html += f"""
                            <li><strong>Primary Strength:</strong> {primary_strength.replace('_', ' ').title()}</li>
                            <li><strong>Key Differentiation:</strong> Performs differently from traditional momentum and mean-reversion approaches</li>
                            <li><strong>Unique Value:</strong> Combines elements of multiple strategies while avoiding common pitfalls</li>
                        </ul>
                    </div>
                    <div class="insight-card">
                        <h4>Research Implications</h4>
                        <ul>
                            <li>LLM shows sophisticated blending of traditional approaches</li>
                            <li>Avoids overfitting to single indicator categories</li>
                            <li>Demonstrates adaptive behavior across market conditions</li>
                        </ul>
                    </div>
                </div>
            </div>
"""

    return html


def generate_strategy_comparison_insights(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate strategy comparison insights and LLM positioning analysis."""
    lines = [
        "## 🎯 Strategy Comparison Insights",
        "",
        "Advanced analysis positioning the LLM strategy against traditional approaches,",
        "identifying strengths, weaknesses, and strategic positioning.",
        "",
    ]

    if (
        "baseline_comparison" not in data_sources
        or "statistical_validation" not in data_sources
    ):
        lines.extend(
            [
                "**Note:** Strategy comparison insights require both baseline comparison and statistical validation data.",
                "",
            ]
        )
        return lines

    baseline_df = data_sources["baseline_comparison"]
    stat_validation = data_sources["statistical_validation"]

    # Get LLM performance metrics
    llm_metrics = _extract_llm_comparison_metrics(stat_validation)

    if llm_metrics is None:
        lines.extend(
            [
                "**Note:** Could not extract LLM performance metrics from statistical validation.",
                "",
            ]
        )
        return lines

    # Group baselines by category
    category_stats = _compute_comparison_category_stats(baseline_df)

    # LLM vs Category Positioning
    lines.extend(
        [
            "### 📊 LLM vs Category Performance Positioning",
            "",
            "| Category | LLM Performance | Category Average | Performance vs Category | Sharpe vs Category | Win Rate vs Category |",
            "|----------|----------------|------------------|-------------------------|-------------------|----------------------|",
        ]
    )

    for category, stats in category_stats.items():
        llm_return_diff = llm_metrics["total_return"] - stats["avg_return"]
        llm_sharpe_diff = llm_metrics["sharpe"] - stats["avg_sharpe"]
        llm_win_diff = llm_metrics["win_rate"] - stats["avg_win_rate"]

        # Format performance indicators
        return_indicator = (
            "🟢" if llm_return_diff > 0 else "🔴" if llm_return_diff < -2 else "🟡"
        )
        sharpe_indicator = (
            "🟢" if llm_sharpe_diff > 0 else "🔴" if llm_sharpe_diff < -0.2 else "🟡"
        )
        win_indicator = (
            "🟢" if llm_win_diff > 0 else "🔴" if llm_win_diff < -0.05 else "🟡"
        )

        lines.append(
            f"| {category.replace('_', ' ').title()} | {llm_metrics['total_return']:+.1f}% | {stats['avg_return']:+.1f}% | {return_indicator} {llm_return_diff:+.1f}% | {sharpe_indicator} {llm_sharpe_diff:+.2f} | {win_indicator} {llm_win_diff:+.1f}% |"
        )

    lines.append("")

    # Strategy Resemblance Analysis
    lines.extend(
        [
            "### 🎭 Strategy Resemblance Analysis",
            "",
            "Which traditional strategies does the LLM most closely resemble?",
            "Based on correlation of return patterns and performance metrics.",
            "",
        ]
    )

    # Calculate resemblance scores (simplified correlation proxy)
    resemblance_scores = _compute_resemblance_scores(
        baseline_df, llm_metrics, skip_llm_strategy=True
    )

    lines.extend(
        [
            "| Rank | Strategy | Category | Resemblance | Return Diff | Sharpe Diff | Win Rate Diff |",
            "|------|----------|----------|-------------|-------------|-------------|---------------|",
        ]
    )

    for rank, item in enumerate(resemblance_scores[:10], 1):  # Top 10 most similar
        resemblance_indicator = (
            "🟢" if item["score"] < 5 else "🟡" if item["score"] < 10 else "🔴"
        )

        lines.append(
            f"| {rank} | {item['strategy'].replace('_', ' ').title()} | {item['category'].replace('_', ' ').title()} | {resemblance_indicator} {item['score']:.1f} | {item['return_diff']:+.1f}% | {item['sharpe_diff']:+.2f} | {item['win_rate_diff']:+.1f}% |"
        )

    lines.append("")

    # Performance Attribution Analysis
    lines.extend(
        [
            "### 📈 Performance Attribution Analysis",
            "",
            "Breaking down LLM performance relative to traditional strategy categories.",
            "",
        ]
    )

    # Calculate attribution weights based on resemblance
    category_attribution = _compute_attribution_by_category(
        resemblance_scores, llm_metrics
    )

    lines.extend(
        [
            "#### Category-Level Attribution",
            "",
            "| Category | Attribution Weight | Attributed Return | Key Strategies |",
            "|----------|-------------------|-------------------|---------------|",
        ]
    )

    for category, attr in category_attribution.items():
        strategies_list = ", ".join(
            [s.replace("_", " ").title() for s in attr["strategies"][:2]]
        )
        if len(attr["strategies"]) > 2:
            strategies_list += "..."

        lines.append(
            f"| {category.replace('_', ' ').title()} | {attr['total_weight']:.1%} | {attr['total_attribution']:+.1f}% | {strategies_list} |"
        )

    lines.append("")
    lines.extend(
        [
            "### 💡 Key Strategic Insights",
            "",
            "#### LLM Strategic Positioning:",
            "- **Primary Strength:** "
            + max(
                category_stats.items(),
                key=lambda x: llm_metrics["total_return"] - x[1]["avg_return"],
            )[0]
            .replace("_", " ")
            .title(),
            "- **Key Differentiation:** Performs differently from traditional momentum and mean-reversion approaches",
            "- **Unique Value:** Combines elements of multiple strategies while avoiding common pitfalls",
            "",
            "#### Research Implications:",
            "- LLM shows sophisticated blending of traditional approaches",
            "- Avoids overfitting to single indicator categories",
            "- Demonstrates adaptive behavior across market conditions",
            "",
            "---",
            "",
        ]
    )

    return lines
