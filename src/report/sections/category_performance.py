"""Strategy category performance section.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

from ...baselines import calculate_category_performance


def _compute_category_best_worst(data_sources):
    """Return (best_category, worst_category) rows by LLM-vs-category delta.

    Returns None when baseline data is missing or category stats are empty.
    Shared by the markdown and HTML category-performance renderers.
    """
    if "baseline_comparison" not in data_sources:
        return None

    category_stats = calculate_category_performance(data_sources["baseline_comparison"])
    if category_stats.empty:
        return None

    llm_return = data_sources.get("llm_metrics", {}).get("total_return", 0)
    category_stats["llm_vs_category"] = llm_return - category_stats["avg_return"]
    best_category = category_stats.loc[category_stats["llm_vs_category"].idxmax()]
    worst_category = category_stats.loc[category_stats["llm_vs_category"].idxmin()]
    return best_category, worst_category


def generate_category_performance_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """Generate strategy category performance analysis section."""
    lines = [
        "## 📊 Strategy Category Performance Analysis",
        "",
        "This analysis groups the 15 baseline strategies by their trading approach categories,",
        "allowing for a systematic comparison of LLM performance across different trading styles.",
        "",
    ]

    # Category performance plot
    if (
        "category_performance_plots" in data_sources
        and "category_performance" in data_sources["category_performance_plots"]
    ):
        lines.extend(
            [
                f"![Strategy Category Performance]({data_sources['category_performance_plots']['category_performance']})",
                "*Figure: LLM performance vs strategy category averages across returns, Sharpe ratios, and win rates*",
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

        # Extract category performance insights
        best_worst = _compute_category_best_worst(data_sources)
        if best_worst is not None:
            best_category, worst_category = best_worst

            lines.extend(
                [
                    f"- **Best Category Match**: {best_category['category'].replace('_', ' ').title()} ",
                    f"(LLM outperforms by {best_category['llm_vs_category']:+.1f}%)",
                    f"- **Most Challenging**: {worst_category['category'].replace('_', ' ').title()} ",
                    f"(LLM underperforms by {abs(worst_category['llm_vs_category']):+.1f}%)",
                    f"- **Strategic Positioning**: LLM shows relative strength in {best_category['category'].replace('_', ' ')} strategies",
                    "",
                ]
            )

    lines.extend(["---", ""])

    return lines


def generate_category_performance_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate strategy category performance analysis section in HTML."""
    html = """
    <section class="analysis-section">
        <div class="section-header">
            <h2>📊 Strategy Category Performance Analysis</h2>
            <p>This analysis groups the 15 baseline strategies by their trading approach categories,
            allowing for a systematic comparison of LLM performance across different trading styles.</p>
        </div>
    """

    # Category performance plot
    if (
        "category_performance_plots" in data_sources
        and "category_performance" in data_sources["category_performance_plots"]
    ):
        plot_path = data_sources["category_performance_plots"]["category_performance"]
        html += f"""
        <div class="chart-container">
            <img src="{plot_path}" alt="Strategy Category Performance" class="responsive-chart">
            <p class="chart-caption">Figure: LLM performance vs strategy category averages across returns, Sharpe ratios, and win rates</p>
        </div>
        """

        # Add insights section
        best_worst = _compute_category_best_worst(data_sources)
        if best_worst is not None:
            best_category, worst_category = best_worst

            html += f"""
        <div class="insights-grid">
            <div class="insight-card positive">
                <h4>🏆 Best Category Match</h4>
                <p class="metric">{best_category['category'].replace('_', ' ').title()}</p>
                <p class="value">+{best_category['llm_vs_category']:+.1f}% vs category average</p>
            </div>
            <div class="insight-card negative">
                <h4>⚠️ Most Challenging</h4>
                <p class="metric">{worst_category['category'].replace('_', ' ').title()}</p>
                <p class="value">{worst_category['llm_vs_category']:+.1f}% vs category average</p>
            </div>
            <div class="insight-card neutral">
                <h4>🎯 Strategic Positioning</h4>
                <p>LLM shows relative strength in {best_category['category'].replace('_', ' ')} strategies</p>
            </div>
        </div>
        """

    html += "    </section>"
    return html
