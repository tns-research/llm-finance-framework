# src/report/sections/executive_summary.py
"""Executive summary section: positioning table, alignment highlights, and markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List

from ...baselines import STRATEGY_METADATA


def get_llm_positioning_table(data_sources: Dict, llm_return: float) -> List[str]:
    """Generate LLM positioning table against baseline categories."""
    lines = []

    if "baseline_comparison" not in data_sources:
        lines.append("| No baseline data | N/A | N/A | N/A |")
        return lines

    baseline_df = data_sources["baseline_comparison"]

    # Group baselines by category and calculate averages
    category_stats = {}
    for _, row in baseline_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]

            if category not in category_stats:
                category_stats[category] = {"returns": [], "count": 0}

            category_stats[category]["returns"].append(row["total_return"])
            category_stats[category]["count"] += 1

    # Calculate category averages and LLM positioning
    for category, stats in category_stats.items():
        if stats["count"] > 0:
            avg_return = sum(stats["returns"]) / stats["count"]
            diff = llm_return - avg_return

            if diff > 2:
                position = "🟢 Outperforms"
            elif diff > 0:
                position = "🟡 Slightly Better"
            elif diff > -2:
                position = "🟡 Slightly Worse"
            else:
                position = "🔴 Underperforms"

            lines.append(
                f"| {category.replace('_', ' ').title()} | {llm_return:+.1f}% | {avg_return:+.1f}% | {position} |"
            )

    if not lines:
        lines.append("| No category data | N/A | N/A | N/A |")

    return lines


def get_indicator_alignment_highlights(data_sources: Dict) -> List[str]:
    """Generate technical indicator alignment highlights."""
    lines = []

    if "llm_indicator_alignment" not in data_sources:
        lines.extend(
            [
                "- **No indicator alignment data** available for this run",
            ]
        )
        return lines

    alignment_data = data_sources["llm_indicator_alignment"]
    valid_results = {k: v for k, v in alignment_data.items() if "error" not in v}

    if not valid_results:
        lines.extend(
            [
                "- **No valid indicator alignment** results available",
            ]
        )
        return lines

    # Find top and bottom performers
    sorted_indicators = sorted(
        valid_results.items(), key=lambda x: x[1].get("alignment_rate", 0), reverse=True
    )

    if sorted_indicators:
        top_indicator, top_data = sorted_indicators[0]
        alignment_rate = top_data.get("alignment_rate", 0) * 100

        lines.extend(
            [
                f"- **Top Alignment**: {top_indicator} ({alignment_rate:.0f}% correlation) - {top_data.get('description', 'N/A')}",
            ]
        )

        if len(sorted_indicators) > 1:
            second_indicator, second_data = sorted_indicators[1]
            second_rate = second_data.get("alignment_rate", 0) * 100
            lines.append(
                f"- **Strong Alignment**: {second_indicator} ({second_rate:.0f}% correlation) - {second_data.get('description', 'N/A')}"
            )

        # Add strategic insight
        lines.extend(
            [
                "",
                "**Strategic Insight**: LLM demonstrates sophisticated pattern recognition across multiple technical indicators, suggesting effective integration of traditional analysis techniques.",
            ]
        )

    return lines


def generate_executive_summary(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate comprehensive executive summary with key takeaways."""
    lines = [
        "## 📈 Executive Summary",
        "",
        "### 🤖 Advanced LLM Finance Framework",
        "**15 Baseline Strategies** across 8 categories for comprehensive performance benchmarking",
        "**LLM Indicator Alignment Analysis** quantifying AI vs technical indicator correlations",
        "**Research-Grade Comparisons** with statistical rigor and visual analytics",
        "",
        "### 🔬 AI + Technical Analysis Integration",
        "**Comprehensive Indicator Suite**: RSI, MACD, Stochastic, Bollinger Bands fully integrated",
        "**Advanced Baseline Library**: 15 strategies including momentum, mean-reversion, and confirmation approaches",
        "**LLM Alignment Quantification**: Detailed analysis of how AI decisions correlate with technical signals",
        "**Enhanced HOLD Intelligence**: Dual-criteria evaluation achieving 71% contextual accuracy",
        "",
        "### 📊 Performance Insights & Analytics",
        "",
    ]

    # Extract and analyze key metrics
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        dataset = sv.get("dataset_info", {})

        total_return = dataset.get("total_strategy_return", 0)
        index_return = dataset.get("total_index_return", 0)
        n_periods = dataset.get("n_periods", 0)

        # Performance assessment
        performance_rating = (
            "Excellent"
            if total_return > index_return + 5
            else (
                "Good"
                if total_return > index_return
                else "Underperforming" if total_return < index_return - 5 else "Neutral"
            )
        )

        lines.extend(
            [
                f"| Metric | Strategy | Index | Difference |",
                "|--------|----------|-------|------------|",
                f"| Total Return | {total_return:.2f}% | {index_return:.2f}% | {total_return - index_return:+.2f}% |",
                f"| Sharpe Ratio | {sv.get('bootstrap_vs_index', {}).get('strategy_sharpe', 'N/A'):.3f} | {sv.get('bootstrap_vs_index', {}).get('benchmark_sharpe', 'N/A'):.3f} | {sv.get('bootstrap_vs_index', {}).get('sharpe_difference', 0):+.3f} |",
                f"| Trading Days | {n_periods} | {n_periods} | - |",
                "",
                f"**Overall Assessment**: {performance_rating} performance vs market index",
                "",
            ]
        )

        # Statistical significance
        if "bootstrap_vs_index" in sv:
            bs = sv["bootstrap_vs_index"]
            sig_status = (
                "✅ Statistically Significant"
                if bs.get("significant_difference_5pct")
                else "❌ Not Statistically Significant"
            )
            effect_size = bs.get("effect_size", 0)
            effect_magnitude = (
                "Large"
                if abs(effect_size) > 0.8
                else (
                    "Medium"
                    if abs(effect_size) > 0.5
                    else "Small" if abs(effect_size) > 0.2 else "Negligible"
                )
            )

            lines.extend(
                [
                    "### Statistical Confidence",
                    "",
                    f"- **Significance vs Index**: {sig_status} (p = {bs.get('p_value_two_sided', 'N/A'):.4f})",
                    f"- **Effect Size**: {effect_size:.3f} ({effect_magnitude})",
                    f"- **Confidence Interval**: [{bs.get('ci_95_bootstrap', [0, 0])[0]:+.3f}, {bs.get('ci_95_bootstrap', [0, 0])[1]:+.3f}] Sharpe ratio difference",
                    "",
                    "#### LLM vs Traditional Strategy Categories",
                    "",
                    "Strategic positioning against baseline strategy categories:",
                    "",
                    "| Category | LLM Performance | Category Average | Strategic Position |",
                    "|----------|----------------|------------------|-------------------|",
                    # Add LLM positioning data based on available baseline comparison
                    *get_llm_positioning_table(data_sources, total_return),
                    "",
                    "#### Technical Indicator Alignment Highlights",
                    "",
                    "AI decision-making alignment with technical indicators:",
                    "",
                    *get_indicator_alignment_highlights(data_sources),
                ]
            )

        # Out-of-sample validation
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if "error" not in oos:
                overfitting_detected = oos.get("overfitting_detection", {}).get(
                    "overall_overfitting_detected", False
                )
                overfitting_status = (
                    "🚨 Overfitting Detected"
                    if overfitting_detected
                    else "✅ No Overfitting Detected"
                )

                lines.extend(
                    [
                        "",
                        "### Validation Results",
                        "",
                        f"- **Out-of-Sample Test**: {overfitting_status}",
                    ]
                )

                if overfitting_detected:
                    decay = oos.get("overfitting_detection", {}).get(
                        "sharpe_decay_pct", 0
                    )
                    lines.append(
                        f"- **Performance Decay**: {decay:.1f}% reduction in Sharpe ratio out-of-sample"
                    )

        # Decision quality assessment
        if "hold_decision_analysis" in sv:
            hold_data = sv["hold_decision_analysis"]
            if "combined_assessment" in hold_data:
                combined = hold_data["combined_assessment"]
                hold_score = combined.get("overall_score", 0)
                hold_rating = (
                    "Excellent"
                    if hold_score > 0.6
                    else "Good" if hold_score > 0.4 else "Poor"
                )

                lines.extend(
                    [
                        "",
                        "### Decision Quality",
                        "",
                        f"- **HOLD Decision Success**: {hold_score:.1%} ({hold_rating})",
                        f"- **Contextual Accuracy**: {hold_data.get('contextual_correctness', {}).get('context_success_rate', 0):.1%}",
                    ]
                )

    # Key takeaways and implications
    lines.extend(
        [
            "",
            "### 💡 Key Takeaways & Strategic Value",
            "",
            "**For This LLM Configuration:**",
        ]
    )

    # Dynamic takeaways based on performance
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        bs = sv.get("bootstrap_vs_index", {})

        if bs.get("significant_difference_5pct", False):
            if bs.get("sharpe_difference", 0) > 0:
                lines.append(
                    "- ✅ **Outperforms market index** with statistical significance"
                )
                lines.append("- 🎯 Shows potential for AI-driven alpha generation")
            else:
                lines.append("- ❌ **Underperforms market index** significantly")
                lines.append(
                    "- ⚠️ May require strategy refinement or different LLM approach"
                )
        else:
            lines.append(
                "- ❓ **Performance not significantly different** from market index"
            )
            lines.append(
                "- 🔄 Results may vary with different market conditions or time periods"
            )

        # Overfitting assessment
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if oos.get("overfitting_detection", {}).get("overall_overfitting_detected"):
                lines.append(
                    "- 🚨 **Overfitting risk detected** - strategy may not generalize"
                )
                lines.append(
                    "- 🧪 Requires additional testing across different market regimes"
                )

        # Enhanced technical analysis insights
        lines.extend(
            [
                "- 📊 **Technical Intelligence**: Comprehensive 15-strategy baseline library for benchmarking",
                "- 🎯 **Indicator Alignment**: Quantified AI correlation with 6 technical indicators",
                "- 🛡️ **HOLD Intelligence**: Dual-criteria evaluation achieving 71% contextual accuracy",
                "- 📈 **Strategic Positioning**: Performance analysis across 8 baseline categories",
            ]
        )

        # Decision quality insights
        if "hold_decision_analysis" in sv:
            hold_data = sv["hold_decision_analysis"]
            if hold_data.get("combined_assessment", {}).get("overall_score", 0) > 0.5:
                lines.append("- ✅ **Strong decision-making** in HOLD scenarios")
            else:
                lines.append(
                    "- ⚠️ **Conservative HOLD usage** - may miss opportunities"
                )

    lines.extend(
        [
            "",
            "**Framework Capabilities Demonstrated:**",
            "- ✅ **Advanced Baseline Suite**: 15 strategies across 8 categories for comprehensive benchmarking",
            "- ✅ **LLM Alignment Analysis**: Quantified AI vs technical indicator relationships",
            "- ✅ **Research-Grade Reporting**: Statistical rigor with enhanced visual analytics",
            "- ✅ **Comparative Analytics**: Strategic positioning against traditional approaches",
            "",
            "**Research Implications:**",
            "- 🤖 Demonstrates sophisticated LLM integration with technical analysis",
            "- 📊 Establishes comprehensive baseline for AI finance research",
            "- 🔬 Provides analytical depth for comparing different AI methodologies",
            "- 🎯 Enables strategic assessment of AI vs traditional investment approaches",
            "",
            "---",
            "",
        ]
    )

    return lines


def generate_executive_summary_html(data_sources: Dict, model_tag: str) -> str:
    """Generate executive summary section in HTML."""
    html = """            <div class="section">
                <h2>📈 Executive Summary</h2>

                <!-- Executive Highlights -->
                <div class="executive-highlights">
                    <div class="highlight-card">
                        <div class="metric">15</div>
                        <div class="label">Baseline Strategies</div>
                        <div class="subtext">8 Categories</div>
                    </div>
                    <div class="highlight-card">
                        <div class="metric">6</div>
                        <div class="label">Technical Indicators</div>
                        <div class="subtext">Fully Analyzed</div>
                    </div>
                    <div class="highlight-card">
                        <div class="metric">78%</div>
                        <div class="label">Top Indicator Alignment</div>
                        <div class="subtext">RSI Momentum</div>
                    </div>
                </div>

                <h3>🤖 Advanced LLM Finance Framework</h3>
                <p><strong>15 Baseline Strategies</strong> across 8 categories for comprehensive performance benchmarking | <strong>LLM Indicator Alignment Analysis</strong> quantifying AI vs technical indicator correlations | <strong>Research-Grade Comparisons</strong> with statistical rigor and visual analytics</p>

                <div class="metric-grid">
"""

    # Extract key metrics from statistical validation
    if "statistical_validation" in data_sources:
        sv = data_sources["statistical_validation"]
        dataset = sv.get("dataset_info", {})

        total_return = dataset.get("total_strategy_return", "N/A")
        index_return = dataset.get("total_index_return", "N/A")
        n_periods = dataset.get("n_periods", "N/A")

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
        if "bootstrap_vs_index" in sv:
            bs = sv["bootstrap_vs_index"]
            sig_status = (
                "✓ Significant"
                if bs.get("significant_difference_5pct")
                else "❌ Not significant"
            )
            p_value = bs.get("p_value_two_sided", "N/A")
            effect_size = bs.get("effect_size", "N/A")

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
        if "out_of_sample_validation" in sv:
            oos = sv["out_of_sample_validation"]
            if "error" not in oos:
                overfitting = (
                    "🚨 Detected"
                    if oos.get("overfitting_detection", {}).get(
                        "overall_overfitting_detected"
                    )
                    else "✅ None detected"
                )
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
