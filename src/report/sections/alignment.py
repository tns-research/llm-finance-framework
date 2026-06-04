# src/report/sections/alignment.py
"""LLM indicator alignment section: shared ranking/summary plus markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


def _rank_alignment_indicators(valid_results):
    """Return alignment results sorted by alignment rate, highest first.

    Shared by the markdown and HTML alignment renderers.
    """
    return sorted(
        valid_results.items(),
        key=lambda x: x[1].get("alignment_rate", 0),
        reverse=True,
    )


def _compute_alignment_pattern_summary(valid_results):
    """Aggregate buy/sell/agreement ratios across valid alignment results.

    Returns None when there are not both bullish and bearish signals to
    compare. Shared by the markdown and HTML alignment renderers.
    """
    total_buy_signals = sum(
        data.get("bullish_signals", 0) for data in valid_results.values()
    )
    total_sell_signals = sum(
        data.get("bearish_signals", 0) for data in valid_results.values()
    )
    if not (total_buy_signals > 0 and total_sell_signals > 0):
        return None

    total_llm_buys = sum(
        data.get("llm_buy_signals", 0)
        for data in valid_results.values()
        if "llm_buy_signals" in data
    )
    total_llm_sells = sum(
        data.get("llm_sell_signals", 0)
        for data in valid_results.values()
        if "llm_sell_signals" in data
    )
    buy_signal_ratio = total_buy_signals / (total_buy_signals + total_sell_signals)
    llm_buy_ratio = (
        total_llm_buys / (total_llm_buys + total_llm_sells)
        if (total_llm_buys + total_llm_sells) > 0
        else 0
    )
    agreement_rate = sum(
        data.get("agreements", 0) for data in valid_results.values()
    ) / sum(data.get("total_signals", 1) for data in valid_results.values())
    return {
        "buy_signal_ratio": buy_signal_ratio,
        "llm_buy_ratio": llm_buy_ratio,
        "agreement_rate": agreement_rate,
    }


def generate_llm_indicator_alignment_section(
    data_sources: Dict, model_tag: str
) -> List[str]:
    """
    Generate enhanced LLM indicator alignment analysis section with insights and patterns.
    """
    lines = [
        "## 🤖 Enhanced LLM Indicator Alignment Analysis",
        "",
        "Comprehensive analysis of how well LLM trading decisions align with key technical indicators,",
        "including effectiveness ranking, pattern analysis, and market condition correlations.",
        "",
    ]

    if "llm_indicator_alignment" not in data_sources:
        lines.extend(
            [
                "**Note:** LLM indicator alignment analysis was not available for this run.",
                "",
            ]
        )
        return lines

    alignment_results = data_sources["llm_indicator_alignment"]

    if not alignment_results:
        lines.extend(
            [
                "**Note:** No alignment data could be calculated.",
                "",
            ]
        )
        return lines

    # Filter out error results for analysis
    valid_results = {k: v for k, v in alignment_results.items() if "error" not in v}

    if valid_results:
        # Effectiveness Ranking
        lines.extend(
            [
                "### 📊 Indicator Effectiveness Ranking",
                "",
                "Ranked by how effectively the LLM aligns with each technical indicator:",
                "",
            ]
        )

        # Sort by alignment rate (highest first)
        sorted_indicators = _rank_alignment_indicators(valid_results)

        lines.extend(
            [
                "| Rank | Indicator | Alignment Rate | Signal Strength | Description |",
                "|------|-----------|----------------|----------------|-------------|",
            ]
        )

        for rank, (indicator, data) in enumerate(sorted_indicators, 1):
            align_rate = data.get("alignment_rate", 0) * 100
            total_signals = data.get("total_signals", 0)
            description = data.get("description", "")

            # Classify signal strength
            if align_rate >= 70:
                strength = "🟢 Strong"
            elif align_rate >= 50:
                strength = "🟡 Moderate"
            else:
                strength = "🔴 Weak"

            lines.append(
                f"| {rank} | {indicator} | {align_rate:.1f}% | {strength} | {description} |"
            )

        lines.append("")

        # Pattern Analysis
        lines.extend(
            [
                "### 🔍 Decision Pattern Analysis",
                "",
                "Analysis of LLM decision patterns relative to indicator signals:",
                "",
            ]
        )

        pattern = _compute_alignment_pattern_summary(valid_results)
        if pattern is not None:
            lines.extend(
                [
                    f"- **Market Direction Bias**: Indicators show {pattern['buy_signal_ratio']:.1%} bullish signals",
                    f"- **LLM Direction Bias**: LLM shows {pattern['llm_buy_ratio']:.1%} buy decisions",
                    f"- **Agreement Rate**: {pattern['agreement_rate']:.1%} overall alignment",
                    "",
                ]
            )

        # Individual indicator breakdown
        lines.extend(
            [
                "#### Individual Indicator Breakdown",
                "",
                "| Indicator | Bullish Signals | Bearish Signals | LLM Buy | LLM Sell | Agreements |",
                "|-----------|----------------|----------------|----------|----------|------------|",
            ]
        )

        for indicator, data in sorted_indicators:
            bullish = data.get("bullish_signals", 0)
            bearish = data.get("bearish_signals", 0)
            llm_buy = data.get("llm_buy_signals", 0)
            llm_sell = data.get("llm_sell_signals", 0)
            agreements = data.get("agreements", 0)

            lines.append(
                f"| {indicator} | {bullish} | {bearish} | {llm_buy} | {llm_sell} | {agreements} |"
            )

        lines.append("")

    # Create detailed alignment table
    lines.extend(
        [
            "### 📋 Detailed Alignment Results",
            "",
            "| Indicator | Alignment Rate | Total Signals | Agreements | Description |",
            "|-----------|----------------|---------------|------------|-------------|",
        ]
    )

    for indicator, data in alignment_results.items():
        if "error" in data:
            lines.append(
                f"| {indicator} | Error | - | - | {data.get('error', 'Unknown error')} |"
            )
        else:
            align_rate = data.get("alignment_rate", 0) * 100
            total_signals = data.get("total_signals", 0)
            agreements = data.get("agreements", 0)
            description = data.get("description", "")

            lines.append(
                f"| {indicator} | {align_rate:.1f}% | {total_signals} | {agreements} | {description} |"
            )

    lines.extend(
        [
            "",
            "### 🎯 Strategic Insights",
            "",
            "#### Effectiveness Categories:",
            "- **🟢 Strong Alignment (>70%)**: LLM effectively incorporates this indicator",
            "- **🟡 Moderate Alignment (50-70%)**: Partial indicator usage",
            "- **🔴 Weak Alignment (<50%)**: Limited or no indicator usage",
            "",
            "#### Pattern Implications:",
            "- **High RSI alignment**: LLM recognizes momentum extremes effectively",
            "- **Low Stochastic alignment**: LLM may use more sophisticated mean-reversion logic",
            "- **MACD alignment**: Shows trend-following behavior patterns",
            "",
            "#### Research Applications:",
            "- Use high-alignment indicators to understand LLM's technical analysis approach",
            "- Low-alignment areas reveal unique LLM decision-making patterns",
            "- Alignment patterns help calibrate LLM vs traditional strategy performance",
            "",
        ]
    )

    return lines


def generate_llm_indicator_alignment_section_html(
    data_sources: Dict, model_tag: str
) -> str:
    """Generate enhanced LLM indicator alignment analysis section in HTML."""
    html = """            <div class="section">
                <h2>🤖 Enhanced LLM Indicator Alignment Analysis</h2>
                <p>Comprehensive analysis of how well LLM trading decisions align with key technical indicators, including effectiveness ranking, pattern analysis, and market condition correlations.</p>
"""

    if "llm_indicator_alignment" not in data_sources:
        html += """                <p><strong>Note:</strong> LLM indicator alignment analysis was not available for this run.</p>
            </div>
"""
        return html

    alignment_results = data_sources["llm_indicator_alignment"]

    if not alignment_results:
        html += """                <p><strong>Note:</strong> No alignment data could be calculated.</p>
            </div>
"""
        return html

    # Filter out error results for analysis
    valid_results = {k: v for k, v in alignment_results.items() if "error" not in v}

    if valid_results:
        # Effectiveness Ranking
        html += """
                <h3>📊 Indicator Effectiveness Ranking</h3>
                <p>Ranked by how effectively the LLM aligns with each technical indicator:</p>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Indicator</th>
                                <th>Alignment Rate</th>
                                <th>Signal Strength</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        # Sort by alignment rate (highest first)
        sorted_indicators = _rank_alignment_indicators(valid_results)

        for rank, (indicator, data) in enumerate(sorted_indicators, 1):
            align_rate = data.get("alignment_rate", 0) * 100
            description = data.get("description", "")

            # Classify signal strength
            if align_rate >= 70:
                strength = "🟢 Strong"
                strength_class = "positive"
            elif align_rate >= 50:
                strength = "🟡 Moderate"
                strength_class = "neutral"
            else:
                strength = "🔴 Weak"
                strength_class = "negative"

            html += f"""
                            <tr>
                                <td>{rank}</td>
                                <td><strong>{indicator}</strong></td>
                                <td class="{strength_class}">{align_rate:.1f}%</td>
                                <td>{strength}</td>
                                <td>{description}</td>
                            </tr>
"""

        html += """
                        </tbody>
                    </table>
                </div>
"""

        # Pattern Analysis
        html += """
                <h3>🔍 Decision Pattern Analysis</h3>
                <p>Analysis of LLM decision patterns relative to indicator signals:</p>
"""

        pattern = _compute_alignment_pattern_summary(valid_results)
        if pattern is not None:
            html += f"""
                <ul>
                    <li><strong>Market Direction Bias:</strong> Indicators show {pattern['buy_signal_ratio']:.1%} bullish signals</li>
                    <li><strong>LLM Direction Bias:</strong> LLM shows {pattern['llm_buy_ratio']:.1%} buy decisions</li>
                    <li><strong>Agreement Rate:</strong> {pattern['agreement_rate']:.1%} overall alignment</li>
                </ul>
"""

        # Individual indicator breakdown
        html += """
                <h4>Individual Indicator Breakdown</h4>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Indicator</th>
                                <th>Bullish Signals</th>
                                <th>Bearish Signals</th>
                                <th>LLM Buy</th>
                                <th>LLM Sell</th>
                                <th>Agreements</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        for indicator, data in sorted_indicators:
            bullish = data.get("bullish_signals", 0)
            bearish = data.get("bearish_signals", 0)
            llm_buy = data.get("llm_buy_signals", 0)
            llm_sell = data.get("llm_sell_signals", 0)
            agreements = data.get("agreements", 0)

            html += f"""
                            <tr>
                                <td><strong>{indicator}</strong></td>
                                <td>{bullish}</td>
                                <td>{bearish}</td>
                                <td>{llm_buy}</td>
                                <td>{llm_sell}</td>
                                <td class="positive">{agreements}</td>
                            </tr>
"""

        html += """
                        </tbody>
                    </table>
                </div>
"""

    # Create detailed alignment table
    html += """
                <h3>📋 Detailed Alignment Results</h3>
                <div class="table-container">
                    <table class="performance-table">
                        <thead>
                            <tr>
                                <th>Indicator</th>
                                <th>Alignment Rate</th>
                                <th>Total Signals</th>
                                <th>Agreements</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    for indicator, data in alignment_results.items():
        if "error" in data:
            html += f"""
                            <tr>
                                <td>{indicator}</td>
                                <td class="negative">Error</td>
                                <td>-</td>
                                <td>-</td>
                                <td>{data.get('error', 'Unknown error')}</td>
                            </tr>
"""
        else:
            align_rate = data.get("alignment_rate", 0) * 100
            total_signals = data.get("total_signals", 0)
            agreements = data.get("agreements", 0)
            description = data.get("description", "")

            align_class = (
                "positive"
                if align_rate >= 70
                else "neutral" if align_rate >= 50 else "negative"
            )

            html += f"""
                            <tr>
                                <td><strong>{indicator}</strong></td>
                                <td class="{align_class}">{align_rate:.1f}%</td>
                                <td>{total_signals}</td>
                                <td>{agreements}</td>
                                <td>{description}</td>
                            </tr>
"""

    html += """
                        </tbody>
                    </table>
                </div>

                <h3>🎯 Strategic Insights</h3>
                <div class="insights-grid">
                    <div class="insight-card">
                        <h4>Effectiveness Categories</h4>
                        <ul>
                            <li><span class="positive">🟢 Strong Alignment (>70%):</span> LLM effectively incorporates this indicator</li>
                            <li><span class="neutral">🟡 Moderate Alignment (50-70%):</span> Partial indicator usage</li>
                            <li><span class="negative">🔴 Weak Alignment (<50%):</span> Limited or no indicator usage</li>
                        </ul>
                    </div>
                    <div class="insight-card">
                        <h4>Pattern Implications</h4>
                        <ul>
                            <li><strong>High RSI alignment:</strong> LLM recognizes momentum extremes effectively</li>
                            <li><strong>Low Stochastic alignment:</strong> LLM may use more sophisticated mean-reversion logic</li>
                            <li><strong>MACD alignment:</strong> Shows trend-following behavior patterns</li>
                        </ul>
                    </div>
                    <div class="insight-card">
                        <h4>Research Applications</h4>
                        <ul>
                            <li>Use high-alignment indicators to understand LLM's technical analysis approach</li>
                            <li>Low-alignment areas reveal unique LLM decision-making patterns</li>
                            <li>Alignment patterns help calibrate LLM vs traditional strategy performance</li>
                        </ul>
                    </div>
                </div>
            </div>
"""

    return html
