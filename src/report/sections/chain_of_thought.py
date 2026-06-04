# src/report/sections/chain_of_thought.py
"""Chain-of-thought reasoning analysis section: markdown/HTML renderers.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
"""

from typing import Dict, List


def generate_chain_of_thought_section(data_sources: Dict, model_tag: str) -> List[str]:
    """Generate chain of thought analysis section for markdown reports."""
    lines = [
        "## 🧠 Chain of Thought Reasoning Analysis",
        "",
        "Analysis of reasoning quality and performance correlation.",
        "",
    ]

    # Check for quality analysis results
    if "chain_of_thought_quality" in data_sources:
        quality_data = data_sources["chain_of_thought_quality"]

        # Skip if error
        if "error" in quality_data:
            lines.extend(
                [
                    f"**Analysis Error**: {quality_data['error']}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "### Quality Metrics Summary",
                    "",
                    f"- **Total Reasoning Entries**: {quality_data.get('total_reasoning_entries', 0)}",
                    f"- **Average Reasoning Length**: {quality_data.get('average_reasoning_length', 0):.0f} characters",
                    "",
                    "### Analytical Framework Completeness",
                    "",
                    f"- **Mean Completeness Score**: {quality_data.get('framework_completeness', {}).get('mean_score', 0):.3f}",
                    f"- **High Quality Reasoning**: {quality_data.get('framework_completeness', {}).get('high_quality', 0)} entries (>70% complete)",
                    f"- **Low Quality Reasoning**: {quality_data.get('framework_completeness', {}).get('low_quality', 0)} entries (<30% complete)",
                    "",
                ]
            )

            if "confidence_correlation" in quality_data:
                lines.extend(
                    [
                        "### Confidence Correlation",
                        "",
                        f"- **Correlation between reasoning quality and decision confidence**: {quality_data['confidence_correlation']:.3f}",
                        "",
                    ]
                )

    # Check for performance correlation results
    if "chain_of_thought_performance" in data_sources:
        perf_data = data_sources["chain_of_thought_performance"]

        # Skip if error
        if "error" in perf_data:
            lines.extend(
                [
                    f"**Performance Analysis Error**: {perf_data['error']}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "### Performance Correlation Analysis",
                    "",
                    f"**Sample Size**: {perf_data.get('sample_size', 0)} reasoning entries analyzed",
                    "",
                    "### Quality Score Distribution",
                    "",
                    f"- **Mean Quality Score**: {perf_data.get('quality_score_distribution', {}).get('mean', 0):.3f}",
                    f"- **High Quality Entries**: {perf_data.get('quality_score_distribution', {}).get('high_quality', 0)} (>0.7)",
                    f"- **Low Quality Entries**: {perf_data.get('quality_score_distribution', {}).get('low_quality', 0)} (<0.3)",
                    "",
                    "### Performance Metrics",
                    "",
                    f"- **Overall Win Rate**: {perf_data.get('return_distribution', {}).get('win_rate', 0):.1%}",
                    f"- **Average Return**: {perf_data.get('return_distribution', {}).get('mean_return', 0):.4f}",
                    "",
                ]
            )

            if "quality_return_correlation" in perf_data:
                corr = perf_data["quality_return_correlation"]
                lines.extend(
                    [
                        "### Quality-Performance Correlation",
                        "",
                        f"- **Correlation between reasoning quality and returns**: {corr:.3f}",
                        "",
                    ]
                )

                if abs(corr) > 0.2:
                    significance = "**strong**" if abs(corr) > 0.5 else "**moderate**"
                    direction = "positive" if corr > 0 else "negative"
                    lines.extend(
                        [
                            f"> **Key Finding**: There is a {significance} {direction} correlation between reasoning quality and trading performance.",
                            "",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "> **Finding**: No significant correlation detected between reasoning quality and trading performance.",
                            "",
                        ]
                    )

            if "quality_tercile_analysis" in perf_data:
                tercile = perf_data["quality_tercile_analysis"]
                lines.extend(
                    [
                        "### Quality Tercile Analysis",
                        "",
                        f"- **Low Quality Reasoning Win Rate**: {tercile.get('low_quality_win_rate', 0):.1%}",
                        f"- **High Quality Reasoning Win Rate**: {tercile.get('high_quality_win_rate', 0):.1%}",
                        f"- **Win Rate Difference**: {tercile.get('win_rate_difference', 0):.1%}",
                        "",
                    ]
                )

    # Include plots if available
    if "plots" in data_sources:
        if "chain_of_thought_quality" in data_sources["plots"]:
            lines.extend(
                [
                    "### Reasoning Quality Visualization",
                    "",
                    f"![Chain of Thought Quality]({data_sources['plots']['chain_of_thought_quality']})",
                    "*Figure: Chain of thought reasoning quality analysis and performance correlation*",
                    "",
                ]
            )

    return lines


def generate_chain_of_thought_section_html(data_sources: Dict, model_tag: str) -> str:
    """Generate chain of thought analysis section for HTML reports."""
    html = """            <div class="section">
                <h2>🧠 Chain of Thought Reasoning Analysis</h2>
                <p>Analysis of reasoning quality and performance correlation.</p>
"""

    # Check for quality analysis results
    if "chain_of_thought_quality" in data_sources:
        quality_data = data_sources["chain_of_thought_quality"]

        # Skip if error
        if "error" in quality_data:
            html += f"""
                <div class="error-message">
                    <strong>Analysis Error:</strong> {quality_data['error']}
                </div>
"""
        else:
            html += """
                <h3>Quality Metrics Summary</h3>
                <div class="metric-grid">
"""

            html += f"""
                    <div class="metric-card">
                        <div class="label">Total Reasoning Entries</div>
                        <div class="value">{quality_data.get('total_reasoning_entries', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Average Reasoning Length</div>
                        <div class="value">{quality_data.get('average_reasoning_length', 0):.0f}</div>
                        <small>characters</small>
                    </div>
                </div>

                <h3>Analytical Framework Completeness</h3>
                <div class="metric-grid">
"""

            completeness = quality_data.get("framework_completeness", {})
            html += f"""
                    <div class="metric-card">
                        <div class="label">Mean Completeness Score</div>
                        <div class="value">{completeness.get('mean_score', 0):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">High Quality Reasoning</div>
                        <div class="value">{completeness.get('high_quality', 0)}</div>
                        <small>>70% complete</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Low Quality Reasoning</div>
                        <div class="value">{completeness.get('low_quality', 0)}</div>
                        <small><30% complete</small>
                    </div>
                </div>
"""

            if "confidence_correlation" in quality_data:
                html += f"""
                <h3>Confidence Correlation</h3>
                <div class="metric-card">
                    <div class="label">Reasoning Quality vs Decision Confidence</div>
                    <div class="value">{quality_data['confidence_correlation']:.3f}</div>
                </div>
"""

    # Check for performance correlation results
    if "chain_of_thought_performance" in data_sources:
        perf_data = data_sources["chain_of_thought_performance"]

        # Skip if error
        if "error" in perf_data:
            html += f"""
                <div class="error-message">
                    <strong>Performance Analysis Error:</strong> {perf_data['error']}
                </div>
"""
        else:
            html += f"""
                <h3>Performance Correlation Analysis</h3>
                <p><strong>Sample Size:</strong> {perf_data.get('sample_size', 0)} reasoning entries analyzed</p>

                <h4>Quality Score Distribution</h4>
                <div class="metric-grid">
"""

            quality_dist = perf_data.get("quality_score_distribution", {})
            html += f"""
                    <div class="metric-card">
                        <div class="label">Mean Quality Score</div>
                        <div class="value">{quality_dist.get('mean', 0):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">High Quality Entries</div>
                        <div class="value">{quality_dist.get('high_quality', 0)}</div>
                        <small>>0.7 score</small>
                    </div>
                    <div class="metric-card">
                        <div class="label">Low Quality Entries</div>
                        <div class="value">{quality_dist.get('low_quality', 0)}</div>
                        <small><0.3 score</small>
                    </div>
                </div>

                <h4>Performance Metrics</h4>
                <div class="metric-grid">
"""

            return_dist = perf_data.get("return_distribution", {})
            html += f"""
                    <div class="metric-card">
                        <div class="label">Overall Win Rate</div>
                        <div class="value">{return_dist.get('win_rate', 0):.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Average Return</div>
                        <div class="value">{return_dist.get('mean_return', 0):.4f}</div>
                    </div>
                </div>
"""

            if "quality_return_correlation" in perf_data:
                corr = perf_data["quality_return_correlation"]
                significance = (
                    "strong"
                    if abs(corr) > 0.5
                    else "moderate" if abs(corr) > 0.2 else "weak"
                )
                direction = "positive" if corr > 0 else "negative"

                html += f"""
                <h4>Quality-Performance Correlation</h4>
                <div class="metric-card">
                    <div class="label">Correlation Coefficient</div>
                    <div class="value">{corr:.3f}</div>
                    <small>{significance} {direction} correlation</small>
                </div>
"""

                if abs(corr) > 0.2:
                    html += f"""
                <div class="insight-box">
                    <strong>Key Finding:</strong> There is a {significance} {direction} correlation between reasoning quality and trading performance.
                </div>
"""
                else:
                    html += """
                <div class="insight-box">
                    <strong>Finding:</strong> No significant correlation detected between reasoning quality and trading performance.
                </div>
"""

            if "quality_tercile_analysis" in perf_data:
                tercile = perf_data["quality_tercile_analysis"]
                html += f"""
                <h4>Quality Tercile Analysis</h4>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="label">Low Quality Win Rate</div>
                        <div class="value">{tercile.get('low_quality_win_rate', 0):.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">High Quality Win Rate</div>
                        <div class="value">{tercile.get('high_quality_win_rate', 0):.1%}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Win Rate Difference</div>
                        <div class="value">{tercile.get('win_rate_difference', 0):.1%}</div>
                    </div>
                </div>
"""

    # Include plots if available
    if "plots" in data_sources:
        if "chain_of_thought_quality" in data_sources["plots"]:
            html += f"""
                <h3>Reasoning Quality Visualization</h3>
                <div class="chart-container">
                    <img src="{data_sources['plots']['chain_of_thought_quality']}" alt="Chain of Thought Quality">
                    <div class="chart-caption">Figure: Chain of thought reasoning quality analysis and performance correlation</div>
                </div>
"""

    html += "            </div>\n"
    return html
