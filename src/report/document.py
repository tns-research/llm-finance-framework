# src/report/document.py
"""Master report assembly: combine all section renderers into one document.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
Holds the two top-level orchestrators that stitch the section renderers into a
full markdown / HTML report.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

from .sections.alignment import (
    generate_llm_indicator_alignment_section,
    generate_llm_indicator_alignment_section_html,
)
from .sections.baselines import (
    generate_baseline_strategies_section,
    generate_baseline_strategies_section_html,
)
from .sections.category_performance import (
    generate_category_performance_section,
    generate_category_performance_section_html,
)
from .sections.chain_of_thought import (
    generate_chain_of_thought_section,
    generate_chain_of_thought_section_html,
)
from .sections.decision_behavior import (
    generate_decision_behavior_analysis,
    generate_decision_behavior_analysis_html,
)
from .sections.executive_summary import (
    generate_executive_summary,
    generate_executive_summary_html,
)
from .sections.insights import (
    generate_insights_recommendations,
    generate_insights_recommendations_html,
)
from .sections.market_regime import (
    generate_market_regime_analysis,
    generate_market_regime_analysis_html,
)
from .sections.methodology import (
    generate_methodology_section,
    generate_methodology_section_html,
)
from .sections.performance_overview import (
    generate_performance_overview,
    generate_performance_overview_html,
)
from .sections.practical_considerations import (
    generate_practical_considerations,
    generate_practical_considerations_html,
)
from .sections.risk import (
    generate_comprehensive_risk_analysis,
    generate_comprehensive_risk_analysis_html,
)
from .sections.statistical import (
    generate_statistical_validation_section,
    generate_statistical_validation_section_html,
)
from .sections.strategy_comparison import (
    generate_strategy_comparison_insights,
    generate_strategy_comparison_insights_html,
)
from .sections.technical_details import (
    generate_technical_details,
    generate_technical_details_html,
)
from .theme import REPORT_CSS


def generate_master_report(
    model_tag: str, data_sources: Dict, analysis_dir: Path, plots_dir: Path
) -> str:
    """
    Generate the master markdown report combining all analyses.
    """
    report_lines = []

    # Header
    report_lines.extend(
        [
            f"# LLM Trading Strategy Experiment Report",
            f"## Model: {model_tag} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]
    )

    # Executive Summary - Start with key takeaways
    report_lines.extend(generate_executive_summary(data_sources, model_tag))

    # Methodology & Technical Implementation
    report_lines.extend(generate_methodology_section(data_sources, model_tag))

    # Enhanced Baseline Strategy Suite - Show off our 15 strategies
    report_lines.extend(generate_baseline_strategies_section(data_sources, model_tag))

    # Performance Overview - Visual summary of results
    report_lines.extend(generate_performance_overview(data_sources, model_tag))

    # Comprehensive Risk Analysis - Combine risk attribution and risk analysis
    report_lines.extend(generate_comprehensive_risk_analysis(data_sources, model_tag))

    # Market Environment Analysis - How strategy performs in different conditions
    report_lines.extend(generate_market_regime_analysis(data_sources, model_tag))

    # Statistical Rigor - Validation and confidence assessment
    report_lines.extend(
        generate_statistical_validation_section(data_sources, model_tag)
    )

    # Decision Behavior Analysis - LLM decision-making patterns
    report_lines.extend(generate_decision_behavior_analysis(data_sources, model_tag))

    # Chain of Thought Reasoning Analysis (conditional)
    if "chain_of_thought_quality" in data_sources:
        report_lines.extend(generate_chain_of_thought_section(data_sources, model_tag))

    # LLM Indicator Alignment Analysis - How decisions align with technical indicators
    report_lines.extend(
        generate_llm_indicator_alignment_section(data_sources, model_tag)
    )

    # Strategy Comparison Insights - LLM positioning vs traditional strategies
    report_lines.extend(generate_strategy_comparison_insights(data_sources, model_tag))

    # Strategy Category Performance Analysis - Group performance by trading style
    report_lines.extend(generate_category_performance_section(data_sources, model_tag))

    # Indicator-Specific Performance Analysis - DISABLED
    # TODO: Re-enable after fixing conceptual bug (2025-12-23-indicator-performance-analysis-conceptual-bug.md)
    # Issue: Analysis incorrectly attempts to compare "with indicators" vs "without indicators"
    # using data from single run, producing meaningless huge percentages through improper
    # cumulative return compounding over 500+ trading days.
    # report_lines.extend(generate_indicator_performance_section(data_sources, model_tag))

    # Practical Implementation - Real-world deployment considerations
    report_lines.extend(generate_practical_considerations(data_sources, model_tag))

    # Key Insights & Strategic Recommendations
    report_lines.extend(generate_insights_recommendations(data_sources, model_tag))

    # Technical Appendix - Data sources and methodology
    report_lines.extend(generate_technical_details(data_sources, model_tag))

    return "\n".join(report_lines)


def generate_master_report_html(
    model_tag: str, data_sources: Dict, analysis_dir: Path, plots_dir: Path
) -> str:
    """
    Generate the master HTML report combining all analyses with beautiful styling.
    """
    html_parts = []

    # HTML Header with CSS
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Trading Strategy Report - {model_tag}</title>
    <style>
{REPORT_CSS}    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LLM Trading Strategy Experiment Report</h1>
            <div class="subtitle">Model: {model_tag} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        <div class="content">
""")

    # Executive Summary Section
    html_parts.append(generate_executive_summary_html(data_sources, model_tag))

    # Methodology & Technical Implementation Section
    html_parts.append(generate_methodology_section_html(data_sources, model_tag))

    # Enhanced Baseline Strategy Suite
    html_parts.append(
        generate_baseline_strategies_section_html(data_sources, model_tag)
    )

    # Performance Overview Section
    html_parts.append(generate_performance_overview_html(data_sources, model_tag))

    # Comprehensive Risk Analysis Section
    html_parts.append(
        generate_comprehensive_risk_analysis_html(data_sources, model_tag)
    )

    # Market Environment Analysis Section
    html_parts.append(generate_market_regime_analysis_html(data_sources, model_tag))

    # Statistical Rigor Section
    html_parts.append(
        generate_statistical_validation_section_html(data_sources, model_tag)
    )

    # Decision Behavior Analysis Section
    html_parts.append(generate_decision_behavior_analysis_html(data_sources, model_tag))

    # Chain of Thought Reasoning Analysis (conditional)
    if "chain_of_thought_quality" in data_sources:
        html_parts.append(
            generate_chain_of_thought_section_html(data_sources, model_tag)
        )

    # Enhanced LLM Indicator Alignment Analysis Section
    html_parts.append(
        generate_llm_indicator_alignment_section_html(data_sources, model_tag)
    )

    # Strategy Comparison Insights Section
    html_parts.append(
        generate_strategy_comparison_insights_html(data_sources, model_tag)
    )

    # Strategy Category Performance Analysis
    html_parts.append(
        generate_category_performance_section_html(data_sources, model_tag)
    )

    # Indicator-Specific Performance Analysis - DISABLED
    # TODO: Re-enable after fixing conceptual bug (2025-12-23-indicator-performance-analysis-conceptual-bug.md)
    # Issue: Analysis incorrectly attempts to compare "with indicators" vs "without indicators"
    # using data from single run, producing meaningless huge percentages through improper
    # cumulative return compounding over 500+ trading days.
    # html_parts.append(
    #     generate_indicator_performance_section_html(data_sources, model_tag)
    # )

    # Practical Implementation Section
    html_parts.append(generate_practical_considerations_html(data_sources, model_tag))

    # Key Insights & Strategic Recommendations Section
    html_parts.append(generate_insights_recommendations_html(data_sources, model_tag))

    # Technical Appendix Section
    html_parts.append(generate_technical_details_html(data_sources, model_tag))

    # Close HTML
    html_parts.append(
        """
        </div>
        <div class="footer">
            <p><strong>LLM Finance Experiment Framework</strong></p>
            <p>This report was automatically generated. For questions about methodology or results, refer to the technical documentation.</p>
            <p>Generated on """
        + datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
        + """</p>
        </div>
    </div>
</body>
</html>"""
    )

    return "".join(html_parts)
