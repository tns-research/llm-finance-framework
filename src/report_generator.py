# src/report_generator.py
"""
Comprehensive Experiment Report Generator.

Creates a single, showable document consolidating all analysis results:
- Statistical validation
- Baseline comparisons
- Calibration analysis
- Decision patterns
- Risk analysis
- Performance metrics

Generates missing charts and provides executive summary.
"""

# Import existing analysis modules.
#
# report_generator is the public facade: it re-exports the section renderers and
# the two master orchestrators (now living in report.document) so external
# callers keep importing them from here. The names rg's own body uses
# (calculate_category_performance, create_statistical_visualizations) are kept
# alongside the re-exported public surface.
from .report.data_collection import collect_data_sources
from .report.document import (
    generate_master_report,
    generate_master_report_html,
)
from .report.pipeline import (
    generate_additional_charts,
    generate_comprehensive_report,
)
from .report.sections.alignment import (
    generate_llm_indicator_alignment_section,
    generate_llm_indicator_alignment_section_html,
)
from .report.sections.baselines import (
    generate_baseline_strategies_section,
    generate_baseline_strategies_section_html,
)
from .report.sections.category_performance import (
    generate_category_performance_section,
    generate_category_performance_section_html,
)
from .report.sections.chain_of_thought import (
    generate_chain_of_thought_section,
    generate_chain_of_thought_section_html,
)
from .report.sections.decision_behavior import (
    generate_decision_behavior_analysis,
    generate_decision_behavior_analysis_html,
)
from .report.sections.executive_summary import (
    generate_executive_summary,
    generate_executive_summary_html,
)
from .report.sections.indicator_performance import (
    generate_indicator_performance_section,
    generate_indicator_performance_section_html,
)
from .report.sections.insights import (
    generate_insights_recommendations,
    generate_insights_recommendations_html,
)
from .report.sections.market_regime import (
    generate_market_regime_analysis,
    generate_market_regime_analysis_html,
)
from .report.sections.methodology import (
    generate_methodology_section,
    generate_methodology_section_html,
)
from .report.sections.performance_overview import (
    generate_performance_overview,
    generate_performance_overview_html,
)
from .report.sections.practical_considerations import (
    generate_practical_considerations,
    generate_practical_considerations_html,
)
from .report.sections.risk import (
    generate_comprehensive_risk_analysis,
    generate_comprehensive_risk_analysis_html,
)
from .report.sections.statistical import (
    generate_statistical_validation_section,
    generate_statistical_validation_section_html,
)
from .report.sections.strategy_comparison import (
    generate_strategy_comparison_insights,
    generate_strategy_comparison_insights_html,
)
from .report.sections.technical_details import (
    generate_technical_details,
    generate_technical_details_html,
)

# Public facade surface: orchestrators + section renderers re-exported from here,
# plus the entry points defined in this module. Listing them in __all__ keeps the
# re-exports honest (they are intentional, not dead imports).
__all__ = [
    "generate_comprehensive_report",
    "collect_data_sources",
    "generate_additional_charts",
    "generate_master_report",
    "generate_master_report_html",
    "generate_executive_summary",
    "generate_executive_summary_html",
    "generate_methodology_section",
    "generate_methodology_section_html",
    "generate_baseline_strategies_section",
    "generate_baseline_strategies_section_html",
    "generate_performance_overview",
    "generate_performance_overview_html",
    "generate_comprehensive_risk_analysis",
    "generate_comprehensive_risk_analysis_html",
    "generate_market_regime_analysis",
    "generate_market_regime_analysis_html",
    "generate_statistical_validation_section",
    "generate_statistical_validation_section_html",
    "generate_decision_behavior_analysis",
    "generate_decision_behavior_analysis_html",
    "generate_chain_of_thought_section",
    "generate_chain_of_thought_section_html",
    "generate_llm_indicator_alignment_section",
    "generate_llm_indicator_alignment_section_html",
    "generate_strategy_comparison_insights",
    "generate_strategy_comparison_insights_html",
    "generate_category_performance_section",
    "generate_category_performance_section_html",
    "generate_indicator_performance_section",
    "generate_indicator_performance_section_html",
    "generate_practical_considerations",
    "generate_practical_considerations_html",
    "generate_insights_recommendations",
    "generate_insights_recommendations_html",
    "generate_technical_details",
    "generate_technical_details_html",
]
