# src/reporting/__init__.py
"""Reporting package: period-summary text + matplotlib charts.

This package replaces the old monolithic src/reporting.py. The split is a pure
structural move (no logic change); every public name the old module exposed is
re-exported here so ``from src.reporting import X`` keeps working unchanged.
"""

from .charts import (
    calculate_decision_success,
    create_calibration_by_decision_plot,
    create_calibration_plot,
    create_category_performance_plot,
    create_indicator_performance_plot,
    create_risk_analysis_chart,
    create_rolling_performance_chart,
    create_rsi_performance_analysis,
    create_technical_indicators_plot,
    create_technical_indicators_timeline,
    generate_calibration_analysis_report,
)
from .period_summary import (
    build_period_summary,
    compute_period_technical_stats,
    format_period_technical_indicators,
    generate_llm_period_summary,
    make_empty_stats,
)

__all__ = [
    # period_summary
    "make_empty_stats",
    "build_period_summary",
    "compute_period_technical_stats",
    "format_period_technical_indicators",
    "generate_llm_period_summary",
    # charts
    "calculate_decision_success",
    "create_calibration_plot",
    "create_calibration_by_decision_plot",
    "create_risk_analysis_chart",
    "create_rolling_performance_chart",
    "generate_calibration_analysis_report",
    "create_category_performance_plot",
    "create_indicator_performance_plot",
    "create_technical_indicators_plot",
    "create_technical_indicators_timeline",
    "create_rsi_performance_analysis",
]
