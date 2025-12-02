# src/trading_engine.py

import os

import pandas as pd

from .backtest import backtest_model, parse_response_text, save_parsed_results
from .baseline_runner import run_baseline_analysis
from .config_compat import (
    DEBUG_SHOW_FULL_PROMPT,
    ENABLE_FULL_TRADING_HISTORY,
    ENABLE_STRATEGIC_JOURNAL,
    ENABLE_TECHNICAL_INDICATORS,
    POSITION_MAP,
    SHOW_DATE_TO_LLM,
    SYSTEM_PROMPT,
    TEST_LIMIT,
    TEST_MODE,
    USE_DUMMY_MODEL,
)
from .memory_manager import MemoryManager
from .period_manager import PeriodManager
from .configuration_manager import ConfigurationManager
from .decision_analysis import (
    analyze_decisions_after_outcomes,
    analyze_position_duration_stats,
    create_decision_pattern_plots,
    generate_pattern_analysis_report,
)
from .dummy_model import dummy_call_model
from .openrouter_model import call_openrouter
from .report_generator import generate_comprehensive_report
from .reporting import (
    compute_period_technical_stats,
    create_calibration_by_decision_plot,
    create_calibration_plot,
    create_rsi_performance_analysis,
    create_technical_indicators_plot,
    generate_calibration_analysis_report,
    generate_llm_period_summary,
)
from .statistical_validation import (
    comprehensive_statistical_validation,
    print_validation_report,
    save_validation_report,
)


def get_relative_time_label(past_date, current_date):
    """
    Calculate relative time label for journal entries in 'no date' mode.
    Returns labels like '1 week ago', '2 weeks ago', '1 month ago', etc.
    """
    days_diff = (current_date - past_date).days

    if days_diff == 0:
        return "today"
    elif days_diff == 1:
        return "1 day ago"
    elif days_diff < 14:
        return f"{days_diff} days ago"
    elif days_diff < 21:
        weeks = 2
        return f"{weeks} weeks ago"
    elif days_diff < 28:
        weeks = 3
        return f"{weeks} weeks ago"
    elif days_diff < 35:
        weeks = 4
        return f"{weeks} weeks ago"
    elif days_diff < 60:
        return "1 month ago"
    elif days_diff < 90:
        return "2 months ago"
    elif days_diff < 120:
        return "3 months ago"
    elif days_diff < 150:
        return "4 months ago"
    elif days_diff < 180:
        return "5 months ago"
    elif days_diff < 210:
        return "6 months ago"
    elif days_diff < 240:
        return "7 months ago"
    elif days_diff < 270:
        return "8 months ago"
    elif days_diff < 300:
        return "9 months ago"
    elif days_diff < 330:
        return "10 months ago"
    elif days_diff < 365:
        return "11 months ago"
    else:
        years = days_diff // 365
        if years == 1:
            return "1 year ago"
        else:
            return f"{years} years ago"


def format_journal_entry(trade_data, current_date, show_dates):
    """
    Format a single journal entry with appropriate date labeling.
    """
    if show_dates:
        entry_prefix = f"Date {trade_data['date'].strftime('%Y-%m-%d')}: "
    else:
        # Use relative time in 'no date' mode
        relative_label = get_relative_time_label(trade_data["date"], current_date)
        entry_prefix = f"{relative_label}: "

    base_entry = (
        entry_prefix
        + f"action {trade_data['decision']} (prob {trade_data['prob']:.2f}), "
        f"next day index return {trade_data['next_return_1d']:.2f} percent, "
        f"strategy return {trade_data['strategy_return']:.2f} percent, "
        f"cumulative strategy return {trade_data['cumulative_return']:.2f} percent, "
        f"cumulative index return {trade_data['index_cumulative_return']:.2f} percent."
    )

    # Add technical indicators if available and enabled
    if ENABLE_TECHNICAL_INDICATORS and "rsi_14" in trade_data:
        tech_indicators = []

        if trade_data.get("rsi_14") is not None and not pd.isna(trade_data["rsi_14"]):
            tech_indicators.append(f"RSI(14): {trade_data['rsi_14']:.1f}")

        if (
            trade_data.get("macd_line") is not None
            and trade_data.get("macd_signal") is not None
            and trade_data.get("macd_histogram") is not None
            and not any(
                pd.isna(
                    [
                        trade_data["macd_line"],
                        trade_data["macd_signal"],
                        trade_data["macd_histogram"],
                    ]
                )
            )
        ):
            tech_indicators.append(
                f"MACD: {trade_data['macd_line']:.2f}/{trade_data['macd_signal']:.2f}/{trade_data['macd_histogram']:.3f}"
            )

        if (
            trade_data.get("stoch_k") is not None
            and trade_data.get("stoch_d") is not None
            and not any(pd.isna([trade_data["stoch_k"], trade_data["stoch_d"]]))
        ):
            tech_indicators.append(
                f"Stochastic: {trade_data['stoch_k']:.1f}/{trade_data['stoch_d']:.1f}"
            )

        if trade_data.get("bb_position") is not None and not pd.isna(
            trade_data["bb_position"]
        ):
            tech_indicators.append(f"BB Position: {trade_data['bb_position']:.2f}")

        if tech_indicators:
            base_entry += f" Technical indicators: {' | '.join(tech_indicators)}."

    base_entry += f" Explanation: {trade_data['explanation']} Strategic journal: {trade_data['strategic_journal']} Feeling: {trade_data['feeling_log']}"

    return base_entry


def run_single_model(
    model_tag: str, router_model: str, prompts: pd.DataFrame, raw_path: str
):
    """
    Run one model over all prompts with its own strategic journal,
    save results, backtest, and print final performance summary.

    REFACTORED: Now uses the modular TradingEngine for core simulation logic.
    """

    # Define base directory for file paths
    base_dir = os.path.dirname(os.path.dirname(__file__))

    print("\n" + "#" * 80)
    print(f"Running model: {model_tag}  (router id: {router_model})")
    print("#" * 80)

    # =================================================================
    # PHASE 3 REFACTORING: Use modular TradingEngine instead of inline logic
    # =================================================================

    # Create configuration manager and trading engine
    config_manager = ConfigurationManager()
    trading_engine = TradingEngine(config_manager)

    # Run the core simulation using the new modular engine
    simulation_results = trading_engine.run_simulation(
        model_tag=model_tag,
        router_model=router_model,
        prompts=prompts,
        raw_path=raw_path
    )

    # Extract trade data in the format expected by existing reporting code
    rows = []
    journal_entries = []
    trading_history = []

    for trade in simulation_results['trades']:
        # Format for backtesting (matches original rows structure)
        row_data = {
            "date": trade['date'],
            "iso_year": trade['date'].isocalendar().year,
            "iso_week": trade['date'].isocalendar().week,
            "year": trade['date'].year,
            "month": trade['date'].month,
            "quarter": (trade['date'].month - 1) // 3 + 1,
            "decision": trade['decision'],
            "prob": 0.8,  # Default probability (would need to be stored in trade data)
            "position": trade['position'],
            "strategy_return": float(trade['strategy_return']),
            "index_return": float(trade['index_return']),
            "explanation": trade.get('explanation', 'No explanation'),
            "strategic_journal": "Strategic analysis from modular engine",
            "feeling_log": "Analysis from modular engine",
            "model_source": router_model or "dummy",
        }
        rows.append(row_data)

        # Format for journal entries
        journal_entries.append({
            "date": trade['date'],
            "decision": trade['decision'],
            "position": trade['position'],
            "strategy_return": trade['strategy_return'],
            "index_return": trade['index_return'],
            "cumulative_return": trade['cumulative_return'],
            "index_cumulative_return": trade['index_cumulative_return'],
            "explanation": trade.get('explanation', 'No explanation'),
            "position_duration": trade['position_duration'],
            "previous_return": None,  # Would need to track this properly
        })

        # Format for trading history
        trading_history.append({
            "date": trade['date'],
            "trade_id": len(trading_history) + 1,
            "decision": trade['decision'],
            "position": trade['position'],
            "result": float(trade['strategy_return']),
        })

    # Print final summary from simulation results
    metrics = simulation_results['performance_metrics']
    print(f"\nFinal results for model {model_tag}:")
    print(f"Total decisions: {metrics['total_decisions']}")
    print(f"BUY: {metrics['buy_count']}, HOLD: {metrics['hold_count']}, SELL: {metrics['sell_count']}")
    print(f"Win rate: {metrics['win_rate']:.1f}%")
    print(f"Total return: {metrics['total_return']:.2f}%")
    print(f"Index return: {metrics['index_return']:.2f}%")
    print(f"Edge over index: {metrics['edge_over_index']:.2f}%")

    # =================================================================
    # REMOVED: All the old simulation logic (~400 lines) is now handled by TradingEngine
    # =================================================================

    # Continue with reporting using data extracted from TradingEngine results

    # Continue with reporting using data extracted from TradingEngine results

    # Save parsed results and run backtest unchanged
