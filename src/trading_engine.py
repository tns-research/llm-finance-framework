# src/trading_engine.py

import logging
import os

import pandas as pd

from .backtest import backtest_model, parse_response_text, save_parsed_results
from .baseline_runner import run_baseline_analysis
from .config import (
    DEBUG_SHOW_FULL_PROMPT,
    ENABLE_CHAIN_OF_THOUGHT,
    ENABLE_FULL_TRADING_HISTORY,
    ENABLE_STRATEGIC_JOURNAL,
    ENABLE_TECHNICAL_INDICATORS,
    POSITION_MAP,
    SHOW_DATE_TO_LLM,
    SYSTEM_PROMPT,
    TEST_LIMIT,
    TEST_MODE,
    USE_DUMMY_MODEL,
    get_current_symbol_info,
)
from .configuration_manager import ConfigurationManager
from .decision_analysis import (
    analyze_decisions_after_outcomes,
    analyze_position_duration_stats,
    create_decision_pattern_plots,
    generate_pattern_analysis_report,
)
from .dummy_model import dummy_call_model
from .journal_manager import JournalManager
from .memory_manager import MemoryManager
from .model_router import generate_response
from .performance_tracker import PerformanceTracker
from .period_manager import PeriodManager
from .reporting import (
    create_calibration_by_decision_plot,
    create_calibration_plot,
    create_rsi_performance_analysis,
    create_technical_indicators_plot,
    generate_calibration_analysis_report,
)
from .statistical_validation import (
    comprehensive_statistical_validation,
    print_validation_report,
    save_validation_report,
)
from .trade_history_manager import TradeHistoryManager

logger = logging.getLogger(__name__)


def _run_decision_loop(
    model_tag,
    router_model,
    prompts,
    performance_tracker,
    journal_manager,
    trade_history_manager,
    memory_manager,
    period_manager,
):
    """Run the daily decision loop over all prompts and return the result rows."""
    rows = []
    previous_decision = None
    previous_return = None
    last_date = None
    total_rows = len(prompts)

    for idx, row in prompts.iterrows():

        if TEST_MODE and idx >= TEST_LIMIT:
            logger.info(
                "TEST MODE ACTIVE: stopping after %d rows for model %s.",
                TEST_LIMIT,
                model_tag,
            )
            break

        current_date = row["date"]

        # Check period boundaries and generate summaries using unified system
        if last_date is not None:
            period_manager.check_all_periods(
                current_date, last_date, router_model, model_tag
            )

        base_prompt = row["prompt_text"]

        # Short term journal based on past daily trades
        journal_text = journal_manager.get_journal_block(
            current_date, SHOW_DATE_TO_LLM, ENABLE_TECHNICAL_INDICATORS
        )

        # Performance summary based only on past days
        performance_summary = performance_tracker.get_performance_summary()

        # Generate memory blocks using unified system
        memory_blocks = memory_manager.get_all_memory_blocks()
        weekly_block = memory_blocks["weekly"]
        monthly_block = memory_blocks["monthly"]
        quarterly_block = memory_blocks["quarterly"]
        yearly_block = memory_blocks["yearly"]

        # Full trading history block (always enabled if feature is on)
        trading_history_block = trade_history_manager.get_history_block(
            SHOW_DATE_TO_LLM, ENABLE_FULL_TRADING_HISTORY
        )

        # Final user prompt sent to the model
        if ENABLE_STRATEGIC_JOURNAL:
            user_prompt = (
                base_prompt
                + "\n\nStrategic journal\n"
                + journal_text
                + "\n\nPerformance summary so far\n"
                + performance_summary
                + "\n\n"
                + weekly_block
                + "\n\n"
                + monthly_block
                + "\n\n"
                + quarterly_block
                + "\n\n"
                + yearly_block
                + "\n\n"
                + trading_history_block
            )
        else:
            # Without strategic journal section
            user_prompt = base_prompt + "\n\n" + trading_history_block

        # Show full prompt for debugging if enabled
        if DEBUG_SHOW_FULL_PROMPT:
            full_debug_prompt = (
                "===== SYSTEM PROMPT =====\n"
                f"{SYSTEM_PROMPT()}\n\n"
                "===== USER MESSAGE =====\n"
                f"{user_prompt}\n"
            )
            logger.debug(
                "Full prompt sent to model %s:\n%s", model_tag, full_debug_prompt
            )

        # Call the model
        if USE_DUMMY_MODEL or router_model is None:
            response = dummy_call_model(SYSTEM_PROMPT(), user_prompt)
        else:
            response = generate_response(router_model, SYSTEM_PROMPT(), user_prompt)

        # Parse the model response with conditional signature handling
        try:
            (
                decision,
                prob,
                explanation,
                chain_of_thought,
                strategic_journal,
                feeling_log,
            ) = parse_response_text(response)
            # If chain of thought is disabled, override with default message
            if not ENABLE_CHAIN_OF_THOUGHT:
                chain_of_thought = "Chain of thought reasoning disabled."
        except Exception as e:
            logger.warning(
                "Malformed response for model %s: %s. Raw response: %s",
                model_tag,
                e,
                response,
            )

            # Error handling maintains system stability with sensible defaults
            chain_of_thought = "Error in chain of thought parsing."
            decision = "HOLD"
            prob = 0.5
            explanation = "Malformed response. Defaulting to HOLD based on uncertainty."
            strategic_journal = "Model produced an invalid output format."
            feeling_log = "Feeling cautious about reliability and focused on stability."

        position = POSITION_MAP[decision]

        # Daily performance calculation
        daily_return = position * row["next_return_1d"]

        # Update performance tracker with decision and returns
        performance_tracker.update_daily_performance(
            decision, daily_return, row["next_return_1d"]
        )

        # Get position duration info for backward compatibility
        current_decision, current_position_duration = (
            performance_tracker.get_position_duration_info()
        )
        position_changed = (
            previous_decision is not None and decision != previous_decision
        )

        # Update period stats using unified system
        decision_kwarg = {"BUY": "buys", "HOLD": "holds", "SELL": "sells"}.get(decision)
        for period in period_manager.periods:
            period_manager.update_stats(
                period,
                strategy_return=daily_return,
                index_return=row["next_return_1d"],
                days=1,
            )
            if daily_return > 0:
                period_manager.update_stats(period, wins=1)
            if decision_kwarg:
                period_manager.update_stats(period, **{decision_kwarg: 1})

        last_date = current_date

        # Store trade data for dynamic journal formatting
        final_metrics = performance_tracker.get_final_metrics()
        trade_data = {
            "date": row["date"],
            "decision": decision,
            "prob": prob,
            "next_return_1d": row["next_return_1d"],
            "strategy_return": daily_return,
            "cumulative_return": final_metrics["total_return"],
            "index_cumulative_return": final_metrics["index_return"],
            "explanation": explanation,
            "strategic_journal": strategic_journal,
            "feeling_log": feeling_log,
        }

        # Add technical indicators to trade data when enabled
        if ENABLE_TECHNICAL_INDICATORS:
            trade_data.update(
                {
                    "rsi_14": row.get("rsi_14"),
                    "macd_line": row.get("macd_line"),
                    "macd_signal": row.get("macd_signal"),
                    "macd_histogram": row.get("macd_histogram"),
                    "stoch_k": row.get("stoch_k"),
                    "stoch_d": row.get("stoch_d"),
                    "bb_position": row.get("bb_position"),
                }
            )

        journal_manager.add_trade_entry(trade_data)

        # Accumulate full trading history for future prompts
        if ENABLE_FULL_TRADING_HISTORY:
            trade_history_manager.add_trade_entry(
                row["date"], decision, position, daily_return, SHOW_DATE_TO_LLM
            )

        rows.append(
            {
                "date": row["date"],
                "decision": decision,
                "prob": prob,
                "explanation": explanation,
                "chain_of_thought": chain_of_thought,  # NEW COLUMN
                "strategic_journal": strategic_journal,
                "feeling_log": feeling_log,
                "position": position,
                "next_return_1d": row["next_return_1d"],
                "strategy_return": daily_return,
                "cumulative_return": final_metrics["total_return"],
                "position_duration": current_position_duration,
                "position_changed": position_changed,
                "previous_decision": previous_decision,
                "previous_return": previous_return,
            }
        )

        # Update previous tracking variables for next iteration
        previous_decision = decision
        previous_return = daily_return

        # Console print
        print("\n" + "=" * 80)
        print(
            f"Model {model_tag}  Day {idx+1} / {total_rows}   Date: {row['date'].strftime('%Y-%m-%d')}"
        )
        print(f"Decision:     {decision}   (probability {prob:.2f})")
        print(f"Index return: {row['next_return_1d']:.2f} percent")
        print(
            f"Strat return: {daily_return:.2f} percent   Cumulative: {final_metrics['total_return']:.2f} percent"
        )
        print("\nExplanation (today's decision):")
        print(explanation)
        print("\nStrategic journal (past and future reasoning):")
        print(strategic_journal)
        print("\nFeeling log:")
        print(feeling_log)

    return rows


def _run_post_analysis(parsed_df, metrics, model_tag, base_dir, raw_path, symbol_name):
    """Run validation, plots, pattern analysis, baselines, and report generation."""
    # Statistical validation
    logger.info("Step 4.5: statistical validation for model %s", model_tag)

    # Determine split date for out-of-sample testing (roughly 70/30 split)
    dates = sorted(parsed_df["date"].unique())
    split_idx = int(len(dates) * 0.7)
    split_date = str(dates[split_idx]) if split_idx < len(dates) else None

    validation_results = comprehensive_statistical_validation(
        parsed_df, model_tag, split_date=split_date
    )

    # Print validation report to console
    print_validation_report(validation_results, model_tag)

    # Save validation results
    analysis_dir = os.path.join(base_dir, "results", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    validation_path = os.path.join(
        analysis_dir, f"{model_tag}_statistical_validation.json"
    )
    save_validation_report(validation_results, validation_path)

    # Generate calibration plot
    plots_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    calibration_plot_path = os.path.join(plots_dir, f"{model_tag}_calibration.png")
    calibration_data = create_calibration_plot(
        parsed_df, model_tag, calibration_plot_path
    )

    # Generate calibration by decision plot
    calibration_by_decision_plot_path = os.path.join(
        plots_dir, f"{model_tag}_calibration_by_decision.png"
    )
    create_calibration_by_decision_plot(
        parsed_df, model_tag, calibration_by_decision_plot_path
    )

    # Generate calibration analysis report
    analysis_dir = os.path.join(base_dir, "results", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    calibration_report_path = os.path.join(
        analysis_dir, f"{model_tag}_calibration_analysis.md"
    )
    generate_calibration_analysis_report(
        calibration_data, parsed_df, model_tag, calibration_report_path
    )

    # Generate decision pattern analysis
    logger.info("Step 5: decision pattern analysis for model %s", model_tag)

    # Analyze and print summary
    decision_stats = analyze_decisions_after_outcomes(parsed_df)
    if "error" not in decision_stats:
        print("\nDecision Pattern Summary:")
        print(f"  Total decisions analyzed: {decision_stats['total_decisions']}")
        print(
            f"  After wins: {decision_stats['total_wins']}, After losses: {decision_stats['total_losses']}"
        )

        if decision_stats.get("chi_square_test"):
            chi_test = decision_stats["chi_square_test"]
            sig_text = "SIGNIFICANT" if chi_test["significant"] else "not significant"
            print(f"  Chi-square test: p={chi_test['p_value']:.4f} ({sig_text})")

    duration_stats = analyze_position_duration_stats(parsed_df)
    print("\nPosition Duration Summary:")
    print(f"  Average duration: {duration_stats['average_position_duration']:.2f} days")
    print(f"  Position changes: {duration_stats['total_position_changes']}")
    if duration_stats.get("longest_streak"):
        streak = duration_stats["longest_streak"]
        print(f"  Longest streak: {streak['decision']} for {streak['duration']} days")

    # Create visualizations
    create_decision_pattern_plots(parsed_df, model_tag, plots_dir)

    # Load features data for technical indicators analysis
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")
    features_df = pd.read_csv(features_path, parse_dates=["date"])

    # Check if technical indicators are available
    has_rsi = "rsi_14" in features_df.columns
    has_macd = "macd_line" in features_df.columns
    has_stoch = "stoch_k" in features_df.columns
    has_bb = "bb_upper" in features_df.columns

    # Create technical indicators plots (conditionally)
    if has_rsi or has_macd or has_stoch or has_bb:
        logger.info("Generating technical indicators plots...")
        technical_plot_path = os.path.join(
            plots_dir, f"{model_tag}_technical_indicators.png"
        )
        create_technical_indicators_plot(
            features_df, parsed_df, model_tag, technical_plot_path
        )
    else:
        logger.info(
            "Technical indicators plots skipped (no technical indicators enabled)"
        )

    # RSI performance analysis (only if RSI data available)
    if has_rsi:
        logger.info("Generating RSI performance analysis...")
        rsi_plot_path = os.path.join(plots_dir, f"{model_tag}_rsi_performance.png")
        create_rsi_performance_analysis(
            parsed_df, features_df, model_tag, rsi_plot_path
        )
    else:
        logger.info("RSI performance analysis skipped (RSI not enabled)")

    # Generate comprehensive report
    analysis_dir = os.path.join(base_dir, "results", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    report_path = os.path.join(analysis_dir, f"{model_tag}_pattern_analysis.md")
    generate_pattern_analysis_report(parsed_df, model_tag, report_path)

    period_start = parsed_df["date"].min()
    period_end = parsed_df["date"].max()

    raw_df = pd.read_csv(raw_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df = raw_df.sort_values("Date").reset_index(drop=True)

    mask = (raw_df["Date"] >= period_start) & (raw_df["Date"] <= period_end)
    period_df = raw_df.loc[mask].reset_index(drop=True)

    if period_df.empty:
        raise RuntimeError("No SP500 data found for the LLM trading period.")

    sp500_start = period_df.loc[0, "Close"]
    sp500_end = period_df.loc[len(period_df) - 1, "Close"]
    sp500_return = (sp500_end - sp500_start) / sp500_start * 100.0

    llm_return = metrics["total_return"]

    print("\n" + "=" * 70)
    print(f"FINAL PERFORMANCE SUMMARY for model {model_tag}")
    print("=" * 70)
    print(f"Period: {period_start.date()} to {period_end.date()}")
    print(f"{symbol_name} buy and hold return: {sp500_return:.2f} percent")
    print(f"LLM strategy total return:       {llm_return:.2f} percent")

    if llm_return > sp500_return:
        print("Result: LLM strategy outperformed buy and hold in this period.")
    else:
        print("Result: LLM strategy underperformed buy and hold in this period.")
    print("=" * 70)

    # Step 6: Baseline comparison
    logger.info("Step 6: baseline comparison for model %s", model_tag)
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")

    run_baseline_analysis(
        features_path=features_path,
        output_dir=analysis_dir,
        llm_metrics=metrics,
        llm_parsed_df=parsed_df,
        model_tag=model_tag,
    )

    # Generate comprehensive reports (after all analysis is complete)
    logger.info(
        "Step 6: generating comprehensive experiment reports for model %s", model_tag
    )
    try:
        # Import here to avoid conflicts with main module loading
        from .report_generator import generate_comprehensive_report

        # Generate both Markdown and HTML reports
        md_report_path = generate_comprehensive_report(
            model_tag, base_dir, output_format="markdown"
        )
        html_report_path = generate_comprehensive_report(
            model_tag, base_dir, output_format="html"
        )
        logger.info("Markdown report generated: %s", md_report_path)
        logger.info("HTML report generated: %s", html_report_path)
        logger.info("Open HTML report in browser: file://%s", html_report_path)
    except Exception:
        logger.exception("Could not generate comprehensive reports")


def run_single_model(
    model_tag: str, router_model: str, prompts: pd.DataFrame, raw_path: str
):
    """
    Run one model over all prompts with its own strategic journal,
    save results, backtest, and print final performance summary.
    """

    # Define base directory for file paths
    base_dir = os.path.dirname(os.path.dirname(__file__))

    logger.info("Running model: %s  (router id: %s)", model_tag, router_model)

    # Unified manager systems
    _, symbol_name = get_current_symbol_info()
    performance_tracker = PerformanceTracker(symbol_name)
    journal_manager = JournalManager()
    trade_history_manager = TradeHistoryManager()
    config_manager = ConfigurationManager()
    memory_manager = MemoryManager()
    period_manager = PeriodManager(memory_manager, config_manager)

    rows = _run_decision_loop(
        model_tag,
        router_model,
        prompts,
        performance_tracker,
        journal_manager,
        trade_history_manager,
        memory_manager,
        period_manager,
    )

    # Save parsed results and run backtest unchanged
    parsed_results_path = os.path.join(
        base_dir, "results", "parsed", f"{model_tag}_parsed.csv"
    )
    parsed_df = save_parsed_results(parsed_results_path, rows)

    logger.info("Step 4: backtest for model %s", model_tag)
    metrics = backtest_model(parsed_df)
    print("Metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    _run_post_analysis(parsed_df, metrics, model_tag, base_dir, raw_path, symbol_name)
