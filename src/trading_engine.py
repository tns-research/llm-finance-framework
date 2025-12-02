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
from .performance_tracker import PerformanceTracker
from .journal_manager import JournalManager
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


def run_single_model(
    model_tag: str, router_model: str, prompts: pd.DataFrame, raw_path: str
):
    """
    Run one model over all prompts with its own strategic journal,
    save results, backtest, and print final performance summary.
    """

    # Define base directory for file paths
    base_dir = os.path.dirname(os.path.dirname(__file__))

    print("\n" + "#" * 80)
    print(f"Running model: {model_tag}  (router id: {router_model})")
    print("#" * 80)

    trading_history = []  # Will store all past trades as data for full history feature
    rows = []

    # Unified performance tracking system
    performance_tracker = PerformanceTracker()

    # Unified journal management system
    journal_manager = JournalManager()

    # Position duration tracking for backward compatibility
    previous_decision = None
    previous_return = None

    # Unified memory and period management system
    config_manager = ConfigurationManager()
    memory_manager = MemoryManager()
    period_manager = PeriodManager(memory_manager, config_manager)

    last_date = None
    total_rows = len(prompts)

    for idx, row in prompts.iterrows():

        if TEST_MODE and idx >= TEST_LIMIT:
            print(
                f"\nTEST MODE ACTIVE  stopping after {TEST_LIMIT} rows for model {model_tag}.\n"
            )
            break

        current_date = row["date"]

        # Compute temporal identifiers for current date
        iso = current_date.isocalendar()
        try:
            iso_year = iso.year
            iso_week = iso.week
        except AttributeError:
            # older pandas returns a tuple
            iso_year = int(iso[0])
            iso_week = int(iso[1])

        year = current_date.year
        month = current_date.month
        quarter = (month - 1) // 3 + 1

        # Check period boundaries and generate summaries using unified system
        if last_date is not None:
            period_manager.check_all_periods(current_date, last_date, router_model, model_tag)

        base_prompt = row["prompt_text"]

        # Short term journal based on past daily trades
        journal_text = journal_manager.get_journal_block(current_date, SHOW_DATE_TO_LLM, ENABLE_TECHNICAL_INDICATORS)

        # Performance summary based only on past days
        performance_summary = performance_tracker.get_performance_summary()

        # Generate memory blocks using unified system
        memory_blocks = memory_manager.get_all_memory_blocks()
        weekly_block = memory_blocks['weekly']
        monthly_block = memory_blocks['monthly']
        quarterly_block = memory_blocks['quarterly']
        yearly_block = memory_blocks['yearly']

        # Full trading history block (always enabled if feature is on)
        if ENABLE_FULL_TRADING_HISTORY and trading_history:
            if SHOW_DATE_TO_LLM:
                # Include dates when date mode is enabled
                history_lines = ["date,decision,position,result"]
                for entry in trading_history:
                    history_lines.append(
                        f"{entry['date']},{entry['decision']},{entry['position']},{entry['result']}"
                    )
            else:
                # Omit dates in anonymized mode
                history_lines = ["trade_id,decision,position,result"]
                for entry in trading_history:
                    history_lines.append(
                        f"{entry['trade_id']},{entry['decision']},{entry['position']},{entry['result']}"
                    )
            trading_history_block = "TRADING_HISTORY:\n" + "\n".join(history_lines)
        else:
            trading_history_block = "TRADING_HISTORY:\nNo trading history yet."

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
                f"{SYSTEM_PROMPT}\n\n"
                "===== USER MESSAGE =====\n"
                f"{user_prompt}\n"
            )
            print("\n==============================")
            print(f"FULL PROMPT SENT TO MODEL {model_tag} :")
            print("==============================")
            print(full_debug_prompt)
            print("=============== END FULL PROMPT ===============\n")

        # Call the model
        if USE_DUMMY_MODEL or router_model is None:
            response = dummy_call_model(SYSTEM_PROMPT, user_prompt)
            model_source = "dummy"
        else:
            response = call_openrouter(router_model, SYSTEM_PROMPT, user_prompt)
            model_source = model_tag

        # Parse the model response
        try:
            decision, prob, explanation, strategic_journal, feeling_log = (
                parse_response_text(response)
            )
        except Exception as e:
            print(f"\n[WARN] Malformed response for model {model_tag}: {e}")
            print("Raw response:")
            print(response)

            decision = "HOLD"
            prob = 0.5
            explanation = (
                "Malformed response. Defaulting to HOLD based on uncertainty "
                "and risk management."
            )
            strategic_journal = (
                "Model produced an invalid output format. Strategy remains neutral "
                "while awaiting consistent behavior."
            )
            feeling_log = "Feeling cautious about reliability and focused on stability."

        position = POSITION_MAP[decision]

        # Daily performance calculation
        daily_return = position * row["next_return_1d"]

        # Update performance tracker with decision and returns
        performance_tracker.update_daily_performance(decision, daily_return, row["next_return_1d"])

        # Get position duration info for backward compatibility
        current_decision, current_position_duration = performance_tracker.get_position_duration_info()
        position_changed = (previous_decision is not None and decision != previous_decision)

        # Update period stats using unified system
        period_manager.update_stats('weekly', strategy_return=daily_return, index_return=row["next_return_1d"], days=1)
        period_manager.update_stats('monthly', strategy_return=daily_return, index_return=row["next_return_1d"], days=1)
        period_manager.update_stats('quarterly', strategy_return=daily_return, index_return=row["next_return_1d"], days=1)
        period_manager.update_stats('yearly', strategy_return=daily_return, index_return=row["next_return_1d"], days=1)

        if daily_return > 0:
            period_manager.update_stats('weekly', wins=1)
            period_manager.update_stats('monthly', wins=1)
            period_manager.update_stats('quarterly', wins=1)
            period_manager.update_stats('yearly', wins=1)

        if decision == "BUY":
            period_manager.update_stats('weekly', buys=1)
            period_manager.update_stats('monthly', buys=1)
            period_manager.update_stats('quarterly', buys=1)
            period_manager.update_stats('yearly', buys=1)
        elif decision == "HOLD":
            period_manager.update_stats('weekly', holds=1)
            period_manager.update_stats('monthly', holds=1)
            period_manager.update_stats('quarterly', holds=1)
            period_manager.update_stats('yearly', holds=1)
        elif decision == "SELL":
            period_manager.update_stats('weekly', sells=1)
            period_manager.update_stats('monthly', sells=1)
            period_manager.update_stats('quarterly', sells=1)
            period_manager.update_stats('yearly', sells=1)

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
            if SHOW_DATE_TO_LLM:
                # Include actual dates when date mode is enabled
                history_entry = {
                    "date": str(
                        row["date"].date()
                    ),  # Convert to string for JSON serialization
                    "decision": decision,
                    "position": position,
                    "result": round(
                        float(daily_return), 6
                    ),  # Strategy return for this trade
                }
            else:
                # Omit dates in anonymized mode to prevent data leakage
                history_entry = {
                    "trade_id": len(trading_history) + 1,  # Simple sequential ID
                    "decision": decision,
                    "position": position,
                    "result": round(
                        float(daily_return), 6
                    ),  # Strategy return for this trade
                }
            trading_history.append(history_entry)

        rows.append(
            {
                "date": row["date"],
                "decision": decision,
                "prob": prob,
                "explanation": explanation,
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

    # Save parsed results and run backtest unchanged
    parsed_results_path = os.path.join(
        base_dir, "results", "parsed", f"{model_tag}_parsed.csv"
    )
    parsed_df = save_parsed_results(parsed_results_path, rows)

    print("\nStep 4  backtest for model", model_tag)
    metrics = backtest_model(parsed_df)
    print("Metrics")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Statistical validation
    print("\nStep 4.5  statistical validation for model", model_tag)

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
    print("\nStep 5  decision pattern analysis for model", model_tag)

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
        print("Generating technical indicators plots...")
        technical_plot_path = os.path.join(
            plots_dir, f"{model_tag}_technical_indicators.png"
        )
        create_technical_indicators_plot(
            features_df, parsed_df, model_tag, technical_plot_path
        )
    else:
        print("Technical indicators plots skipped (no technical indicators enabled)")

    # RSI performance analysis (only if RSI data available)
    if has_rsi:
        print("Generating RSI performance analysis...")
        rsi_plot_path = os.path.join(plots_dir, f"{model_tag}_rsi_performance.png")
        create_rsi_performance_analysis(
            parsed_df, features_df, model_tag, rsi_plot_path
        )
    else:
        print("RSI performance analysis skipped (RSI not enabled)")

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
    print(f"S and P 500 buy and hold return: {sp500_return:.2f} percent")
    print(f"LLM strategy total return:       {llm_return:.2f} percent")

    if llm_return > sp500_return:
        print("Result: LLM strategy outperformed buy and hold in this period.")
    else:
        print("Result: LLM strategy underperformed buy and hold in this period.")
    print("=" * 70)

    # Step 6: Baseline comparison
    print("\nStep 6  baseline comparison for model", model_tag)
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")

    run_baseline_analysis(
        features_path=features_path,
        output_dir=analysis_dir,
        llm_metrics=metrics,
        llm_parsed_df=parsed_df,
        model_tag=model_tag,
    )

    # Generate comprehensive reports (after all analysis is complete)
    print("\nStep 6  generating comprehensive experiment reports for model", model_tag)
    try:
        # Generate both Markdown and HTML reports
        md_report_path = generate_comprehensive_report(
            model_tag, base_dir, output_format="markdown"
        )
        html_report_path = generate_comprehensive_report(
            model_tag, base_dir, output_format="html"
        )
        print(f"✓ Markdown report generated: {md_report_path}")
        print(f"✓ HTML report generated: {html_report_path}")
        print(f"🌐 Open HTML report in browser: file://{html_report_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate comprehensive reports: {e}")
        import traceback

        traceback.print_exc()
