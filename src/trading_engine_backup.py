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
            recent_entries = journal_entries[-10:]
            # Format each entry dynamically with relative dates if needed
            formatted_entries = [
                format_journal_entry(entry, current_date, SHOW_DATE_TO_LLM)
                for entry in recent_entries
            ]
            journal_text = "Past trades and results so far:\n" + "\n".join(
                formatted_entries
            )
        else:
            journal_text = "No past trades yet. You are starting your strategy."

        # Performance summary based only on past days
        if decision_count == 0:
            performance_summary = (
                "No trades executed yet.\n"
                "Strategy cumulative return so far  0.00 percent.\n"
                "S and P 500 cumulative return so far  0.00 percent.\n"
                "BUY 0, HOLD 0, SELL 0.\n"
                "Win rate undefined."
            )
        else:
            edge = cumulative_return - index_cumulative_return
            outperform_word = "outperforming" if edge > 0 else "underperforming"
            win_rate_pct = (win_count / decision_count) * 100.0

            performance_summary = (
                f"Total strategy return so far  {cumulative_return:.2f} percent.\n"
                f"Total S and P 500 return so far  {index_cumulative_return:.2f} percent.\n"
                f"You are {outperform_word} the index by {edge:.2f} percent.\n"
                f"Number of decisions so far  {decision_count} "
                f"(BUY {buy_count}, HOLD {hold_count}, SELL {sell_count}).\n"
                f"Win rate so far  {win_rate_pct:.1f} percent."
            )

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

        # Track position duration and changes
        if current_decision is None:
            # First day
            current_decision = decision
            current_position_duration = 1
            position_changed = False
        elif decision == current_decision:
            # Same position as yesterday
            current_position_duration += 1
            position_changed = False
        else:
            # Position changed
            current_decision = decision
            current_position_duration = 1
            position_changed = True

        # Daily performance
        daily_return = position * row["next_return_1d"]
        cumulative_return += daily_return
        index_cumulative_return += row["next_return_1d"]

        # Update global decision statistics
        decision_count += 1
        if decision == "BUY":
            buy_count += 1
        elif decision == "HOLD":
            hold_count += 1
        elif decision == "SELL":
            sell_count += 1

        if daily_return > 0:
            win_count += 1

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
        trade_data = {
            "date": row["date"],
            "decision": decision,
            "prob": prob,
            "next_return_1d": row["next_return_1d"],
            "strategy_return": daily_return,
            "cumulative_return": cumulative_return,
            "index_cumulative_return": index_cumulative_return,
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

        journal_entries.append(trade_data)

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
                "cumulative_return": cumulative_return,
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
            f"Strat return: {daily_return:.2f} percent   Cumulative: {cumulative_return:.2f} percent"
        )
        print("\nExplanation (today's decision):")
        print(explanation)
        print("\nStrategic journal (past and future reasoning):")
        print(strategic_journal)
    # =================================================================
    # [END OF OLD SIMULATION CODE - Everything above this line was removed]
    # =================================================================

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


# ============================================================================
# NEW MODULAR TRADING ENGINE (Phase 3)
# ============================================================================

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from .performance_tracker import PerformanceTracker
from .trade_history_manager import TradeHistoryManager
from .prompt_builder import PromptBuilder
from .model_client import ModelClient, create_model_client
from .response_parser import ResponseParser
from .configuration_manager import ConfigurationManager
from .memory_manager import MemoryManager
from .period_manager import PeriodManager
from .config_compat import POSITION_MAP, DEBUG_SHOW_FULL_PROMPT, USE_DUMMY_MODEL
from .reporting import generate_llm_period_summary


class TradingEngine:
    """
    Main orchestrator for trading simulations.

    Coordinates all components (performance tracking, prompt building, model interaction,
    memory management, etc.) to run complete trading simulations with clean separation of concerns.
    """

    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize the trading engine with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

        # Core components
        self.memory_manager = MemoryManager()
        self.period_manager = PeriodManager(self.memory_manager, config_manager)

        # Initialize other components as needed
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.trade_history_manager: Optional[TradeHistoryManager] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.model_client: Optional[ModelClient] = None
        self.response_parser: Optional[ResponseParser] = None

    def run_simulation(self, model_tag: str, router_model: Optional[str],
                      prompts: pd.DataFrame, raw_path: str) -> Dict[str, Any]:
        """
        Run a complete trading simulation for a single model.

        Args:
            model_tag: String identifier for the model (for logging)
            router_model: OpenRouter model identifier (None for dummy)
            prompts: DataFrame containing prompts and market data
            raw_path: Path for saving raw results

        Returns:
            Dictionary containing simulation results and metadata
        """
        # Initialize components for this simulation
        self._initialize_simulation_components(router_model)

        print(f"\nRunning simulation: {model_tag} (router: {router_model or 'dummy'})")

        # Track simulation metadata
        simulation_results = {
            'model_tag': model_tag,
            'router_model': router_model,
            'start_time': datetime.now(),
            'rows_processed': 0,
            'trades': [],
            'period_summaries': {},
            'performance_metrics': {},
            'errors': []
        }

        last_date = None
        total_rows = len(prompts)

        # Main simulation loop
        for idx, row in prompts.iterrows():
            try:
                # Process this trading day
                trade_result = self._process_trading_day(
                    row, idx, total_rows, last_date, router_model, model_tag
                )

                simulation_results['trades'].append(trade_result)
                simulation_results['rows_processed'] += 1
                last_date = row['date']

            except Exception as e:
                error_info = {
                    'row_index': idx,
                    'date': row.get('date'),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                simulation_results['errors'].append(error_info)
                print(f"⚠️  Error processing row {idx}: {e}")

        # Finalize simulation
        simulation_results['end_time'] = datetime.now()
        simulation_results['duration'] = (
            simulation_results['end_time'] - simulation_results['start_time']
        ).total_seconds()
        simulation_results['performance_metrics'] = self.performance_tracker.get_final_metrics()

        # Save results
        self._save_simulation_results(simulation_results, raw_path, model_tag)

        return simulation_results

    def _initialize_simulation_components(self, router_model: Optional[str]):
        """
        Initialize all components needed for the simulation.

        Args:
            router_model: OpenRouter model identifier
        """
        flags = self.config_manager.get_feature_flags()

        # Initialize core components
        self.performance_tracker = PerformanceTracker()
        self.trade_history_manager = TradeHistoryManager(show_dates=flags.get('SHOW_DATE_TO_LLM', False))
        self.prompt_builder = PromptBuilder(
            self.config_manager,
            memory_manager=self.memory_manager,
            trade_history_manager=self.trade_history_manager
        )
        self.model_client = create_model_client(
            "simulation_model",
            router_model,
            use_dummy=USE_DUMMY_MODEL
        )
        self.response_parser = ResponseParser()

    def _process_trading_day(self, row: pd.Series, idx: int, total_rows: int,
                           last_date: Optional[datetime], router_model: Optional[str],
                           model_tag: str) -> Dict[str, Any]:
        """
        Process a single trading day.

        Args:
            row: DataFrame row containing prompt and market data
            idx: Row index
            total_rows: Total number of rows in simulation
            last_date: Date of previous trading day
            router_model: Model identifier for period summaries
            model_tag: Model tag for period summaries

        Returns:
            Dictionary with trade result data
        """
        current_date = row['date']

        # Check period boundaries and generate summaries
        if last_date is not None:
            self.period_manager.check_all_periods(current_date, last_date, router_model, model_tag)

        # Build user prompt
        journal_entries = self.trade_history_manager.get_recent_trades(10)
        user_prompt = self.prompt_builder.build_user_prompt(
            row['prompt_text'],
            self.performance_tracker,
            journal_entries,
            current_date
        )

        # Debug logging if enabled
        if DEBUG_SHOW_FULL_PROMPT:
            self._show_debug_prompt(row['prompt_text'], user_prompt, model_tag)

        # Call model and parse response
        response = self.model_client.call_model(
            self.prompt_builder.build_system_prompt(),
            user_prompt
        )

        decision, probability, explanation, strategic_journal, feeling_log = (
            self.response_parser.parse_response(response)
        )

        # Calculate trading result
        position = POSITION_MAP[decision]
        daily_return = position * row['next_return_1d']

        # Update performance tracking
        position_info = self.performance_tracker.update_daily_performance(
            decision, daily_return, row['next_return_1d']
        )

        # Update period statistics
        self.period_manager.update_stats('weekly', strategy_return=daily_return,
                                       index_return=row['next_return_1d'], days=1)
        self.period_manager.update_stats('monthly', strategy_return=daily_return,
                                       index_return=row['next_return_1d'], days=1)
        self.period_manager.update_stats('quarterly', strategy_return=daily_return,
                                       index_return=row['next_return_1d'], days=1)
        self.period_manager.update_stats('yearly', strategy_return=daily_return,
                                       index_return=row['next_return_1d'], days=1)

        # Record win in period stats if applicable
        if daily_return > 0:
            self.period_manager.update_stats('weekly', wins=1)
            self.period_manager.update_stats('monthly', wins=1)
            self.period_manager.update_stats('quarterly', wins=1)
            self.period_manager.update_stats('yearly', wins=1)

        # Add trade to history
        trade_data = {
            'date': current_date,
            'decision': decision,
            'position': position,
            'strategy_return': daily_return,
            'index_return': row['next_return_1d'],
            'cumulative_return': self.performance_tracker.cumulative_return,
            'index_cumulative_return': self.performance_tracker.index_cumulative_return,
            'explanation': explanation,
            'position_duration': position_info['position_duration']
        }
        self.trade_history_manager.add_trade(trade_data)

        # Progress logging
        if idx % 50 == 0 or idx == total_rows - 1:
            self._log_progress(idx, total_rows, current_date, decision, daily_return,
                             self.performance_tracker.cumulative_return)

        return trade_data

    def _show_debug_prompt(self, base_prompt: str, user_prompt: str, model_tag: str):
        """Show full debug prompt if enabled."""
        if not DEBUG_SHOW_FULL_PROMPT:
            return

        full_debug_prompt = (
            "===== SYSTEM PROMPT =====\n"
            f"{self.prompt_builder.build_system_prompt()}\n\n"
            "===== USER MESSAGE =====\n"
            f"{user_prompt}"
        )
        print(f"\n==============================")
        print(f"FULL PROMPT SENT TO MODEL {model_tag} :")
        print("==============================")
        print(full_debug_prompt)
        print("=============== END FULL PROMPT ===============\n")

    def _log_progress(self, idx: int, total_rows: int, current_date: datetime,
                     decision: str, daily_return: float, cumulative_return: float):
        """Log simulation progress."""
        print(f"[{idx+1}/{total_rows}] {current_date.strftime('%Y-%m-%d')} | "
              f"{decision} | Return: {daily_return:.2f}% | "
              f"Cumulative: {cumulative_return:.2f}%")

    def _save_simulation_results(self, results: Dict[str, Any], raw_path: str, model_tag: str):
        """
        Save simulation results to files.

        Args:
            results: Simulation results dictionary
            raw_path: Base path for saving results
            model_tag: Model identifier for filenames
        """
        try:
            # Convert trades to DataFrame and save
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_path = os.path.join(raw_path, f"{model_tag}_trades.csv")
                trades_df.to_csv(trades_path, index=False)
                print(f"✅ Saved {len(results['trades'])} trades to {trades_path}")

            # Save performance metrics
            metrics_path = os.path.join(raw_path, f"{model_tag}_metrics.json")
            import json
            with open(metrics_path, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2, default=str)
            print(f"✅ Saved metrics to {metrics_path}")

        except Exception as e:
            print(f"⚠️  Warning: Could not save simulation results: {e}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
