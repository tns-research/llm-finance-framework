# src/trading_engine.py

import os

import pandas as pd

from .backtest import backtest_model, parse_response_text, save_parsed_results
from .baseline_runner import run_baseline_analysis
from .config import (
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
    make_empty_stats,
)
from .statistical_validation import (
    comprehensive_statistical_validation,
    print_validation_report,
    save_validation_report,
)


def format_period_technical_indicators(technical_stats: dict, period_name: str) -> str:
    """Format aggregated technical indicators for memory display."""
    if not technical_stats:
        return ""

    lines = []

    # RSI
    if 'rsi_avg' in technical_stats:
        rsi_parts = [f"Average {technical_stats['rsi_avg']:.1f}"]
        if 'rsi_overbought_pct' in technical_stats:
            rsi_parts.append(f"{technical_stats['rsi_overbought_pct']:.0f}% overbought")
        if 'rsi_oversold_pct' in technical_stats:
            rsi_parts.append(f"{technical_stats['rsi_oversold_pct']:.0f}% oversold")
        if 'rsi_min' in technical_stats and 'rsi_max' in technical_stats:
            rsi_parts.append(f"range {technical_stats['rsi_min']:.1f}-{technical_stats['rsi_max']:.1f}")
        lines.append(f"RSI(14): {' ('.join(rsi_parts)})")

    # MACD
    if 'macd_bullish_pct' in technical_stats:
        macd_parts = [f"{technical_stats['macd_bullish_pct']:.0f}% bullish periods"]
        if 'macd_avg_histogram' in technical_stats:
            macd_parts.append(f"avg histogram {technical_stats['macd_avg_histogram']:.3f}")
        if 'macd_crossovers' in technical_stats and technical_stats['macd_crossovers'] > 0:
            macd_parts.append(f"{technical_stats['macd_crossovers']} crossovers")
        lines.append(f"MACD: {', '.join(macd_parts)}")

    # Stochastic
    if 'stoch_overbought_pct' in technical_stats:
        stoch_parts = [f"{technical_stats['stoch_overbought_pct']:.0f}% overbought days"]
        if 'stoch_oversold_pct' in technical_stats:
            stoch_parts.append(f"{technical_stats['stoch_oversold_pct']:.0f}% oversold days")
        lines.append(f"Stochastic: {', '.join(stoch_parts)}")

    # Bollinger Bands
    if 'bb_avg_position' in technical_stats:
        bb_parts = [f"Avg position {technical_stats['bb_avg_position']:.2f}"]
        if 'bb_upper_touch_pct' in technical_stats and 'bb_lower_touch_pct' in technical_stats:
            total_touches = technical_stats['bb_upper_touch_pct'] + technical_stats['bb_lower_touch_pct']
            bb_parts.append(f"band touches {total_touches:.0f}%")
        lines.append(f"Bollinger Bands: {', '.join(bb_parts)}")

    if lines:
        return f"\n\n{period_name} technical indicators:\n" + "\n".join(lines) + "\n"
    return ""


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

        if (trade_data.get("macd_line") is not None and
            trade_data.get("macd_signal") is not None and
            trade_data.get("macd_histogram") is not None and
            not any(pd.isna([trade_data["macd_line"], trade_data["macd_signal"], trade_data["macd_histogram"]]))):
            tech_indicators.append(
                f"MACD: {trade_data['macd_line']:.2f}/{trade_data['macd_signal']:.2f}/{trade_data['macd_histogram']:.3f}"
            )

        if (trade_data.get("stoch_k") is not None and
            trade_data.get("stoch_d") is not None and
            not any(pd.isna([trade_data["stoch_k"], trade_data["stoch_d"]]))):
            tech_indicators.append(f"Stochastic: {trade_data['stoch_k']:.1f}/{trade_data['stoch_d']:.1f}")

        if trade_data.get("bb_position") is not None and not pd.isna(trade_data["bb_position"]):
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
    """

    # Define base directory for file paths
    base_dir = os.path.dirname(os.path.dirname(__file__))

    print("\n" + "#" * 80)
    print(f"Running model: {model_tag}  (router id: {router_model})")
    print("#" * 80)

    journal_entries = []  # Will store dicts with trade data for dynamic formatting
    trading_history = []  # Will store all past trades as data for full history feature
    rows = []

    # Performance tracking over the whole backtest
    cumulative_return = 0.0
    index_cumulative_return = 0.0

    decision_count = 0
    buy_count = 0
    hold_count = 0
    sell_count = 0
    win_count = 0  # days with positive strategy return

    # Position duration tracking
    current_decision = None
    current_position_duration = 0
    previous_decision = None
    previous_return = None

    # Mid and long term memory
    weekly_memory = []
    monthly_memory = []
    quarterly_memory = []
    yearly_memory = []

    week_stats = make_empty_stats()
    month_stats = make_empty_stats()
    quarter_stats = make_empty_stats()
    year_stats = make_empty_stats()

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

        # If we have a previous date, check for period boundaries and close periods
        if last_date is not None:
            prev_iso = last_date.isocalendar()
            try:
                prev_iso_year = prev_iso.year
                prev_iso_week = prev_iso.week
            except AttributeError:
                prev_iso_year = int(prev_iso[0])
                prev_iso_week = int(prev_iso[1])

            prev_year = last_date.year
            prev_month = last_date.month
            prev_quarter = (prev_month - 1) // 3 + 1

            # Week boundary
            if iso_week != prev_iso_week or iso_year != prev_iso_year:
                if week_stats["days"] > 0:
                    technical_stats = None
                    if ENABLE_TECHNICAL_INDICATORS:
                        # Load features data to compute technical stats for the week
                        features_path = os.path.join(base_dir, "data", "processed", "features.csv")
                        features_df = pd.read_csv(features_path, parse_dates=['date'])
                        technical_stats = compute_period_technical_stats(
                            features_df, last_date - pd.Timedelta(days=6), last_date
                        )

                    summary = generate_llm_period_summary(
                        "Week",
                        last_date,
                        week_stats,
                        router_model,
                        model_tag,
                        technical_stats,
                    )
                    weekly_memory.append({
                        "summary": summary,
                        "technical_stats": technical_stats
                    })
                    weekly_memory = weekly_memory[-5:]  # keep last 5
                    week_stats = make_empty_stats()

            # Month boundary
            if month != prev_month or year != prev_year:
                if month_stats["days"] > 0:
                    technical_stats = None
                    if ENABLE_TECHNICAL_INDICATORS:
                        # Load features data to compute technical stats for the month
                        features_path = os.path.join(base_dir, "data", "processed", "features.csv")
                        features_df = pd.read_csv(features_path, parse_dates=['date'])
                        technical_stats = compute_period_technical_stats(
                            features_df, last_date - pd.Timedelta(days=30), last_date
                        )

                    summary = generate_llm_period_summary(
                        "Month",
                        last_date,
                        month_stats,
                        router_model,
                        model_tag,
                        technical_stats,
                    )
                    monthly_memory.append({
                        "summary": summary,
                        "technical_stats": technical_stats
                    })
                    monthly_memory = monthly_memory[-5:]
                    month_stats = make_empty_stats()

            # Quarter boundary
            if quarter != prev_quarter or year != prev_year:
                if quarter_stats["days"] > 0:
                    technical_stats = None
                    if ENABLE_TECHNICAL_INDICATORS:
                        # Load features data to compute technical stats for the quarter
                        features_path = os.path.join(base_dir, "data", "processed", "features.csv")
                        features_df = pd.read_csv(features_path, parse_dates=['date'])
                        technical_stats = compute_period_technical_stats(
                            features_df, last_date - pd.Timedelta(days=90), last_date
                        )

                    summary = generate_llm_period_summary(
                        "Quarter",
                        last_date,
                        quarter_stats,
                        router_model,
                        model_tag,
                        technical_stats,
                    )
                    quarterly_memory.append({
                        "summary": summary,
                        "technical_stats": technical_stats
                    })
                    quarterly_memory = quarterly_memory[-5:]
                    quarter_stats = make_empty_stats()

            # Year boundary
            if year != prev_year:
                if year_stats["days"] > 0:
                    technical_stats = None
                    if ENABLE_TECHNICAL_INDICATORS:
                        # Load features data to compute technical stats for the year
                        features_path = os.path.join(base_dir, "data", "processed", "features.csv")
                        features_df = pd.read_csv(features_path, parse_dates=['date'])
                        technical_stats = compute_period_technical_stats(
                            features_df, last_date - pd.Timedelta(days=365), last_date
                        )

                    summary = generate_llm_period_summary(
                        "Year",
                        last_date,
                        year_stats,
                        router_model,
                        model_tag,
                        technical_stats,
                    )
                    yearly_memory.append({
                        "summary": summary,
                        "technical_stats": technical_stats
                    })
                    yearly_memory = yearly_memory[-5:]
                    year_stats = make_empty_stats()

        base_prompt = row["prompt_text"]

        # Short term journal based on past daily trades
        if journal_entries:
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

        # Mid and long term memory blocks with relative time labels
        if weekly_memory:
            # Add relative time labels to weekly summaries (most recent first)
            labeled_weekly = []
            for i, memory_item in enumerate(
                weekly_memory[::-1]
            ):  # Reverse to get most recent first
                weeks_ago = i + 1
                if weeks_ago == 1:
                    label = "1 week ago"
                else:
                    label = f"{weeks_ago} weeks ago"

                # Handle both new dict format and old string format for backward compatibility
                if isinstance(memory_item, dict):
                    memory_text = f"{label}:\n{memory_item['summary']}"
                    memory_text += format_period_technical_indicators(memory_item['technical_stats'], "Weekly")
                else:
                    # Backward compatibility: old string format
                    memory_text = f"{label}:\n{memory_item}"

                labeled_weekly.append(memory_text)
            weekly_block = "Weekly memory (most recent first)\n" + "\n\n".join(
                labeled_weekly
            )
        else:
            weekly_block = "No weekly summaries yet."

        if monthly_memory:
            # Add relative time labels to monthly summaries (most recent first)
            labeled_monthly = []
            for i, memory_item in enumerate(
                monthly_memory[::-1]
            ):  # Reverse to get most recent first
                months_ago = i + 1
                if months_ago == 1:
                    label = "1 month ago"
                else:
                    label = f"{months_ago} months ago"

                # Handle both new dict format and old string format for backward compatibility
                if isinstance(memory_item, dict):
                    memory_text = f"{label}:\n{memory_item['summary']}"
                    memory_text += format_period_technical_indicators(memory_item['technical_stats'], "Monthly")
                else:
                    # Backward compatibility: old string format
                    memory_text = f"{label}:\n{memory_item}"

                labeled_monthly.append(memory_text)
            monthly_block = "Monthly memory (most recent first)\n" + "\n\n".join(
                labeled_monthly
            )
        else:
            monthly_block = "No monthly summaries yet."

        if quarterly_memory:
            # Add relative time labels to quarterly summaries (most recent first)
            labeled_quarterly = []
            for i, memory_item in enumerate(
                quarterly_memory[::-1]
            ):  # Reverse to get most recent first
                quarters_ago = i + 1
                if quarters_ago == 1:
                    label = "1 quarter ago"
                else:
                    label = f"{quarters_ago} quarters ago"

                # Handle both new dict format and old string format for backward compatibility
                if isinstance(memory_item, dict):
                    memory_text = f"{label}:\n{memory_item['summary']}"
                    memory_text += format_period_technical_indicators(memory_item['technical_stats'], "Quarterly")
                else:
                    # Backward compatibility: old string format
                    memory_text = f"{label}:\n{memory_item}"

                labeled_quarterly.append(memory_text)
            quarterly_block = "Quarterly memory (most recent first)\n" + "\n\n".join(
                labeled_quarterly
            )
        else:
            quarterly_block = "No quarterly summaries yet."

        if yearly_memory:
            # Add relative time labels to yearly summaries (most recent first)
            labeled_yearly = []
            for i, memory_item in enumerate(
                yearly_memory[::-1]
            ):  # Reverse to get most recent first
                years_ago = i + 1
                if years_ago == 1:
                    label = "1 year ago"
                else:
                    label = f"{years_ago} years ago"

                # Handle both new dict format and old string format for backward compatibility
                if isinstance(memory_item, dict):
                    memory_text = f"{label}:\n{memory_item['summary']}"
                    memory_text += format_period_technical_indicators(memory_item['technical_stats'], "Yearly")
                else:
                    # Backward compatibility: old string format
                    memory_text = f"{label}:\n{memory_item}"

                labeled_yearly.append(memory_text)
            yearly_block = "Yearly memory (most recent first)\n" + "\n\n".join(
                labeled_yearly
            )
        else:
            yearly_block = "No yearly summaries yet."

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

        # Update period stats for current date
        week_stats["strategy_return"] += daily_return
        week_stats["index_return"] += row["next_return_1d"]
        week_stats["days"] += 1
        if daily_return > 0:
            week_stats["wins"] += 1
        if decision == "BUY":
            week_stats["buys"] += 1
        elif decision == "HOLD":
            week_stats["holds"] += 1
        elif decision == "SELL":
            week_stats["sells"] += 1

        month_stats["strategy_return"] += daily_return
        month_stats["index_return"] += row["next_return_1d"]
        month_stats["days"] += 1
        if daily_return > 0:
            month_stats["wins"] += 1
        if decision == "BUY":
            month_stats["buys"] += 1
        elif decision == "HOLD":
            month_stats["holds"] += 1
        elif decision == "SELL":
            month_stats["sells"] += 1

        quarter_stats["strategy_return"] += daily_return
        quarter_stats["index_return"] += row["next_return_1d"]
        quarter_stats["days"] += 1
        if daily_return > 0:
            quarter_stats["wins"] += 1
        if decision == "BUY":
            quarter_stats["buys"] += 1
        elif decision == "HOLD":
            quarter_stats["holds"] += 1
        elif decision == "SELL":
            quarter_stats["sells"] += 1

        year_stats["strategy_return"] += daily_return
        year_stats["index_return"] += row["next_return_1d"]
        year_stats["days"] += 1
        if daily_return > 0:
            year_stats["wins"] += 1
        if decision == "BUY":
            year_stats["buys"] += 1
        elif decision == "HOLD":
            year_stats["holds"] += 1
        elif decision == "SELL":
            year_stats["sells"] += 1

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
            trade_data.update({
                "rsi_14": row.get("rsi_14"),
                "macd_line": row.get("macd_line"),
                "macd_signal": row.get("macd_signal"),
                "macd_histogram": row.get("macd_histogram"),
                "stoch_k": row.get("stoch_k"),
                "stoch_d": row.get("stoch_d"),
                "bb_position": row.get("bb_position"),
            })

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
    features_df = pd.read_csv(features_path, parse_dates=['date'])

    # Check if technical indicators are available
    has_rsi = 'rsi_14' in features_df.columns
    has_macd = 'macd_line' in features_df.columns
    has_stoch = 'stoch_k' in features_df.columns
    has_bb = 'bb_upper' in features_df.columns

    # Create technical indicators plots (conditionally)
    if has_rsi or has_macd or has_stoch or has_bb:
        print("Generating technical indicators plots...")
        technical_plot_path = os.path.join(plots_dir, f"{model_tag}_technical_indicators.png")
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
