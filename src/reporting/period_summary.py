# src/reporting/period_summary.py
"""Period-summary text generation (compute + LLM journal).

Moved verbatim out of the old monolithic src/reporting.py (pure structural
split, no logic change). Re-exported via the reporting package __init__.
"""

import logging
from typing import Union

import pandas as pd

from ..config import (
    DEBUG_SHOW_FULL_PROMPT,
    JOURNAL_SYSTEM_PROMPT,
    SHOW_DATE_TO_LLM,
    USE_DUMMY_MODEL,
)
from ..memory_classes import PeriodStats
from ..model_router import generate_response

logger = logging.getLogger(__name__)


def make_empty_stats():
    """
    Helper to initialize per period statistics.
    """
    return {
        "strategy_return": 0.0,
        "index_return": 0.0,
        "days": 0,
        "wins": 0,
        "buys": 0,
        "holds": 0,
        "sells": 0,
    }


def build_period_summary(period_label: str, end_date, stats: dict) -> str:
    """
    Build a compact text summary for a completed period (week, month, quarter, year).

    period_label  "Week", "Month", "Quarter", "Year"
    end_date      last trading date of the period (pandas Timestamp)
    stats         dict from make_empty_stats, aggregated over that period
    """
    if stats["days"] == 0:
        return f"{period_label} ending {end_date.strftime('%Y-%m-%d')}  no trading activity recorded."

    strat_ret = stats["strategy_return"]
    idx_ret = stats["index_return"]
    days = stats["days"]
    win_rate = (stats["wins"] / days) * 100.0 if days > 0 else 0.0

    explanation = (
        f"{period_label} ending {end_date.strftime('%Y-%m-%d')}. "
        f"Market total return over the period  {idx_ret:.2f} percent. "
        f"Strategy total return  {strat_ret:.2f} percent over {days} trading days."
    )

    journal = (
        f"During this period you took BUY {stats['buys']} times, "
        f"HOLD {stats['holds']} times, SELL {stats['sells']} times, "
        f"with a win rate of {win_rate:.1f} percent on daily returns. "
        "Reflect on whether your positioning matched the prevailing trend and volatility, "
        "and whether you managed risk consistently."
    )

    if strat_ret > idx_ret:
        feeling = (
            "Feeling confident and satisfied, having outperformed the index, "
            "but still cautious about overconfidence."
        )
    elif strat_ret > 0:
        feeling = (
            "Feeling cautiously positive. You earned a positive return but did "
            "not beat the index, so there is room for improvement in timing and sizing."
        )
    else:
        feeling = (
            "Feeling dissatisfied and reflective. Losses highlight the need to improve "
            "signal quality and risk management, especially during volatile regimes."
        )

    return (
        f"{period_label} ending {end_date.strftime('%Y-%m-%d')}\n"
        f"{explanation}\n"
        f"Strategic journal  {journal}\n"
        f"Feeling  {feeling}"
    )


def compute_period_technical_stats(
    features_df: pd.DataFrame, start_date, end_date
) -> dict:
    """
    Compute technical indicator statistics for a time period.

    This function calculates period-level summaries of technical indicators
    to provide rich context to the LLM for memory/reflection purposes.
    Only includes statistics when data is available and valid.

    Args:
        features_df: DataFrame with technical indicator columns
        start_date: Start date of the period (inclusive)
        end_date: End date of the period (inclusive)

    Returns:
        dict: Technical indicator statistics for the period
    """
    # Filter data for the period
    period_data = features_df[
        (features_df["date"] >= start_date) & (features_df["date"] <= end_date)
    ].copy()

    stats = {}

    # RSI Statistics
    if "rsi_14" in period_data.columns:
        rsi_valid = period_data["rsi_14"].dropna()
        if len(rsi_valid) > 0:
            stats["rsi_avg"] = rsi_valid.mean()
            stats["rsi_overbought_pct"] = (rsi_valid > 70).sum() / len(rsi_valid) * 100
            stats["rsi_oversold_pct"] = (rsi_valid < 30).sum() / len(rsi_valid) * 100
            stats["rsi_min"] = rsi_valid.min()
            stats["rsi_max"] = rsi_valid.max()

    # MACD Statistics
    if "macd_histogram" in period_data.columns:
        hist_valid = period_data["macd_histogram"].dropna()
        if len(hist_valid) > 0:
            stats["macd_bullish_pct"] = (hist_valid > 0).sum() / len(hist_valid) * 100
            stats["macd_avg_histogram"] = hist_valid.mean()
            # Count signal line crossovers (histogram changes sign)
            hist_sign_changes = ((hist_valid > 0) != (hist_valid.shift(1) > 0)).sum()
            stats["macd_crossovers"] = hist_sign_changes

    # Stochastic Statistics
    if "stoch_k" in period_data.columns:
        stoch_valid = period_data["stoch_k"].dropna()
        if len(stoch_valid) > 0:
            stats["stoch_overbought_pct"] = (
                (stoch_valid > 80).sum() / len(stoch_valid) * 100
            )
            stats["stoch_oversold_pct"] = (
                (stoch_valid < 20).sum() / len(stoch_valid) * 100
            )
            stats["stoch_min"] = stoch_valid.min()
            stats["stoch_max"] = stoch_valid.max()

    # Bollinger Bands Statistics
    if "bb_position" in period_data.columns:
        bb_valid = period_data["bb_position"].dropna()
        if len(bb_valid) > 0:
            # Near upper band (position > 0.95) or near lower band (position < 0.05)
            stats["bb_upper_touch_pct"] = (bb_valid > 0.95).sum() / len(bb_valid) * 100
            stats["bb_lower_touch_pct"] = (bb_valid < 0.05).sum() / len(bb_valid) * 100
            stats["bb_avg_position"] = bb_valid.mean()

    # Add current/latest values for short periods or when aggregated stats are not meaningful
    if len(period_data) > 0:
        last_row = period_data.iloc[-1]

        # Current RSI value
        if "rsi_14" in period_data.columns and not pd.isna(last_row.get("rsi_14")):
            stats["rsi_current"] = last_row["rsi_14"]

        # Current MACD values
        macd_cols = ["macd_line", "macd_signal", "macd_histogram"]
        if all(col in period_data.columns for col in macd_cols):
            macd_values = [last_row.get(col) for col in macd_cols]
            if not any(pd.isna(macd_values)):
                stats["macd_current"] = {
                    "line": last_row["macd_line"],
                    "signal": last_row["macd_signal"],
                    "histogram": last_row["macd_histogram"],
                }

        # Current Stochastic values
        stoch_cols = ["stoch_k", "stoch_d"]
        if all(col in period_data.columns for col in stoch_cols):
            stoch_values = [last_row.get(col) for col in stoch_cols]
            if not any(pd.isna(stoch_values)):
                stats["stoch_current"] = {
                    "k": last_row["stoch_k"],
                    "d": last_row["stoch_d"],
                }

        # Current Bollinger Band position
        if "bb_position" in period_data.columns and not pd.isna(
            last_row.get("bb_position")
        ):
            stats["bb_current_position"] = last_row["bb_position"]

    return stats


def format_period_technical_indicators(technical_stats: dict, period_name: str) -> str:
    """Format aggregated technical indicators for memory display."""
    if not technical_stats:
        return ""

    lines = []

    # RSI
    if "rsi_avg" in technical_stats:
        rsi_parts = [f"Average {technical_stats['rsi_avg']:.1f}"]
        if "rsi_overbought_pct" in technical_stats:
            rsi_parts.append(f"{technical_stats['rsi_overbought_pct']:.0f}% overbought")
        if "rsi_oversold_pct" in technical_stats:
            rsi_parts.append(f"{technical_stats['rsi_oversold_pct']:.0f}% oversold")
        if "rsi_min" in technical_stats and "rsi_max" in technical_stats:
            rsi_parts.append(
                f"range {technical_stats['rsi_min']:.1f}-{technical_stats['rsi_max']:.1f}"
            )
        lines.append(f"RSI(14): {', '.join(rsi_parts)}")
    elif "rsi_current" in technical_stats:
        # Fallback for short periods - show current value
        rsi_val = technical_stats["rsi_current"]
        status = "neutral"
        if rsi_val > 70:
            status = "overbought"
        elif rsi_val < 30:
            status = "oversold"
        lines.append(f"RSI(14): {rsi_val:.1f} ({status})")

    # MACD
    if "macd_bullish_pct" in technical_stats:
        macd_parts = [f"{technical_stats['macd_bullish_pct']:.0f}% bullish periods"]
        if "macd_avg_histogram" in technical_stats:
            macd_parts.append(
                f"avg histogram {technical_stats['macd_avg_histogram']:.3f}"
            )
        if (
            "macd_crossovers" in technical_stats
            and technical_stats["macd_crossovers"] > 0
        ):
            macd_parts.append(f"{technical_stats['macd_crossovers']} crossovers")
        lines.append(f"MACD: {', '.join(macd_parts)}")
    elif "macd_current" in technical_stats:
        # Fallback for short periods - show current values
        macd = technical_stats["macd_current"]
        signal = "bullish" if macd["histogram"] > 0 else "bearish"
        lines.append(
            f"MACD: {macd['line']:.2f}/{macd['signal']:.2f}/{macd['histogram']:.3f} ({signal})"
        )

    # Stochastic
    if "stoch_overbought_pct" in technical_stats:
        stoch_parts = [
            f"{technical_stats['stoch_overbought_pct']:.0f}% overbought days"
        ]
        if "stoch_oversold_pct" in technical_stats:
            stoch_parts.append(
                f"{technical_stats['stoch_oversold_pct']:.0f}% oversold days"
            )
        lines.append(f"Stochastic: {', '.join(stoch_parts)}")
    elif "stoch_current" in technical_stats:
        # Fallback for short periods - show current values
        stoch = technical_stats["stoch_current"]
        status = "neutral"
        if stoch["k"] > 80:
            status = "overbought"
        elif stoch["k"] < 20:
            status = "oversold"
        lines.append(f"Stochastic: {stoch['k']:.1f}/{stoch['d']:.1f} ({status})")

    # Bollinger Bands
    if "bb_avg_position" in technical_stats:
        bb_parts = [f"Avg position {technical_stats['bb_avg_position']:.2f}"]
        if (
            "bb_upper_touch_pct" in technical_stats
            and "bb_lower_touch_pct" in technical_stats
        ):
            total_touches = (
                technical_stats["bb_upper_touch_pct"]
                + technical_stats["bb_lower_touch_pct"]
            )
            bb_parts.append(f"band touches {total_touches:.0f}%")
        lines.append(f"Bollinger Bands: {', '.join(bb_parts)}")
    elif "bb_current_position" in technical_stats:
        # Fallback for short periods - show current position
        pos = technical_stats["bb_current_position"]
        location = "middle"
        if pos > 0.8:
            location = "upper band"
        elif pos < 0.2:
            location = "lower band"
        lines.append(f"Bollinger Bands: Position {pos:.2f} ({location})")

    if lines:
        return f"\n\n{period_name} technical indicators:\n" + "\n".join(lines) + "\n"
    return ""


def _fallback_period_summary(
    period_label,
    end_date,
    strat_ret,
    idx_ret,
    days,
    wins,
    buys,
    holds,
    sells,
    technical_stats,
):
    """Template summary used in dummy mode or when no router model is set."""
    win_rate = (wins / days) * 100.0 if days > 0 else 0.0
    edge = strat_ret - idx_ret
    outperform_word = "outperformed" if edge > 0 else "underperformed"

    # Respect SHOW_DATE_TO_LLM setting in fallback summaries
    if SHOW_DATE_TO_LLM:
        period_header = f"{period_label} ending {end_date.strftime('%Y-%m-%d')}"
        explanation_date = f"{period_label} ending {end_date.strftime('%Y-%m-%d')}. "
    else:
        period_header = f"{period_label} summary (date hidden)"
        explanation_date = f"{period_label} summary. "

    explanation = (
        f"{explanation_date}"
        f"Market total return  {idx_ret:.2f} percent. "
        f"Strategy total return  {strat_ret:.2f} percent. "
        f"The strategy {outperform_word} the index by {edge:.2f} percent over {days} days."
    )

    # Add technical analysis to explanation if available
    if technical_stats:
        if "rsi_avg" in technical_stats:
            explanation += f" RSI averaged {technical_stats['rsi_avg']:.1f} with {technical_stats['rsi_overbought_pct']:.1f}% overbought days."
        if "macd_bullish_pct" in technical_stats:
            explanation += f" MACD was bullish {technical_stats['macd_bullish_pct']:.1f}% of the time."
        if "stoch_overbought_pct" in technical_stats:
            explanation += f" Stochastic showed {technical_stats['stoch_overbought_pct']:.1f}% overbought conditions."
        if "bb_upper_touch_pct" in technical_stats:
            explanation += f" Price touched Bollinger upper band on {technical_stats['bb_upper_touch_pct']:.1f}% of days."

    journal = (
        f"During this period you traded BUY {buys} times, HOLD {holds} times, SELL {sells} times, "
        f"with a win rate of {win_rate:.1f} percent on daily returns. "
    )

    # Add technical analysis to strategic journal if available
    if technical_stats:
        journal += "Technical indicators provided "
        tech_signals = []

        if "rsi_avg" in technical_stats:
            rsi_signal = (
                "bullish signals"
                if technical_stats["rsi_avg"] < 50
                else "bearish signals"
            )
            tech_signals.append(f"RSI {rsi_signal}")

        if "macd_bullish_pct" in technical_stats:
            macd_signal = (
                "mostly bullish momentum"
                if technical_stats["macd_bullish_pct"] > 50
                else "mostly bearish momentum"
            )
            tech_signals.append(f"MACD showing {macd_signal}")

        if "stoch_overbought_pct" in technical_stats:
            stoch_signal = (
                "frequent overbought conditions"
                if technical_stats["stoch_overbought_pct"] > 20
                else "limited overbought conditions"
            )
            tech_signals.append(f"Stochastic with {stoch_signal}")

        if "bb_upper_touch_pct" in technical_stats:
            bb_signal = (
                "frequent band touches"
                if technical_stats["bb_upper_touch_pct"]
                + technical_stats["bb_lower_touch_pct"]
                > 10
                else "rare band extremes"
            )
            tech_signals.append(f"Bollinger Bands with {bb_signal}")

        if tech_signals:
            journal += f"mixed signals: {', '.join(tech_signals)}. "
        else:
            journal += "consistent signals. "

    journal += (
        "Reflect on whether your positioning matched the prevailing trend and volatility, "
        "and whether your risk management was consistent."
    )

    feeling = (
        "Feeling cautiously reflective about this period. Use the results to refine your process "
        "without becoming overconfident or discouraged."
    )

    return (
        f"{period_header}\n"
        f"Explanation: {explanation}\n"
        f"Strategic journal: {journal}\n"
        f"Feeling log: {feeling}"
    )


def _build_period_prompt(
    period_label,
    end_date,
    strat_ret,
    idx_ret,
    days,
    wins,
    buys,
    holds,
    sells,
    technical_stats,
):
    """Assemble the user message handed to the journal LLM."""
    win_rate = (wins / days) * 100.0 if days > 0 else 0.0
    edge = strat_ret - idx_ret

    # When SHOW_DATE_TO_LLM is False, we send NO date information to the LLM at all
    # This prevents the LLM from including dates in its summaries
    if SHOW_DATE_TO_LLM:
        date_info = f"- End date  {end_date.strftime('%Y-%m-%d')}\n"
        period_desc = f"You are summarizing a completed {period_label}.\n\n"
    else:
        date_info = ""  # No date information sent to LLM in anonymized mode
        period_desc = f"You are summarizing a completed {period_label} (time period anonymized).\n\n"

    # Add technical indicators section if available
    technical_info = ""
    if technical_stats:
        logger.debug("LLM received technical_stats = %s", technical_stats)
        technical_info = "\nTechnical indicators summary for this period:\n"

        if "rsi_avg" in technical_stats:
            technical_info += (
                f"- RSI(14): Average {technical_stats['rsi_avg']:.1f}, "
                f"{technical_stats['rsi_overbought_pct']:.1f}% overbought days (>70), "
                f"{technical_stats['rsi_oversold_pct']:.1f}% oversold days (<30), "
                f"range {technical_stats['rsi_min']:.1f}-{technical_stats['rsi_max']:.1f}\n"
            )

        if "macd_bullish_pct" in technical_stats:
            technical_info += (
                f"- MACD(12,26,9): {technical_stats['macd_bullish_pct']:.1f}% bullish periods, "
                f"avg histogram {technical_stats['macd_avg_histogram']:.3f}, "
                f"{technical_stats.get('macd_crossovers', 0)} signal crossovers\n"
            )

        if "stoch_overbought_pct" in technical_stats:
            technical_info += (
                f"- Stochastic(14,3): {technical_stats['stoch_overbought_pct']:.1f}% overbought days (>80), "
                f"{technical_stats['stoch_oversold_pct']:.1f}% oversold days (<20), "
                f"range {technical_stats['stoch_min']:.1f}-{technical_stats['stoch_max']:.1f}\n"
            )

        if "bb_upper_touch_pct" in technical_stats:
            technical_info += (
                f"- Bollinger Bands(20,2): {technical_stats['bb_upper_touch_pct']:.1f}% days touched upper band, "
                f"{technical_stats['bb_lower_touch_pct']:.1f}% touched lower band, "
                f"avg position {technical_stats['bb_avg_position']:.2f}\n"
            )
    else:
        logger.debug("LLM received technical_stats = None/empty")

    user_message = (
        f"{period_desc}"
        f"Period information\n"
        f"{date_info}"
        f"- Trading days in period  {days}\n"
        f"- Strategy total return over the period  {strat_ret:.2f} percent\n"
        f"- Index total return over the period  {idx_ret:.2f} percent\n"
        f"- Difference strategy minus index  {edge:.2f} percent\n"
        f"- Winning days (positive strategy return)  {wins} out of {days}\n"
        f"- Number of BUY decisions  {buys}\n"
        f"- Number of HOLD decisions  {holds}\n"
        f"- Number of SELL decisions  {sells}\n"
        f"- Daily win rate  {win_rate:.1f} percent"
        f"{technical_info}\n\n"
        "Write a reflection journal for this period. Do not include any dates or calendar references. Use only the numerical information provided."
    )

    return user_message


def generate_llm_period_summary(
    period_label: str,
    end_date,
    stats: Union[dict, PeriodStats],
    router_model: str,
    model_tag: str,
    technical_stats: dict = None,
) -> str:
    """
    Use the LLM itself to write a weekly, monthly, quarterly or yearly journal
    based on aggregated stats for that period.

    If USE_DUMMY_MODEL or router_model is None, we fall back to a simple
    template summary.

    Args:
        period_label: "Week", "Month", "Quarter", or "Year"
        end_date: End date of the period
        stats: Statistics dict or PeriodStats object
        router_model: Model identifier for LLM calls
        model_tag: Model tag for identification
        technical_stats: Optional technical indicators data
    """
    # Convert PeriodStats to dict for backward compatibility
    if hasattr(stats, "to_dict"):
        stats = stats.to_dict()

    if stats["days"] == 0:
        return f"{period_label} ending {end_date.strftime('%Y-%m-%d')}  no trading activity recorded."

    strat_ret = stats["strategy_return"]
    idx_ret = stats["index_return"]
    days = stats["days"]
    wins = stats["wins"]
    buys = stats["buys"]
    holds = stats["holds"]
    sells = stats["sells"]

    # Fallback template if we are in dummy mode or no router model
    if USE_DUMMY_MODEL or router_model is None:
        return _fallback_period_summary(
            period_label,
            end_date,
            strat_ret,
            idx_ret,
            days,
            wins,
            buys,
            holds,
            sells,
            technical_stats,
        )

    # If we are here, we can call the real LLM via OpenRouter
    edge = strat_ret - idx_ret
    user_message = _build_period_prompt(
        period_label,
        end_date,
        strat_ret,
        idx_ret,
        days,
        wins,
        buys,
        holds,
        sells,
        technical_stats,
    )

    try:
        logger.debug(
            "generate_llm_period_summary called for %s, DEBUG_SHOW_FULL_PROMPT=%s",
            period_label,
            DEBUG_SHOW_FULL_PROMPT,
        )
        if DEBUG_SHOW_FULL_PROMPT:
            debug_block = (
                "===== JOURNAL SYSTEM PROMPT =====\n"
                f"{JOURNAL_SYSTEM_PROMPT()}\n\n"
                "===== JOURNAL USER MESSAGE =====\n"
                f"{user_message}\n"
            )
            logger.debug(
                "FULL JOURNAL PROMPT SENT TO MODEL %s:\n%s", model_tag, debug_block
            )

        response_text = generate_response(
            router_model, JOURNAL_SYSTEM_PROMPT(), user_message
        )
        clean = str(response_text).strip()

        # Optional: small numeric recap so the memory always carries the hard data
        if SHOW_DATE_TO_LLM:
            period_id = f"{period_label} ending {end_date.strftime('%Y-%m-%d')}"
        else:
            period_id = f"{period_label} summary (date hidden)"

        header = (
            f"{period_id}\n"
            f"Stats  strategy {strat_ret:.2f} percent, "
            f"index {idx_ret:.2f} percent, "
            f"edge (strategy minus index) {edge:.2f} percent, "
            f"days {days}, wins {wins}, "
            f"BUY {buys}, HOLD {holds}, SELL {sells}.\n\n"
        )

        wrapped = header + clean

        if DEBUG_SHOW_FULL_PROMPT:
            logger.debug("JOURNAL MODEL OUTPUT:\n%s", wrapped)

        return wrapped

    except Exception as e:
        logger.warning(
            "Failed to generate %s journal with LLM for model %s: %s",
            period_label,
            model_tag,
            e,
        )
        # Fallback to simple template
        win_rate = (wins / days) * 100.0 if days > 0 else 0.0

        # Respect SHOW_DATE_TO_LLM setting in error fallback
        if SHOW_DATE_TO_LLM:
            period_header = f"{period_label} ending {end_date.strftime('%Y-%m-%d')}"
            explanation_date = (
                f"{period_label} ending {end_date.strftime('%Y-%m-%d')}. "
            )
        else:
            period_header = f"{period_label} summary (date hidden)"
            explanation_date = f"{period_label} summary. "

        explanation = (
            f"{explanation_date}"
            f"Market total return  {idx_ret:.2f} percent. "
            f"Strategy total return  {strat_ret:.2f} percent over {days} days."
        )
        journal = (
            f"BUY {buys}, HOLD {holds}, SELL {sells}, win rate {win_rate:.1f} percent. "
            "LLM journal generation failed, using fallback summary."
        )
        feeling = "Feeling neutral due to technical issues."

        return (
            f"{period_header}\n"
            f"Explanation: {explanation}\n"
            f"Strategic journal: {journal}\n"
            f"Feeling log: {feeling}"
        )
