# src/prompts.py

import os
import pandas as pd
from . import config


def build_summary_row(row: pd.Series) -> str:
    trend = "positive" if row["ma20_pct"] > 0 else "negative"
    vol_level = "low to moderate" if row["vol20_annualized"] < 20 else "elevated"

    summary_parts = [
        f"The index has a {trend} 20 day total return and "
        f"{vol_level} volatility. Recent 5 day return is {row['ret_5d']:.2f} percent."
    ]

    # Add RSI information conditionally (RSI data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "rsi_14" in row.index and not pd.isna(row["rsi_14"]):
        rsi_level = ("overbought" if row["rsi_14"] > config.RSI_OVERBOUGHT
                    else "oversold" if row["rsi_14"] < config.RSI_OVERSOLD
                    else "neutral")
        summary_parts.append(f"RSI(14) is {row['rsi_14']:.1f} ({rsi_level}).")

    # Add MACD information conditionally (MACD data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "macd_histogram" in row.index and not pd.isna(row["macd_histogram"]):
        macd_signal = "bullish" if row["macd_histogram"] > 0 else "bearish"
        summary_parts.append(f"MACD histogram is {row['macd_histogram']:.3f} ({macd_signal}).")

    # Add Stochastic information conditionally (Stochastic data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "stoch_k" in row.index and not pd.isna(row["stoch_k"]):
        stoch_level = ("overbought" if row["stoch_k"] > config.STOCH_OVERBOUGHT
                      else "oversold" if row["stoch_k"] < config.STOCH_OVERSOLD
                      else "neutral")
        summary_parts.append(f"Stochastic %K is {row['stoch_k']:.1f} ({stoch_level}).")

    # Add Bollinger Bands information conditionally (BB data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "bb_position" in row.index and not pd.isna(row["bb_position"]):
        bb_pos = row["bb_position"]
        bb_desc = ("upper band" if bb_pos > 0.8 else "lower band" if bb_pos < 0.2 else "middle range")
        summary_parts.append(f"Price is at {bb_desc} of Bollinger Bands.")

    return " ".join(summary_parts)


def row_to_prompt(row: pd.Series) -> str:
    past_rets = [row[f"ret_lag_{k}"] for k in range(config.PAST_RET_LAGS, 0, -1)]
    past_rets_str = ", ".join(f"{r:.2f}" for r in past_rets)

    header = ""
    if config.SHOW_DATE_TO_LLM:
        header = f"Date  {row['date'].strftime('%Y %m %d')}\n\n"

    text_parts = [
        header,
        f"Past {config.PAST_RET_LAGS} daily returns in percent, most recent last",
        f"{past_rets_str}",
    ]

    # Historical Technical Indicators (only when enabled)
    if config.ENABLE_TECHNICAL_INDICATORS:
        # Historical RSI values
        if all(f"rsi_lag_{k}" in row.index for k in range(1, config.PAST_RET_LAGS + 1)):
            past_rsi = [row[f"rsi_lag_{k}"] for k in range(config.PAST_RET_LAGS, 0, -1)]
            past_rsi_str = ", ".join(f"{r:.1f}" for r in past_rsi if not pd.isna(r))
            if past_rsi_str:
                text_parts.extend([
                    "",
                    f"Past {config.PAST_RET_LAGS} days RSI(14) values, most recent last",
                    f"{past_rsi_str}"
                ])

        # Historical MACD histogram values
        if all(f"macd_hist_lag_{k}" in row.index for k in range(1, config.PAST_RET_LAGS + 1)):
            past_macd = [row[f"macd_hist_lag_{k}"] for k in range(config.PAST_RET_LAGS, 0, -1)]
            past_macd_str = ", ".join(f"{r:.3f}" for r in past_macd if not pd.isna(r))
            if past_macd_str:
                text_parts.extend([
                    "",
                    f"Past {config.PAST_RET_LAGS} days MACD histogram values, most recent last",
                    f"{past_macd_str}"
                ])

        # Historical Stochastic %K values
        if all(f"stoch_k_lag_{k}" in row.index for k in range(1, config.PAST_RET_LAGS + 1)):
            past_stoch = [row[f"stoch_k_lag_{k}"] for k in range(config.PAST_RET_LAGS, 0, -1)]
            past_stoch_str = ", ".join(f"{r:.1f}" for r in past_stoch if not pd.isna(r))
            if past_stoch_str:
                text_parts.extend([
                    "",
                    f"Past {config.PAST_RET_LAGS} days Stochastic %K values, most recent last",
                    f"{past_stoch_str}"
                ])

        # Historical Bollinger Band positions
        if all(f"bb_position_lag_{k}" in row.index for k in range(1, config.PAST_RET_LAGS + 1)):
            past_bb = [row[f"bb_position_lag_{k}"] for k in range(config.PAST_RET_LAGS, 0, -1)]
            past_bb_str = ", ".join(f"{r:.2f}" for r in past_bb if not pd.isna(r))
            if past_bb_str:
                text_parts.extend([
                    "",
                    f"Past {config.PAST_RET_LAGS} days Bollinger Band positions (0=lower, 1=upper), most recent last",
                    f"{past_bb_str}"
                ])

    text_parts.extend([
        "",
        f"{config.MA20_WINDOW} day total return  {row['ma20_pct']:.2f} percent",
        f"{config.VOL20_WINDOW} day realized volatility  {row['vol20_annualized']:.2f} percent annualized",
        f"{config.RET_5D_WINDOW} day total return  {row['ret_5d']:.2f} percent"
    ])

    # Add RSI conditionally (RSI data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "rsi_14" in row.index and not pd.isna(row["rsi_14"]):
        text_parts.append(f"RSI({config.RSI_WINDOW})  {row['rsi_14']:.1f}")

    # Add MACD conditionally (MACD data always available, but only show when technical indicators enabled)
    if config.ENABLE_TECHNICAL_INDICATORS and "macd_line" in row.index and not pd.isna(row["macd_line"]):
        text_parts.append(
            f"MACD({config.MACD_FAST},{config.MACD_SLOW},{config.MACD_SIGNAL}) Line: {row['macd_line']:.3f}, Signal: {row['macd_signal']:.3f}, Histogram: {row['macd_histogram']:.3f}"
        )

    if config.ENABLE_TECHNICAL_INDICATORS and "stoch_k" in row.index and not pd.isna(row["stoch_k"]):
        text_parts.append(
            f"Stochastic({config.STOCH_K},{config.STOCH_D}) %K: {row['stoch_k']:.1f}, %D: {row['stoch_d']:.1f}"
        )

    if config.ENABLE_TECHNICAL_INDICATORS and "bb_upper" in row.index and not pd.isna(row["bb_upper"]):
        text_parts.append(
            f"Bollinger Bands({config.BB_WINDOW},{config.BB_STD}) Upper: {row['bb_upper']:.2f}, Middle: {row['bb_middle']:.2f}, Lower: {row['bb_lower']:.2f}"
        )

    text_parts.extend([
        "",
        "Summary",
        f"{build_summary_row(row)}"
    ])

    text = "\n".join(text_parts)
    return text


# Need these constants from config again
from .config import MA20_WINDOW, VOL20_WINDOW, RET_5D_WINDOW


def build_prompts(features_path: str, prompts_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
    df = pd.read_csv(features_path, parse_dates=["date"])

    if config.START_ROW is not None:
        print(f"START_ROW active : skipping first {config.START_ROW} rows")
        df = df.iloc[config.START_ROW:].reset_index(drop=True)
        print(f"New first date after START_ROW = {df.iloc[0]['date']}")

    prompts = []
    for _, row in df.iterrows():
        prompt_text = row_to_prompt(row)
        prompts.append(
            {
                "date": row["date"],
                "prompt_text": prompt_text,
                "next_return_1d": row["next_return_1d"],
                # Include technical indicators (always calculated)
                "rsi_14": row.get("rsi_14"),
                "macd_line": row.get("macd_line"),
                "macd_signal": row.get("macd_signal"),
                "macd_histogram": row.get("macd_histogram"),
                "stoch_k": row.get("stoch_k"),
                "stoch_d": row.get("stoch_d"),
                "bb_position": row.get("bb_position"),
            }
        )

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv(prompts_path, index=False)
    return prompts_df
