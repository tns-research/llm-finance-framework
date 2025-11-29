# src/prompts.py

import os
import pandas as pd
from .config import PAST_RET_LAGS
from .config import MA20_WINDOW, VOL20_WINDOW, RET_5D_WINDOW, SHOW_DATE_TO_LLM
from .config import RSI_WINDOW, RSI_OVERBOUGHT, RSI_OVERSOLD


def build_summary_row(row: pd.Series) -> str:
    trend = "positive" if row["ma20_pct"] > 0 else "negative"
    vol_level = "low to moderate" if row["vol20_annualized"] < 20 else "elevated"
    rsi_level = ("overbought" if row["rsi_14"] > RSI_OVERBOUGHT
                else "oversold" if row["rsi_14"] < RSI_OVERSOLD
                else "neutral")

    return (
        f"The index has a {trend} 20 day total return and "
        f"{vol_level} volatility. RSI(14) is {row['rsi_14']:.1f} ({rsi_level}). "
        f"Recent 5 day return is {row['ret_5d']:.2f} percent."
    )


def row_to_prompt(row: pd.Series) -> str:
    past_rets = [row[f"ret_lag_{k}"] for k in range(PAST_RET_LAGS, 0, -1)]
    past_rets_str = ", ".join(f"{r:.2f}" for r in past_rets)

    header = ""
    if SHOW_DATE_TO_LLM:
        header = f"Date  {row['date'].strftime('%Y %m %d')}\n\n"

    text = (
        header
        + f"Past {PAST_RET_LAGS} daily returns in percent, most recent last\n"
        + f"{past_rets_str}\n\n"
        + f"{MA20_WINDOW} day total return  {row['ma20_pct']:.2f} percent\n"
        + f"{VOL20_WINDOW} day realized volatility  "
        + f"{row['vol20_annualized']:.2f} percent annualized\n"
        + f"{RET_5D_WINDOW} day total return  {row['ret_5d']:.2f} percent\n"
        + f"RSI({RSI_WINDOW})  {row['rsi_14']:.1f}\n\n"
        + "Summary\n"
        + f"{build_summary_row(row)}"
    )
    return text


# Need these constants from config again
from .config import MA20_WINDOW, VOL20_WINDOW, RET_5D_WINDOW


def build_prompts(features_path: str, prompts_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
    df = pd.read_csv(features_path, parse_dates=["date"])

    from .config import START_ROW

    if START_ROW is not None:
        print(f"START_ROW active : skipping first {START_ROW} rows")
        df = df.iloc[START_ROW:].reset_index(drop=True)
        print(f"New first date after START_ROW = {df.iloc[0]['date']}")

    prompts = []
    for _, row in df.iterrows():
        prompt_text = row_to_prompt(row)
        prompts.append(
            {
                "date": row["date"],
                "prompt_text": prompt_text,
                "next_return_1d": row["next_return_1d"],
            }
        )

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv(prompts_path, index=False)
    return prompts_df
