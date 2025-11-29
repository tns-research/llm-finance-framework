# src/backtest.py

import os
import pandas as pd
import numpy as np
from .config import POSITION_MAP
from .config import ENABLE_FEELING_LOG, ENABLE_STRATEGIC_JOURNAL


def parse_response_text(response_text: str):
    lines = [ln.strip() for ln in str(response_text).splitlines() if ln.strip()]

    # Determine expected line count based on config
    expected_lines = 3  # decision, prob, explanation (always present)
    line_index = 3

    if ENABLE_STRATEGIC_JOURNAL:
        expected_lines += 1
    if ENABLE_FEELING_LOG:
        expected_lines += 1

    if len(lines) < expected_lines:
        raise ValueError(f"Expected at least {expected_lines} lines, got {len(lines)}")

    # Always parse first 3 lines
    decision = lines[0].upper()
    prob = float(lines[1])
    explanation = lines[2]

    # Parse strategic journal if enabled
    if ENABLE_STRATEGIC_JOURNAL:
        strategic_journal = lines[line_index]
        line_index += 1
    else:
        strategic_journal = "Strategic journal disabled in this configuration."

    # Parse feeling log if enabled
    if ENABLE_FEELING_LOG:
        feeling_log = lines[line_index]
    else:
        feeling_log = "Feeling log disabled in this configuration."

    if decision not in ("BUY", "HOLD", "SELL"):
        raise ValueError(f"Invalid decision word: {decision}")

    return decision, prob, explanation, strategic_journal, feeling_log


def backtest_model(parsed_df: pd.DataFrame) -> dict:
    df = parsed_df.sort_values("date").copy()
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    mean_ret = df["strategy_return"].mean()
    vol = df["strategy_return"].std()
    sharpe = mean_ret / vol if vol > 0 else np.nan

    df["equity"] = (1 + df["strategy_return"] / 100.0).cumprod()

    running_max = df["equity"].cummax()
    drawdown = df["equity"] / running_max - 1
    max_dd = drawdown.min()

    hit = (df["strategy_return"] > 0).mean()

    total_return = (df["equity"].iloc[-1] - 1.0) * 100.0

    return {
        "mean_return": mean_ret,
        "volatility": vol,
        "sharpe_like": sharpe,
        "max_drawdown": float(max_dd),
        "hit_rate": hit,
        "total_return": total_return,
    }


def save_parsed_results(results_path: str, rows: list) -> pd.DataFrame:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    return df
