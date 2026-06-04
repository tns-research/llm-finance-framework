# src/backtest.py

import os

import numpy as np
import pandas as pd


def parse_response_text(response_text: str):
    # Imported at call time (not module level) so tests can patch these flags
    # on src.config and have the patched values take effect here.
    from .config import (
        ENABLE_CHAIN_OF_THOUGHT,
        ENABLE_FEELING_LOG,
        ENABLE_STRATEGIC_JOURNAL,
    )

    lines = [ln.strip() for ln in str(response_text).splitlines() if ln.strip()]

    # Calculate expected lines based on enabled features
    expected_lines = 3  # Base count: decision, prob, explanation
    if ENABLE_CHAIN_OF_THOUGHT:
        expected_lines += 1
    if ENABLE_STRATEGIC_JOURNAL:
        expected_lines += 1
    if ENABLE_FEELING_LOG:
        expected_lines += 1

    if len(lines) < expected_lines:
        error_msg = (
            f"Expected at least {expected_lines} lines for current config, got {len(lines)}. "
            f"Config: ENABLE_CHAIN_OF_THOUGHT={ENABLE_CHAIN_OF_THOUGHT}, "
            f"ENABLE_STRATEGIC_JOURNAL={ENABLE_STRATEGIC_JOURNAL}, "
            f"ENABLE_FEELING_LOG={ENABLE_FEELING_LOG}. "
            f"Lines received: {lines[:10]}..."  # Show first 10 lines for debugging
        )
        raise ValueError(error_msg)

    # CONDITIONAL PARSING: Parse based on chain of thought flag
    line_index = 0

    if ENABLE_CHAIN_OF_THOUGHT:
        chain_of_thought = lines[line_index]
        line_index += 1
    else:
        chain_of_thought = "Chain of thought reasoning disabled."

    # Decision, probability, explanation (indices shift if chain of thought enabled)
    try:
        decision = lines[line_index].upper()
        prob = float(lines[line_index + 1])
        explanation = lines[line_index + 2]
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Failed to parse decision/probability/explanation at lines {line_index}-{line_index+2}: {e}. Lines: {lines}"
        )
    line_index += 3

    # Parse strategic journal if enabled
    if ENABLE_STRATEGIC_JOURNAL:
        try:
            strategic_journal = lines[line_index]
            line_index += 1
        except IndexError:
            raise ValueError(
                f"Missing strategic journal line at index {line_index}. Expected lines: {expected_lines}, got {len(lines)}"
            )
    else:
        strategic_journal = "Strategic journal disabled in this configuration."

    # Parse feeling log if enabled
    if ENABLE_FEELING_LOG:
        try:
            feeling_log = lines[line_index]
        except IndexError:
            raise ValueError(
                f"Missing feeling log line at index {line_index}. Expected lines: {expected_lines}, got {len(lines)}"
            )
    else:
        feeling_log = "Feeling log disabled in this configuration."

    # Validation
    if decision not in ("BUY", "HOLD", "SELL"):
        raise ValueError(f"Invalid decision word: {decision}")

    return decision, prob, explanation, chain_of_thought, strategic_journal, feeling_log


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
