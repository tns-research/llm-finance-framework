"""Deterministic unit tests for src/decision_analysis.py.

These lock the pure decision-analytics functions (post-outcome behavior,
position-duration stats, indicator-conditioned performance, and LLM/indicator
alignment) against small, hand-verifiable inputs so the module can be
refactored safely. The golden snapshot only covers report rendering, not these
compute paths.
"""

import numpy as np
import pandas as pd
import pytest

from src.decision_analysis import (
    analyze_decisions_after_outcomes,
    analyze_indicator_specific_performance,
    analyze_llm_indicator_alignment,
    analyze_position_duration_stats,
)


def _make_features_df(dates, ma20_pct) -> pd.DataFrame:
    """Build a features frame with every column the signal lambdas reference.

    All indicators except Momentum are held neutral so only `ma20_pct` drives
    the Momentum signal under test.
    """
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "rsi_14": np.full(n, 50.0),
            "macd_histogram": np.zeros(n),
            "stoch_k": np.full(n, 50.0),
            "close": np.full(n, 100.0),
            "bb_lower": np.full(n, 90.0),
            "bb_upper": np.full(n, 110.0),
            "ma20_pct": np.asarray(ma20_pct, dtype=float),
            "vol20_annualized": np.full(n, 0.2),
        }
    )


# ---------------------------------------------------------------------------
# analyze_decisions_after_outcomes
# ---------------------------------------------------------------------------


def test_decisions_after_outcomes_distribution_and_probs():
    df = pd.DataFrame(
        {
            # first row dropped (no previous_return)
            "previous_return": [np.nan, 1.0, -1.0, 2.0, -2.0],
            "decision": ["BUY", "BUY", "SELL", "HOLD", "SELL"],
            "prob": [0.5, 0.6, 0.4, 0.7, 0.3],
        }
    )
    res = analyze_decisions_after_outcomes(df)

    assert res["total_decisions"] == 4
    assert res["total_wins"] == 2
    assert res["total_losses"] == 2
    assert res["total_neutral"] == 0

    wins = res["decisions_after_wins"]
    assert wins["BUY"] == pytest.approx(50.0)
    assert wins["HOLD"] == pytest.approx(50.0)
    assert wins["SELL"] == pytest.approx(0.0)
    assert wins["mean_prob"] == pytest.approx(0.65)

    losses = res["decisions_after_losses"]
    assert losses["SELL"] == pytest.approx(100.0)
    assert losses["mean_prob"] == pytest.approx(0.35)

    # Both wins and losses present -> chi-square contingency runs.
    assert res["chi_square_test"] is not None
    assert "p_value" in res["chi_square_test"]


def test_decisions_after_outcomes_neutral_and_no_losses():
    df = pd.DataFrame(
        {
            "previous_return": [np.nan, 0.0, 0.0, 1.0],
            "decision": ["BUY", "HOLD", "HOLD", "BUY"],
            "prob": [0.5, 0.5, 0.5, 0.5],
        }
    )
    res = analyze_decisions_after_outcomes(df)

    assert res["total_decisions"] == 3
    assert res["total_neutral"] == 2
    assert res["total_wins"] == 1
    assert res["total_losses"] == 0
    # No losses -> the losses bucket and the chi-square test are both absent.
    assert res["decisions_after_losses"] is None
    assert res["chi_square_test"] is None


def test_decisions_after_outcomes_no_history_returns_error():
    df = pd.DataFrame(
        {
            "previous_return": [np.nan, np.nan],
            "decision": ["BUY", "HOLD"],
            "prob": [0.5, 0.5],
        }
    )
    assert "error" in analyze_decisions_after_outcomes(df)


# ---------------------------------------------------------------------------
# analyze_position_duration_stats
# ---------------------------------------------------------------------------


def test_position_duration_runs_and_longest_streak():
    # i==0 starts a run regardless of position_changed; a True flag starts the
    # next run; continuations overwrite the open run's duration in place.
    df = pd.DataFrame(
        {
            "position_changed": [False, False, True, False],
            "position_duration": [1, 2, 1, 2],
            "decision": ["BUY", "BUY", "SELL", "SELL"],
        }
    )
    res = analyze_position_duration_stats(df)

    assert res["total_position_changes"] == 1
    assert res["average_position_duration"] == pytest.approx(1.5)
    assert res["median_position_duration"] == pytest.approx(1.5)
    assert res["max_position_duration"] == 2

    assert res["BUY_stats"]["count"] == 1
    assert res["BUY_stats"]["mean_duration"] == pytest.approx(2.0)
    assert res["SELL_stats"]["count"] == 1
    assert res["SELL_stats"]["max_duration"] == 2
    # No HOLD run was ever opened.
    assert res["HOLD_stats"] is None

    # BUY and SELL tie at duration 2; idxmax picks the first (BUY).
    assert res["longest_streak"]["decision"] == "BUY"
    assert res["longest_streak"]["duration"] == 2


# ---------------------------------------------------------------------------
# analyze_indicator_specific_performance
# ---------------------------------------------------------------------------


def test_indicator_performance_momentum_metrics():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    parsed = pd.DataFrame(
        {
            "date": dates,
            "strategy_return": [0.1, -0.2, 0.3, -0.4],
            "decision": ["BUY", "SELL", "BUY", "SELL"],
        }
    )
    # ma20_pct > 0 -> Momentum signals on days 0 and 2.
    features = _make_features_df(dates, ma20_pct=[1.0, -1.0, 2.0, -2.0])

    res = analyze_indicator_specific_performance(parsed, features)
    m = res["Momentum"]

    assert m["signal_days"] == 2
    assert m["total_days"] == 4
    assert m["signal_percentage"] == pytest.approx(50.0)
    # signal: (1.1 * 1.3) - 1 ; no-signal: (0.8 * 0.6) - 1
    assert m["llm_return_when_signal"] == pytest.approx(0.43)
    assert m["llm_return_when_no_signal"] == pytest.approx(-0.52)
    assert m["signal_win_rate"] == pytest.approx(1.0)
    assert m["no_signal_win_rate"] == pytest.approx(0.0)
    assert m["performance_differential"] == pytest.approx(0.95)
    assert m["signal_avg_daily_return"] == pytest.approx(0.2)
    assert m["no_signal_avg_daily_return"] == pytest.approx(-0.3)


def test_indicator_performance_no_overlapping_dates():
    parsed = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "strategy_return": [0.1, 0.2, 0.3],
            "decision": ["BUY", "BUY", "BUY"],
        }
    )
    features = _make_features_df(
        pd.date_range("2021-01-01", periods=3, freq="D"), ma20_pct=[1.0, 1.0, 1.0]
    )
    res = analyze_indicator_specific_performance(parsed, features)
    assert "error" in res["Momentum"]


# ---------------------------------------------------------------------------
# analyze_llm_indicator_alignment
# ---------------------------------------------------------------------------


def test_llm_indicator_alignment_momentum():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    parsed = pd.DataFrame(
        {
            "date": dates,
            "decision": ["BUY", "BUY", "SELL", "HOLD"],
            "strategy_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    # ma20_pct > 0 -> 2 bullish, 2 bearish signals.
    features = _make_features_df(dates, ma20_pct=[1.0, 1.0, -1.0, -1.0])

    res = analyze_llm_indicator_alignment(parsed, features)
    m = res["Momentum"]

    assert m["bullish_signals"] == 2
    assert m["bearish_signals"] == 2
    assert m["llm_buy_signals"] == 2
    assert m["llm_sell_signals"] == 1
    # buy_agreements = min(2, 2) = 2 ; sell_agreements = min(2, 1) = 1
    assert m["agreements"] == 3
    assert m["total_signals"] == 4
    assert m["alignment_rate"] == pytest.approx(0.75)
