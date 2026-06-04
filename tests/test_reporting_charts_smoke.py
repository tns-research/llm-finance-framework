# Smoke tests for the matplotlib chart functions in ``src/reporting.py``.
#
# These are a prerequisite for splitting reporting.py: the golden snapshot
# embeds charts by PNG *reference*, so it does NOT detect a broken chart. Each test
# below feeds a small, fully deterministic fixture (no network, no RNG, no model)
# to a chart function and asserts it produces a non-empty PNG without raising.
#
# This guards the pure-move refactor: the functions must keep producing a file
# after they are relocated into the reporting/ package.

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from src.baselines import calculate_category_performance
from src.decision_analysis import analyze_indicator_specific_performance
from src.reporting import (
    create_calibration_by_decision_plot,
    create_calibration_plot,
    create_category_performance_plot,
    create_indicator_performance_plot,
    create_risk_analysis_chart,
    create_rolling_performance_chart,
    create_rsi_performance_analysis,
    create_technical_indicators_plot,
    create_technical_indicators_timeline,
    generate_calibration_analysis_report,
)

N = 90


def _dates():
    return pd.date_range("2022-01-03", periods=N, freq="B")


def build_parsed_df() -> pd.DataFrame:
    """Deterministic parsed decisions covering BUY/HOLD/SELL with calibration data."""
    idx = np.arange(N)
    decisions = np.array(["BUY", "HOLD", "SELL"])[idx % 3]
    # Probabilities spread across the 0..1 calibration bins, all in (0, 1).
    prob = 0.05 + 0.9 * (0.5 + 0.5 * np.sin(idx / 5.0))
    # Small alternating returns with both signs so win-rate and VaR are non-trivial.
    strat = 0.01 * np.sin(idx / 3.0) + 0.002 * np.cos(idx / 7.0)
    nxt = 0.008 * np.sin(idx / 3.0 + 0.3)
    return pd.DataFrame(
        {
            "date": _dates(),
            "decision": decisions,
            "prob": prob,
            "strategy_return": strat,
            "next_return_1d": nxt,
        }
    )


def build_features_df() -> pd.DataFrame:
    """Deterministic technical-indicator frame with every column the charts read."""
    idx = np.arange(N)
    close = 400.0 + np.cumsum(0.5 * np.sin(idx / 4.0))
    rsi = 50.0 + 35.0 * np.sin(idx / 6.0)  # swings into <30 and >70 zones
    macd_line = np.sin(idx / 5.0)
    macd_signal = np.sin(idx / 5.0 - 0.4)
    stoch_k = 50.0 + 45.0 * np.sin(idx / 4.5)  # swings into <20 and >80 zones
    bb_middle = pd.Series(close).rolling(20, min_periods=1).mean().to_numpy()
    band = 5.0 + 2.0 * np.abs(np.sin(idx / 8.0))
    return pd.DataFrame(
        {
            "date": _dates(),
            "close": close,
            "rsi_14": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_line - macd_signal,
            "stoch_k": stoch_k,
            "stoch_d": 50.0 + 40.0 * np.sin(idx / 4.5 - 0.5),
            "bb_upper": bb_middle + band,
            "bb_middle": bb_middle,
            "bb_lower": bb_middle - band,
            "ma20_pct": np.sin(idx / 9.0),
            "vol20_annualized": 0.12 + 0.05 * np.abs(np.sin(idx / 10.0)),
        }
    )


def build_baseline_df() -> pd.DataFrame:
    """Baseline comparison frame shaped like ``run_all_baselines`` output."""
    return pd.DataFrame(
        {
            "baseline": [
                "buy_and_hold",
                "momentum",
                "rsi_mean_reversion",
                "stochastic_oscillator",
                "macd_momentum",
                "bollinger_bands",
                "moving_average_crossover",
                "random",
            ],
            "total_return": [45.2, 32.1, 28.7, 24.3, 19.8, 22.5, 30.0, -5.2],
            "sharpe_annualized": [1.2, 0.8, 0.9, 0.7, 0.6, 0.65, 0.85, -0.3],
            "win_rate": [52.1, 51.2, 53.4, 50.8, 49.2, 50.1, 51.9, 48.5],
            "max_drawdown": [-18.2, -22.1, -15.3, -19.8, -25.0, -20.2, -17.5, -35.0],
        }
    )


@pytest.fixture
def parsed_df():
    return build_parsed_df()


@pytest.fixture
def features_df():
    return build_features_df()


def _assert_png(path):
    assert path.exists(), f"chart file was not created: {path}"
    assert path.stat().st_size > 0, f"chart file is empty: {path}"


def test_create_calibration_plot(parsed_df, tmp_path):
    out = tmp_path / "calibration.png"
    result = create_calibration_plot(parsed_df, "test_model", str(out))
    _assert_png(out)
    # Returns the per-bin calibration data used downstream.
    assert result is not None


def test_create_calibration_by_decision_plot(parsed_df, tmp_path):
    out = tmp_path / "calibration_by_decision.png"
    create_calibration_by_decision_plot(parsed_df, "test_model", str(out))
    _assert_png(out)


def test_create_risk_analysis_chart(parsed_df, tmp_path):
    out = tmp_path / "risk.png"
    create_risk_analysis_chart(parsed_df, "test_model", str(out))
    _assert_png(out)


def test_create_rolling_performance_chart(parsed_df, tmp_path):
    out = tmp_path / "rolling.png"
    create_rolling_performance_chart(parsed_df, "test_model", str(out))
    _assert_png(out)


def test_generate_calibration_analysis_report(parsed_df, tmp_path):
    # This produces a markdown report (not a chart) from the calibration data
    # that create_calibration_plot returns; it still lives in the chart cluster.
    calibration_data = create_calibration_plot(
        parsed_df, "test_model", str(tmp_path / "calibration.png")
    )
    out_md = tmp_path / "calibration_analysis.md"
    result = generate_calibration_analysis_report(
        calibration_data, parsed_df, "test_model", str(out_md)
    )
    assert out_md.exists() and out_md.stat().st_size > 0
    assert "ece" in result


def test_create_category_performance_plot(tmp_path):
    category_stats = calculate_category_performance(build_baseline_df())
    llm_metrics = {"total_return": 38.5, "sharpe": 1.1, "win_rate": 54.2}
    out = tmp_path / "category.png"
    create_category_performance_plot(category_stats, llm_metrics, str(out))
    _assert_png(out)


def test_create_indicator_performance_plot(parsed_df, features_df, tmp_path):
    indicator_perf = analyze_indicator_specific_performance(parsed_df, features_df)
    out = tmp_path / "indicator.png"
    create_indicator_performance_plot(indicator_perf, str(out))
    _assert_png(out)


def test_create_technical_indicators_plot(parsed_df, features_df, tmp_path):
    out = tmp_path / "technical.png"
    create_technical_indicators_plot(features_df, parsed_df, "test_model", str(out))
    _assert_png(out)


def test_create_technical_indicators_timeline(parsed_df, features_df, tmp_path):
    out = tmp_path / "timeline.png"
    create_technical_indicators_timeline(features_df, parsed_df, "test_model", str(out))
    _assert_png(out)


def test_create_rsi_performance_analysis(parsed_df, features_df, tmp_path):
    out = tmp_path / "rsi.png"
    create_rsi_performance_analysis(parsed_df, features_df, "test_model", str(out))
    _assert_png(out)
