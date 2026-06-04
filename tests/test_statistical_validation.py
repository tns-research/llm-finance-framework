"""Deterministic unit tests for src/statistical_validation.py.

These lock the scientific core (Sharpe bootstrap, out-of-sample split, VaR/stress,
drawdown, risk attribution, regime and decision-effectiveness math) against small,
hand-verifiable inputs so the module can be refactored safely.
"""

import numpy as np
import pandas as pd
import pytest

from src.statistical_validation import (
    analyze_decision_effectiveness,
    analyze_market_regimes,
    bootstrap_sharpe_comparison,
    calculate_max_drawdown,
    calculate_risk_attribution,
    calculate_var_and_stress_tests,
    evaluate_hold_decisions_dual_criteria,
    out_of_sample_validation,
)

TRADING_DAYS = 252


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Reference Sharpe matching the module's population-std convention."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# calculate_max_drawdown - pure, fully deterministic
# ---------------------------------------------------------------------------


def test_max_drawdown_simple_case():
    # cumulative = [1, -1, 0]; running_max = [1, 1, 1]; drawdowns = [0, -2, -1]
    returns = np.array([1.0, -2.0, 1.0])
    assert calculate_max_drawdown(returns) == pytest.approx(-2.0)


def test_max_drawdown_monotonic_gains_is_zero():
    returns = np.array([0.5, 0.5, 0.5, 0.5])
    assert calculate_max_drawdown(returns) == pytest.approx(0.0)


def test_max_drawdown_all_losses_from_first_peak():
    returns = np.array([-1.0, -2.0, -3.0])
    # cumulative = [-1, -3, -6]; running_max = [-1, -1, -1]; drawdowns = [0, -2, -5]
    assert calculate_max_drawdown(returns) == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# bootstrap_sharpe_comparison - observed stats are deterministic
# ---------------------------------------------------------------------------


def test_bootstrap_observed_sharpe_matches_formula():
    np.random.seed(42)
    strategy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    benchmark = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    res = bootstrap_sharpe_comparison(strategy, benchmark, n_bootstrap=200)

    assert res["strategy_sharpe"] == pytest.approx(
        round(_annualized_sharpe(strategy), 3)
    )
    assert res["benchmark_sharpe"] == pytest.approx(
        round(_annualized_sharpe(benchmark), 3)
    )
    assert res["n_observations"] == 5
    assert res["n_bootstrap"] == 200


def test_bootstrap_identical_series_has_zero_difference():
    np.random.seed(0)
    series = np.array([0.3, -0.1, 0.2, 0.4, -0.2, 0.1])
    res = bootstrap_sharpe_comparison(series, series.copy(), n_bootstrap=100)

    # Strategy and benchmark are identical -> every resample cancels out.
    assert res["sharpe_difference"] == pytest.approx(0.0)
    assert res["bootstrap_mean_diff"] == pytest.approx(0.0)
    assert res["ci_95_bootstrap"] == [0.0, 0.0]
    assert res["effect_size"] == 0


def test_bootstrap_is_reproducible_under_seed():
    strategy = np.array([1.0, -1.0, 2.0, -2.0, 3.0, 0.5])
    benchmark = np.array([0.5, -0.5, 1.0, -1.0, 1.5, 0.25])

    np.random.seed(123)
    first = bootstrap_sharpe_comparison(strategy, benchmark, n_bootstrap=300)
    np.random.seed(123)
    second = bootstrap_sharpe_comparison(strategy, benchmark, n_bootstrap=300)

    assert first == second


# ---------------------------------------------------------------------------
# calculate_risk_attribution
# ---------------------------------------------------------------------------


def test_risk_attribution_identical_series_perfect_correlation():
    rng = np.random.default_rng(7)
    n = 50
    market = rng.normal(0, 1, n)
    res = calculate_risk_attribution(market.copy(), market.copy())

    assert res["correlation"] == pytest.approx(1.0)
    # Identical series -> idiosyncratic risk is clamped to 0 by the sqrt guard.
    assert res["idiosyncratic_risk"] == pytest.approx(0.0, abs=1e-6)
    # beta = cov(ddof=1) / var(ddof=0) = n / (n - 1) for identical series. This
    # documents the module's mixed-ddof convention so a refactor cannot drift it.
    assert res["beta"] == pytest.approx(n / (n - 1), rel=1e-9)


def test_risk_attribution_anticorrelated_series_negative_beta():
    market = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    strategy = -market  # perfectly anti-correlated
    res = calculate_risk_attribution(strategy, market)

    assert res["correlation"] == pytest.approx(-1.0)
    assert res["beta"] < 0


# ---------------------------------------------------------------------------
# calculate_var_and_stress_tests
# ---------------------------------------------------------------------------


def test_var_stress_base_case_preserves_returns():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 1, 80)
    res = calculate_var_and_stress_tests(returns)

    base = res["stress_tests"]["base_case"]
    np.testing.assert_array_equal(base["returns"], returns)
    assert base["total_return"] == pytest.approx(np.cumsum(returns)[-1])
    # high_volatility scenario is exactly 2x the base returns
    np.testing.assert_array_almost_equal(
        res["stress_tests"]["high_volatility"]["returns"], returns * 2
    )
    # rolling VaR computed because len > 63
    assert "var_95" in res and len(res["var_95"]) == len(returns) - 63
    assert res["max_drawdown"] == pytest.approx(calculate_max_drawdown(returns))


def test_var_stress_short_series_skips_rolling_var():
    returns = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    res = calculate_var_and_stress_tests(returns)
    assert "var_95" not in res  # not enough data for the 63-day window
    assert "stress_tests" in res


# ---------------------------------------------------------------------------
# out_of_sample_validation
# ---------------------------------------------------------------------------


def _make_parsed_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    strategy_return = rng.normal(0.05, 1.0, n)
    next_return_1d = rng.normal(0.04, 1.0, n)
    decisions = np.array(["BUY", "HOLD", "SELL"])[rng.integers(0, 3, n)]
    return pd.DataFrame(
        {
            "date": dates,
            "strategy_return": strategy_return,
            "next_return_1d": next_return_1d,
            "decision": decisions,
        }
    )


def test_oos_invalid_split_date_returns_error():
    df = _make_parsed_df(150)
    res = out_of_sample_validation(df, split_date="not-a-date")
    assert "error" in res


def test_oos_insufficient_training_returns_error():
    df = _make_parsed_df(150)
    # Split very late so training set is tiny.
    res = out_of_sample_validation(df, split_date="2017-01-01", min_train_periods=100)
    assert "error" in res


def test_oos_valid_split_partitions_all_rows():
    df = _make_parsed_df(200, seed=3)
    split = "2018-04-01"
    res = out_of_sample_validation(df, split_date=split, min_train_periods=50)

    assert "error" not in res
    n_train = res["train_period"]["n_periods"]
    n_test = res["test_period"]["n_periods"]
    assert n_train + n_test == len(df)
    assert res["data_split"]["train_pct"] + res["data_split"][
        "test_pct"
    ] == pytest.approx(100.0, abs=0.2)


# ---------------------------------------------------------------------------
# analyze_decision_effectiveness
# ---------------------------------------------------------------------------


def test_decision_effectiveness_distribution_counts():
    df = pd.DataFrame(
        {
            "decision": ["BUY", "BUY", "HOLD", "SELL"],
            "strategy_return": [1.0, -0.5, 0.0, 0.5],
            "next_return_1d": [0.8, -0.4, 0.1, 0.3],
        }
    )
    res = analyze_decision_effectiveness(df)

    dist = res["decision_distribution"]
    assert dist["BUY"]["count"] == 2
    assert dist["HOLD"]["count"] == 1
    assert dist["SELL"]["count"] == 1
    assert dist["BUY"]["percentage"] == pytest.approx(50.0)

    overall = res["overall_effectiveness"]
    assert overall["total_decisions"] == 4
    assert overall["overall_total_return"] == pytest.approx(1.0)  # 1 - 0.5 + 0 + 0.5


def test_decision_effectiveness_per_decision_metrics():
    df = pd.DataFrame(
        {
            "decision": ["BUY", "BUY", "BUY"],
            "strategy_return": [1.0, 2.0, 3.0],
            "next_return_1d": [0.5, 0.5, 0.5],
        }
    )
    res = analyze_decision_effectiveness(df)
    buy = res["decision_performance"]["BUY"]
    assert buy["total_decisions"] == 3
    assert buy["win_rate"] == pytest.approx(100.0)
    assert buy["avg_daily_return"] == pytest.approx(2.0)
    assert buy["total_return"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# analyze_market_regimes
# ---------------------------------------------------------------------------


def test_market_regimes_missing_columns_returns_empty():
    df = pd.DataFrame({"strategy_return": [0.1, 0.2]})  # no next_return_1d
    assert analyze_market_regimes(df) == {}


def test_market_regimes_classifies_with_enough_data():
    n = 80
    rng = np.random.default_rng(11)
    # First half calm, second half turbulent -> distinct volatility regimes.
    calm = rng.normal(0, 0.2, n // 2)
    wild = rng.normal(0, 3.0, n // 2)
    market = np.concatenate([calm, wild])
    strategy = market * 0.5
    df = pd.DataFrame({"next_return_1d": market, "strategy_return": strategy})

    res = analyze_market_regimes(df)
    assert res  # non-empty
    regime_keys = set(res.keys()) - {"_summary"}
    assert regime_keys.issubset(
        {"low_volatility", "moderate_volatility", "high_volatility"}
    )
    total_days = sum(res[k]["days"] for k in regime_keys)
    assert 0 < total_days <= n
    if "_summary" in res:
        assert res["_summary"]["best_regime"] in regime_keys


# ---------------------------------------------------------------------------
# evaluate_hold_decisions_dual_criteria - characterization pin
#
# This 200+ line function is reachable in production (reporting.charts ->
# calculate_decision_success) but was previously unguarded. These tests pin the
# full output dict on a fully deterministic sinusoidal fixture so the per-HOLD
# scoring loop can be safely extracted into a helper.
# ---------------------------------------------------------------------------


def _hold_fixture(n: int = 90) -> pd.DataFrame:
    idx = np.arange(n)
    decisions = np.array(["BUY", "HOLD", "SELL"])[idx % 3]
    next_return_1d = 0.008 * np.sin(idx / 3.0 + 0.3)
    return pd.DataFrame({"decision": decisions, "next_return_1d": next_return_1d})


def test_hold_dual_criteria_no_holds_returns_note():
    df = pd.DataFrame({"decision": ["BUY", "SELL"], "next_return_1d": [0.01, -0.01]})
    assert evaluate_hold_decisions_dual_criteria(df) == {
        "note": "No HOLD decisions to evaluate"
    }


def test_hold_dual_criteria_scalar_metrics():
    res = evaluate_hold_decisions_dual_criteria(_hold_fixture())

    assert res["overall_hold_success_rate"] == pytest.approx(0.13333333333333333)

    quiet = res["quiet_market_success"]
    assert quiet["success_rate"] == pytest.approx(0.13333333333333333)
    assert quiet["threshold"] == 0.002
    assert quiet["successful_holds"] == 4
    assert quiet["total_holds"] == 30
    assert quiet["interpretation"] == (
        "HOLD succeeded in 13.3% of very quiet markets (<0.2% daily moves)"
    )

    assert res["relative_performance"]["avg_score"] == pytest.approx(
        0.43333333333333346
    )
    assert res["risk_avoidance"]["avoidance_rate"] == pytest.approx(0.0)

    ctx = res["contextual_correctness"]
    assert ctx["avg_context_score"] == pytest.approx(0.43333333333333346)
    assert ctx["context_success_rate"] == pytest.approx(0.3333333333333333)
    assert ctx["reason_breakdown"] == {
        "quiet_market_timing": 20,
        "reasonable_in_volatility": 6,
        "avoided_significant_loss": 4,
    }

    combined = res["combined_assessment"]
    assert combined["quiet_weight"] == 0.6
    assert combined["context_weight"] == 0.4
    assert combined["overall_score"] == pytest.approx(0.13333333333333333)
    assert combined["performance_category"] == "very_poor"

    stats = res["hold_statistics"]
    assert stats["total_hold_decisions"] == 30
    assert stats["hold_percentage"] == pytest.approx(0.3333333333333333)
    assert stats["avg_market_move_during_hold"] == pytest.approx(0.005196684137217255)
    assert stats["hold_during_quiet_pct"] == pytest.approx(0.13333333333333333)

    assert res["summary"]["hold_effectiveness"] == (
        "VERY_POOR (13.3% combined success rate)"
    )
    assert res["summary"]["key_insight"] == (
        "HOLD decisions achieved 43.3% relative performance and avoided losses in "
        "0.0% of cases. Quiet market success: 13.3%"
    )


def test_hold_dual_criteria_per_decision_indicators():
    res = evaluate_hold_decisions_dual_criteria(_hold_fixture())
    hold_df = res["hold_success_indicators"]

    assert list(hold_df.columns) == ["index", "hold_success", "hold_success_score"]
    assert len(hold_df) == 30
    # The 30 HOLD rows sit at every third position starting at index 1.
    assert hold_df["index"].tolist() == list(range(1, 90, 3))
    # Exactly the 4 quiet-market holds score a success.
    assert hold_df["hold_success"].sum() == 4
    assert hold_df["hold_success_score"].sum() == pytest.approx(4.0)
    successes = hold_df.loc[hold_df["hold_success"] == 1, "index"].tolist()
    assert successes == [28, 37, 46, 55]
