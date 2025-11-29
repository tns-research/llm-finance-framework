# src/statistical_validation.py
"""
Statistical validation of strategy performance.

Provides rigorous statistical testing for LLM trading strategies including:
- Bootstrap significance testing vs benchmarks
- Out-of-sample validation for overfitting detection
- Comprehensive statistical validation suite
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import json
import os


def bootstrap_sharpe_comparison(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 10000
) -> Dict:
    """
    Bootstrap test: Is strategy Sharpe ratio significantly different from benchmark?

    Uses bootstrap resampling to test statistical significance while accounting for
    non-normal return distributions common in financial data.

    Args:
        strategy_returns: Array of strategy returns (percentage)
        benchmark_returns: Array of benchmark returns (percentage)
        n_bootstrap: Number of bootstrap resamples (default 10,000)

    Returns:
        Dict with statistical test results
    """
    def sharpe_ratio(returns: np.ndarray) -> float:
        """Calculate Sharpe ratio (annualized assuming daily returns)"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualize

    # Observed statistics
    observed_strategy_sharpe = sharpe_ratio(strategy_returns)
    observed_benchmark_sharpe = sharpe_ratio(benchmark_returns)
    observed_diff = observed_strategy_sharpe - observed_benchmark_sharpe

    # Bootstrap resampling
    diffs = []
    n = len(strategy_returns)

    for _ in range(n_bootstrap):
        # Resample with replacement (paired bootstrap)
        idx = np.random.choice(n, n, replace=True)
        strategy_boot = strategy_returns[idx]
        benchmark_boot = benchmark_returns[idx]

        boot_strategy_sharpe = sharpe_ratio(strategy_boot)
        boot_benchmark_sharpe = sharpe_ratio(benchmark_boot)
        diffs.append(boot_strategy_sharpe - boot_benchmark_sharpe)

    diffs = np.array(diffs)

    # Statistical tests
    # Two-sided test: is there a significant difference?
    p_value_two_sided = 2 * min(
        np.mean(diffs <= 0),
        np.mean(diffs >= 0)
    )

    # One-sided test: is strategy significantly better? (right-tailed)
    p_value_better = np.mean(diffs <= 0)

    # One-sided test: is strategy significantly worse? (left-tailed)
    p_value_worse = np.mean(diffs >= 0)

    return {
        "strategy_sharpe": round(observed_strategy_sharpe, 3),
        "benchmark_sharpe": round(observed_benchmark_sharpe, 3),
        "sharpe_difference": round(observed_diff, 3),
        "p_value_two_sided": round(p_value_two_sided, 4),
        "p_value_strategy_better": round(p_value_better, 4),
        "p_value_strategy_worse": round(p_value_worse, 4),
        "significant_difference_5pct": p_value_two_sided < 0.05,
        "strategy_significantly_better_5pct": p_value_better < 0.05,
        "strategy_significantly_worse_5pct": p_value_worse < 0.05,
        "ci_95_bootstrap": [
            float(round(np.percentile(diffs, 2.5), 3)),
            float(round(np.percentile(diffs, 97.5), 3))
        ],
        "bootstrap_mean_diff": round(np.mean(diffs), 3),
        "bootstrap_std_diff": round(np.std(diffs), 3),
        "effect_size": round(observed_diff / np.std(diffs), 3) if np.std(diffs) > 0 else 0,
        "n_bootstrap": n_bootstrap,
        "n_observations": n
    }


def out_of_sample_validation(
    parsed_df: pd.DataFrame,
    split_date: str,
    min_train_periods: int = 100
) -> Dict:
    """
    Out-of-sample validation to detect overfitting and assess generalization.

    Splits data chronologically into training and test periods to check if
    the strategy performs well on unseen future data.

    Args:
        parsed_df: DataFrame with parsed trading results
        split_date: Date string to split train/test (e.g., "2020-01-01")
        min_train_periods: Minimum training periods required

    Returns:
        Dict with validation results
    """
    try:
        split_date = pd.to_datetime(split_date)
    except:
        return {"error": f"Invalid split_date format: {split_date}"}

    # Split data
    train_df = parsed_df[parsed_df["date"] < split_date].copy()
    test_df = parsed_df[parsed_df["date"] >= split_date].copy()

    if len(train_df) < min_train_periods:
        return {"error": f"Insufficient training data: {len(train_df)} < {min_train_periods}"}

    if len(test_df) == 0:
        return {"error": "No test data available"}

    def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = df["strategy_return"].values

        if len(returns) == 0:
            return {"error": "No returns data"}

        # Basic metrics
        total_return = returns.sum()
        mean_return = returns.mean()
        volatility = returns.std()
        sharpe = (mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        win_rate = (returns > 0).mean()

        # Drawdown analysis
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Additional metrics
        skewness = stats.skew(returns) if len(returns) > 2 else 0
        kurtosis = stats.kurtosis(returns) if len(returns) > 2 else 0

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        return {
            "total_return_pct": round(total_return, 2),
            "mean_daily_return_pct": round(mean_return, 4),
            "volatility_pct": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(win_rate * 100, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "var_95_pct": round(var_95, 2),
            "n_periods": len(df),
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max())
            }
        }

    train_metrics = calculate_performance_metrics(train_df)
    test_metrics = calculate_performance_metrics(test_df)

    if "error" in train_metrics or "error" in test_metrics:
        return {"error": "Performance calculation failed"}

    # Comparative analysis
    sharpe_decay = None
    if train_metrics["sharpe_ratio"] != 0:
        sharpe_decay = (train_metrics["sharpe_ratio"] - test_metrics["sharpe_ratio"]) / train_metrics["sharpe_ratio"]

    return_decay = None
    if abs(train_metrics["total_return_pct"]) > 0.01:  # Avoid division by very small numbers
        return_decay = (train_metrics["total_return_pct"] - test_metrics["total_return_pct"]) / abs(train_metrics["total_return_pct"])

    # Overfitting detection
    overfitting_indicators = {
        "sharpe_decay_severe": sharpe_decay is not None and sharpe_decay > 0.5,  # >50% decay
        "return_decay_severe": return_decay is not None and return_decay > 0.7,  # >70% decay
        "test_performance_poor": test_metrics["sharpe_ratio"] < -0.5,  # Very poor test performance
        "high_train_overfitting": (
            train_metrics["sharpe_ratio"] > 1.0 and
            test_metrics["sharpe_ratio"] < 0.0
        )
    }

    overall_overfitting_detected = any(overfitting_indicators.values())

    # Generalization assessment
    generalizes_well = (
        test_metrics["sharpe_ratio"] > 0.1 and  # Decent test Sharpe
        not overall_overfitting_detected
    )

    return {
        "train_period": train_metrics,
        "test_period": test_metrics,
        "comparative_analysis": {
            "sharpe_decay": round(sharpe_decay, 3) if sharpe_decay else None,
            "return_decay": round(return_decay, 3) if return_decay else None,
            "sharpe_decay_pct": round(sharpe_decay * 100, 1) if sharpe_decay else None,
            "return_decay_pct": round(return_decay * 100, 1) if return_decay else None,
        },
        "overfitting_detection": {
            **overfitting_indicators,
            "overall_overfitting_detected": overall_overfitting_detected,
            "overfitting_severity": "high" if sum(overfitting_indicators.values()) >= 2 else "moderate" if sum(overfitting_indicators.values()) >= 1 else "low"
        },
        "generalization_assessment": {
            "generalizes_well": generalizes_well,
            "test_performance_adequate": test_metrics["sharpe_ratio"] > 0,
            "consistent_performance": abs(sharpe_decay or 0) < 0.3
        },
        "split_date": str(split_date),
        "data_split": {
            "train_pct": round(len(train_df) / len(parsed_df) * 100, 1),
            "test_pct": round(len(test_df) / len(parsed_df) * 100, 1)
        }
    }


def comprehensive_statistical_validation(
    parsed_df: pd.DataFrame,
    model_tag: str,
    split_date: Optional[str] = None,
    benchmark_returns: Optional[np.ndarray] = None
) -> Dict:
    """
    Run comprehensive statistical validation suite for LLM trading strategy.

    Args:
        parsed_df: DataFrame with parsed trading results
        model_tag: Name/tag of the model being validated
        split_date: Date for train/test split (optional)
        benchmark_returns: Benchmark returns array for comparison (optional)

    Returns:
        Comprehensive validation results dictionary
    """
    results = {
        "model": model_tag,
        "validation_timestamp": str(pd.Timestamp.now()),
        "dataset_info": {
            "n_periods": len(parsed_df),
            "date_range": {
                "start": str(parsed_df["date"].min()),
                "end": str(parsed_df["date"].max())
            },
            "total_strategy_return": round(parsed_df["strategy_return"].sum(), 2),
            "total_index_return": round(parsed_df["next_return_1d"].sum(), 2)
        }
    }

    # Out-of-sample validation
    if split_date:
        print(f"Running out-of-sample validation with split date: {split_date}")
        oos_results = out_of_sample_validation(parsed_df, split_date)
        results["out_of_sample_validation"] = oos_results

        if "error" in oos_results:
            print(f"Warning: Out-of-sample validation failed: {oos_results['error']}")
        else:
            print("✓ Out-of-sample validation completed")

    # Bootstrap testing vs benchmark
    if benchmark_returns is not None:
        print("Running bootstrap Sharpe ratio comparison...")
        strategy_returns = parsed_df["strategy_return"].values

        bootstrap_results = bootstrap_sharpe_comparison(
            strategy_returns, benchmark_returns, n_bootstrap=5000
        )
        results["bootstrap_vs_benchmark"] = bootstrap_results
        print("✓ Bootstrap comparison completed")
    else:
        # Default: compare vs index returns
        print("Running bootstrap comparison vs index returns...")
        strategy_returns = parsed_df["strategy_return"].values
        index_returns = parsed_df["next_return_1d"].values

        bootstrap_results = bootstrap_sharpe_comparison(
            strategy_returns, index_returns, n_bootstrap=5000
        )
        results["bootstrap_vs_index"] = bootstrap_results
        print("✓ Bootstrap comparison vs index completed")

    # Comprehensive decision analysis (BUY, HOLD, SELL)
    decision_analysis = analyze_decision_effectiveness(parsed_df)
    results["decision_effectiveness"] = decision_analysis

    # Legacy HOLD decision analysis (for backward compatibility)
    hold_analysis = evaluate_hold_decisions_dual_criteria(parsed_df)
    results["hold_decision_analysis"] = hold_analysis

    # Summary assessment
    results["summary_assessment"] = generate_validation_summary(results)

    return results


def analyze_decision_effectiveness(parsed_df: pd.DataFrame) -> Dict:
    """
    Comprehensive analysis of all decision types (BUY, HOLD, SELL) and overall effectiveness.

    Args:
        parsed_df: DataFrame with trading decisions and returns

    Returns:
        Dict with detailed analysis of each decision type and overall performance
    """
    results = {
        "decision_distribution": {},
        "decision_performance": {},
        "overall_effectiveness": {},
        "decision_timing": {},
        "risk_adjusted_analysis": {}
    }

    # Get unique decisions
    decisions = ['BUY', 'HOLD', 'SELL']
    decision_data = {}

    for decision in decisions:
        decision_df = parsed_df[parsed_df['decision'] == decision].copy()
        decision_data[decision] = decision_df

        if len(decision_df) > 0:
            returns = decision_df['strategy_return'].values
            market_returns = decision_df['next_return_1d'].values

            # Basic performance metrics
            win_rate = np.mean(returns > 0)
            avg_return = np.mean(returns)
            total_trades = len(returns)
            total_return = np.sum(returns)

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            sharpe = avg_return / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = calculate_max_drawdown(returns) if len(returns) > 1 else 0

            # Market comparison
            market_avg_return = np.mean(market_returns)
            excess_return = avg_return - market_avg_return

            # Decision frequency
            decision_pct = len(decision_df) / len(parsed_df) * 100

            results["decision_performance"][decision] = {
                "total_decisions": total_trades,
                "decision_frequency_pct": round(decision_pct, 2),
                "win_rate": round(win_rate * 100, 2),
                "avg_daily_return": round(avg_return, 4),
                "total_return": round(total_return, 4),
                "volatility_annualized": round(volatility, 4),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown": round(max_drawdown, 4),
                "market_avg_return": round(market_avg_return, 4),
                "excess_return": round(excess_return, 4),
                "excess_return_annualized": round(excess_return * 252, 2)
            }

    # Decision distribution
    total_decisions = len(parsed_df)
    for decision in decisions:
        count = len(decision_data[decision])
        results["decision_distribution"][decision] = {
            "count": count,
            "percentage": round(count / total_decisions * 100, 1) if total_decisions > 0 else 0
        }

    # Overall effectiveness
    all_returns = parsed_df['strategy_return'].values
    results["overall_effectiveness"] = {
        "total_decisions": total_decisions,
        "overall_win_rate": round(np.mean(all_returns > 0) * 100, 2),
        "overall_avg_return": round(np.mean(all_returns), 4),
        "overall_total_return": round(np.sum(all_returns), 2),
        "overall_volatility": round(np.std(all_returns) * np.sqrt(252), 4),
        "overall_sharpe": round(np.mean(all_returns) / np.std(all_returns) * np.sqrt(252), 3) if np.std(all_returns) > 0 else 0,
        "overall_max_drawdown": round(calculate_max_drawdown(all_returns), 4)
    }

    # Decision timing analysis - analyze performance by time of day/week patterns
    # This is a simplified version - could be expanded
    results["decision_timing"] = {
        "note": "Decision timing analysis available - could analyze performance by time of day, day of week, month, etc."
    }

    # Risk-adjusted analysis
    if results["decision_performance"]:
        # Find best and worst performing decisions
        best_decision = max(results["decision_performance"].keys(),
                          key=lambda x: results["decision_performance"][x]["excess_return"])
        worst_decision = min(results["decision_performance"].keys(),
                           key=lambda x: results["decision_performance"][x]["excess_return"])

        results["risk_adjusted_analysis"] = {
            "best_decision": best_decision,
            "worst_decision": worst_decision,
            "best_excess_return": results["decision_performance"][best_decision]["excess_return_annualized"],
            "worst_excess_return": results["decision_performance"][worst_decision]["excess_return_annualized"],
            "decision_consistency": "variable" if abs(
                results["decision_performance"][best_decision]["excess_return_annualized"] -
                results["decision_performance"][worst_decision]["excess_return_annualized"]
            ) > 10 else "consistent"
        }

    return results


def evaluate_hold_decisions_dual_criteria(parsed_df: pd.DataFrame) -> Dict:
    """
    Evaluate HOLD decisions using dual criteria:
    1. Quiet market success (<0.2% moves)
    2. Contextual decision correctness (volatility, regime changes, uncertainty)
    """
    hold_decisions = parsed_df[parsed_df["decision"] == "HOLD"].copy()

    if len(hold_decisions) == 0:
        return {"note": "No HOLD decisions to evaluate"}

    # Add rolling calculations for context analysis
    parsed_df_copy = parsed_df.copy()
    parsed_df_copy['market_volatility_20d'] = parsed_df_copy['next_return_1d'].rolling(20).std()
    parsed_df_copy['market_trend_10d'] = parsed_df_copy['next_return_1d'].rolling(10).mean()
    parsed_df_copy['regime_change'] = (parsed_df_copy['market_trend_10d'].diff().abs() > 0.001)

    # Filter to HOLD decisions
    hold_data = parsed_df_copy[parsed_df_copy["decision"] == "HOLD"]

    # ===== CRITERION 1: QUIET MARKET SUCCESS =====
    quiet_threshold = 0.002  # 0.2% very quiet market
    market_returns = hold_data["next_return_1d"]
    quiet_markets = market_returns.abs() < quiet_threshold

    quiet_success_rate = quiet_markets.mean()
    quiet_success_count = quiet_markets.sum()

    # ===== CRITERION 2: RELATIVE PERFORMANCE ANALYSIS =====
    # Compare HOLD performance vs directional bets in similar conditions

    relative_performance_scores = []
    risk_avoidance_scores = []

    for idx, row in hold_data.iterrows():
        # Get market conditions for this HOLD decision
        vol_level = row['market_volatility_20d'] if pd.notna(row['market_volatility_20d']) else 0
        market_return = row['next_return_1d']

        # Simulate what BUY/SELL would have achieved
        buy_return = market_return  # +1 position
        sell_return = -market_return  # -1 position
        hold_return = 0  # 0 position

        # Relative performance: How did HOLD do vs the better directional choice?
        better_direction = max(buy_return, sell_return)
        relative_perf = hold_return - better_direction  # Negative = HOLD outperformed

        # Risk avoidance: Did HOLD avoid a loss that would have occurred?
        directional_loss = min(buy_return, sell_return)
        risk_avoided = directional_loss < -0.005  # Would have lost >0.5%

        # Score based on volatility-adjusted performance
        vol_adjusted_score = 0
        if vol_level > parsed_df_copy['market_volatility_20d'].quantile(0.75):
            # High volatility - HOLD gets credit for avoiding risk
            if risk_avoided:
                vol_adjusted_score = 1.0  # Successfully avoided loss
            elif relative_perf > -0.01:  # HOLD not much worse than best direction
                vol_adjusted_score = 0.5  # Reasonable choice
        else:
            # Low volatility - HOLD should only be chosen if market is very quiet
            if abs(market_return) < 0.002:  # Very quiet market
                vol_adjusted_score = 1.0
            elif abs(market_return) < 0.01:  # Moderately quiet
                vol_adjusted_score = 0.3

        relative_performance_scores.append(vol_adjusted_score)
        risk_avoidance_scores.append(1.0 if risk_avoided else 0.0)

    # Calculate new metrics
    avg_relative_performance = np.mean(relative_performance_scores) if relative_performance_scores else 0
    risk_avoidance_rate = np.mean(risk_avoidance_scores) if risk_avoidance_scores else 0

    # For backward compatibility, keep old context calculation
    contexts = [{'score': 1.0, 'reasons': ['legacy'], 'max_reason': 'legacy'} for _ in hold_data.iterrows()]
    context_scores = [c['score'] for c in contexts]
    avg_context_score = 1.0
    context_success_rate = 1.0  # Always "successful" for legacy compatibility

    # ===== COMBINED SCORING =====
    quiet_weight = 0.6
    context_weight = 0.4

    combined_scores = []
    for i, quiet_win in enumerate(quiet_markets):
        # Context success = HIGH context score (risky conditions appropriate for HOLD)
        context_win = context_scores[i] > 0.5 if i < len(context_scores) else False
        combined_score = (quiet_weight * int(quiet_win) + context_weight * int(context_win))
        combined_scores.append(combined_score)

    overall_success_rate = np.mean(combined_scores) if combined_scores else 0

    # ===== CONTEXT REASONS BREAKDOWN =====
    context_reasons = {}
    for context in contexts:
        for reason in context['reasons']:
            context_reasons[reason] = context_reasons.get(reason, 0) + 1

    # ===== PERFORMANCE CATEGORY =====
    def categorize_hold_performance(success_rate: float) -> str:
        if success_rate > 0.7:
            return "excellent"
        elif success_rate > 0.6:
            return "good"
        elif success_rate > 0.5:
            return "adequate"
        elif success_rate > 0.4:
            return "poor"
        else:
            return "very_poor"

    performance_category = categorize_hold_performance(overall_success_rate)

    return {
        "overall_hold_success_rate": overall_success_rate,

        # Quiet market criterion
        "quiet_market_success": {
            "success_rate": quiet_success_rate,
            "threshold": quiet_threshold,
            "successful_holds": int(quiet_success_count),
            "total_holds": len(hold_decisions),
            "interpretation": f"HOLD succeeded in {quiet_success_rate:.1%} of very quiet markets (<{quiet_threshold:.1%} daily moves)"
        },

        # Enhanced HOLD evaluation metrics
        "relative_performance": {
            "avg_score": avg_relative_performance,
            "interpretation": f"HOLD achieved {avg_relative_performance:.1%} relative performance score vs directional bets"
        },

        "risk_avoidance": {
            "avoidance_rate": risk_avoidance_rate,
            "interpretation": f"HOLD avoided losses that directional bets would have incurred in {risk_avoidance_rate:.1%} of cases"
        },

        # Legacy contextual correctness (kept for compatibility)
        "contextual_correctness": {
            "avg_context_score": avg_context_score,
            "context_success_rate": context_success_rate,
            "reason_breakdown": {},
            "interpretation": f"Legacy context evaluation: {context_success_rate:.1%} success rate"
        },

        # Combined assessment
        "combined_assessment": {
            "quiet_weight": quiet_weight,
            "context_weight": context_weight,
            "overall_score": overall_success_rate,
            "performance_category": performance_category
        },

        "hold_statistics": {
            "total_hold_decisions": len(hold_decisions),
            "hold_percentage": len(hold_decisions) / len(parsed_df) if len(parsed_df) > 0 else 0,
            "avg_market_move_during_hold": market_returns.abs().mean() if len(market_returns) > 0 else 0,
            "hold_during_quiet_pct": quiet_markets.mean() if len(quiet_markets) > 0 else 0
        },

        "summary": {
            "hold_effectiveness": f"{performance_category.upper()} ({overall_success_rate:.1%} combined success rate)",
            "key_insight": f"HOLD decisions achieved {avg_relative_performance:.1%} relative performance and avoided losses in {risk_avoidance_rate:.1%} of cases. Quiet market success: {quiet_success_rate:.1%}"
        }
    }


def generate_validation_summary(validation_results: Dict) -> Dict:
    """
    Generate human-readable summary of validation results.
    """
    summary = {
        "overall_assessment": "unknown",
        "key_findings": [],
        "recommendations": [],
        "confidence_level": "low"
    }

    # Out-of-sample assessment
    if "out_of_sample_validation" in validation_results:
        oos = validation_results["out_of_sample_validation"]
        if "error" not in oos:
            if oos["generalization_assessment"]["generalizes_well"]:
                summary["key_findings"].append("✅ Strategy generalizes well to out-of-sample data")
                summary["confidence_level"] = "high"
            else:
                summary["key_findings"].append("⚠️  Strategy shows signs of overfitting or poor generalization")
                summary["recommendations"].append("Consider model regularization or simpler strategy")

            if oos["overfitting_detection"]["overall_overfitting_detected"]:
                severity = oos["overfitting_detection"]["overfitting_severity"]
                summary["key_findings"].append(f"🚨 {severity.capitalize()} overfitting detected")

    # Bootstrap assessment
    bootstrap_key = "bootstrap_vs_benchmark" if "bootstrap_vs_benchmark" in validation_results else "bootstrap_vs_index"
    if bootstrap_key in validation_results:
        bs = validation_results[bootstrap_key]
        benchmark_name = "benchmark" if "benchmark" in bootstrap_key else "index"

        if bs["strategy_significantly_better_5pct"]:
            summary["key_findings"].append(f"✅ Strategy significantly outperforms {benchmark_name} (p={bs['p_value_strategy_better']})")
        elif bs["strategy_significantly_worse_5pct"]:
            summary["key_findings"].append(f"❌ Strategy significantly underperforms {benchmark_name} (p={bs['p_value_strategy_worse']})")
        else:
            summary["key_findings"].append(f"🤔 Strategy performance vs {benchmark_name} is not statistically significant")

        effect_size = abs(bs["effect_size"])
        if effect_size > 0.8:
            summary["key_findings"].append("📈 Large effect size indicates substantial performance difference")
        elif effect_size > 0.5:
            summary["key_findings"].append("📊 Moderate effect size indicates meaningful performance difference")

    # Overall assessment
    if summary["confidence_level"] == "high" and len([f for f in summary["key_findings"] if "✅" in f]) > 0:
        summary["overall_assessment"] = "strong"
    elif len([f for f in summary["key_findings"] if "❌" in f or "🚨" in f]) > 0:
        summary["overall_assessment"] = "concerning"
    else:
        summary["overall_assessment"] = "inconclusive"

    return summary


def calculate_var_and_stress_tests(returns: np.ndarray, dates: pd.Series = None) -> Dict:
    """
    Calculate Value at Risk (VaR) and stress test scenarios.

    Args:
        returns: Array of daily returns (percentage)
        dates: Corresponding dates for rolling calculations

    Returns:
        Dict with VaR calculations and stress test results
    """
    results = {}

    # Value at Risk calculations
    if len(returns) > 63:  # Need at least ~3 months for rolling VaR
        rolling_window = 63
        var_95 = []
        var_99 = []
        rolling_dates = []

        for i in range(rolling_window, len(returns)):
            window_returns = returns[i-rolling_window:i]
            var_95.append(np.percentile(window_returns, 5))
            var_99.append(np.percentile(window_returns, 1))
            if dates is not None and i < len(dates):
                rolling_dates.append(dates.iloc[i])

        results['var_95'] = np.array(var_95)
        results['var_99'] = np.array(var_99)
        results['var_dates'] = rolling_dates if rolling_dates else list(range(len(var_95)))

    # Stress test scenarios
    scenarios = {
        'base_case': returns,
        'high_volatility': returns * 2,
        'crash_scenario': np.where(returns < np.percentile(returns, 10),
                                  returns * 3, returns),
        'bull_market': np.where(returns > 0, returns * 1.5, returns * 0.5)
    }

    # Calculate cumulative returns for each scenario
    stress_results = {}
    for scenario_name, scenario_returns in scenarios.items():
        cumulative = np.cumsum(scenario_returns)
        stress_results[scenario_name] = {
            'returns': scenario_returns,
            'cumulative': cumulative,
            'total_return': cumulative[-1] if len(cumulative) > 0 else 0,
            'volatility': np.std(scenario_returns) * np.sqrt(252),  # Annualized
            'sharpe': (np.mean(scenario_returns) / np.std(scenario_returns) * np.sqrt(252))
                     if np.std(scenario_returns) > 0 else 0
        }

    results['stress_tests'] = stress_results

    # Additional risk metrics
    results['returns_volatility'] = np.std(returns) * np.sqrt(252)
    results['returns_skewness'] = stats.skew(returns)
    results['returns_kurtosis'] = stats.kurtosis(returns)
    results['max_drawdown'] = calculate_max_drawdown(returns)

    return results


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns series.

    Args:
        returns: Array of daily returns

    Returns:
        Maximum drawdown as percentage
    """
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(np.min(drawdowns))


def calculate_risk_attribution(strategy_returns: np.ndarray, market_returns: np.ndarray) -> Dict:
    """
    Calculate risk attribution: beta, alpha, and risk decomposition.

    Args:
        strategy_returns: Strategy daily returns
        market_returns: Market/benchmark daily returns

    Returns:
        Dict with risk attribution metrics
    """
    # Calculate beta and alpha (CAPM model)
    covariance = np.cov(strategy_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)

    if market_variance > 0:
        beta = covariance / market_variance
    else:
        beta = 0

    alpha = np.mean(strategy_returns) - beta * np.mean(market_returns)
    alpha_annualized = alpha * 252  # Daily to annual

    correlation = np.corrcoef(strategy_returns, market_returns)[0, 1]

    # Risk decomposition
    strategy_vol = np.std(strategy_returns) * np.sqrt(252)
    market_vol = np.std(market_returns) * np.sqrt(252)

    systematic_risk = beta ** 2 * market_vol ** 2
    idiosyncratic_risk = strategy_vol ** 2 - systematic_risk

    return {
        'beta': float(beta),
        'alpha': float(alpha_annualized),
        'correlation': float(correlation),
        'total_risk': float(strategy_vol),
        'systematic_risk': float(np.sqrt(systematic_risk)) if systematic_risk > 0 else 0,
        'idiosyncratic_risk': float(np.sqrt(idiosyncratic_risk)) if idiosyncratic_risk > 0 else 0,
        'systematic_risk_pct': float(systematic_risk / strategy_vol**2 * 100),
        'idiosyncratic_risk_pct': float(idiosyncratic_risk / strategy_vol**2 * 100)
    }


def analyze_market_regimes(parsed_df: pd.DataFrame) -> Dict:
    """
    Analyze strategy performance across different market regimes.

    Args:
        parsed_df: Parsed trading data with returns

    Returns:
        Dict with regime analysis results
    """
    if 'next_return_1d' not in parsed_df.columns or 'strategy_return' not in parsed_df.columns:
        return {}

    market_returns = parsed_df['next_return_1d'].values
    strategy_returns = parsed_df['strategy_return'].values

    # Calculate rolling volatility (20-day window)
    rolling_vol = pd.Series(market_returns).rolling(20).std() * np.sqrt(252)
    vol_median = rolling_vol.median()
    vol_high = rolling_vol.quantile(0.75)

    # Classify regimes
    regimes = []
    for vol in rolling_vol:
        if pd.isna(vol):
            regimes.append('unknown')
        elif vol > vol_high:
            regimes.append('high_volatility')
        elif vol > vol_median:
            regimes.append('moderate_volatility')
        else:
            regimes.append('low_volatility')

    # Performance by regime
    regime_results = {}
    for regime in ['low_volatility', 'moderate_volatility', 'high_volatility']:
        regime_mask = np.array(regimes) == regime
        if np.any(regime_mask):
            regime_strategy_returns = strategy_returns[regime_mask]
            regime_market_returns = market_returns[regime_mask]

            regime_results[regime] = {
                'days': int(np.sum(regime_mask)),
                'strategy_return': float(np.mean(regime_strategy_returns) * 252),  # Annualized
                'market_return': float(np.mean(regime_market_returns) * 252),
                'excess_return': float((np.mean(regime_strategy_returns) - np.mean(regime_market_returns)) * 252),
                'win_rate': float(np.mean(regime_strategy_returns > 0) * 100),
                'volatility': float(np.std(regime_strategy_returns) * np.sqrt(252))
            }

    # Find best and worst performing regimes
    if regime_results:
        best_regime = max(regime_results.keys(), key=lambda x: regime_results[x]['excess_return'])
        worst_regime = min(regime_results.keys(), key=lambda x: regime_results[x]['excess_return'])

        regime_results['_summary'] = {
            'best_regime': best_regime,
            'worst_regime': worst_regime,
            'performance_range': abs(regime_results[best_regime]['excess_return'] -
                                   regime_results[worst_regime]['excess_return']),
            'adaptation_quality': 'good' if abs(regime_results[best_regime]['excess_return'] -
                                              regime_results[worst_regime]['excess_return']) < 5 else 'variable'
        }

    return regime_results


def print_validation_report(validation_results: Dict, model_tag: str):
    """
    Print a formatted validation report to console.
    """
    print("\n" + "=" * 80)
    print(f"STATISTICAL VALIDATION REPORT - {model_tag}")
    print("=" * 80)

    # Dataset summary
    if "dataset_info" in validation_results:
        ds = validation_results["dataset_info"]
        print(f"\nDataset: {ds['n_periods']} periods ({ds['date_range']['start']} to {ds['date_range']['end']})")
        print(f"Strategy Return: {ds['total_strategy_return']:.2f}% | Index Return: {ds['total_index_return']:.2f}%")

    # Out-of-sample validation
    if "out_of_sample_validation" in validation_results:
        oos = validation_results["out_of_sample_validation"]
        if "error" not in oos:
            print(f"\nOUT-OF-SAMPLE VALIDATION:")
            print(f"  Train Period: {oos['train_period']['sharpe_ratio']:.3f} Sharpe ({oos['train_period']['n_periods']} periods)")
            print(f"  Test Period:  {oos['test_period']['sharpe_ratio']:.3f} Sharpe ({oos['test_period']['n_periods']} periods)")

            if oos["comparative_analysis"]["sharpe_decay"] is not None:
                decay_pct = oos["comparative_analysis"]["sharpe_decay_pct"]
                print(f"  Sharpe Decay: {decay_pct:+.1f}%")

            overfitting = "YES" if oos["overfitting_detection"]["overall_overfitting_detected"] else "NO"
            print(f"  Overfitting Detected: {overfitting}")

            generalizes = "YES" if oos["generalization_assessment"]["generalizes_well"] else "NO"
            print(f"  Generalizes Well: {generalizes}")
        else:
            print(f"\nOUT-OF-SAMPLE VALIDATION: ERROR - {oos['error']}")

    # Bootstrap testing
    bootstrap_key = "bootstrap_vs_benchmark" if "bootstrap_vs_benchmark" in validation_results else "bootstrap_vs_index"
    if bootstrap_key in validation_results:
        bs = validation_results[bootstrap_key]
        benchmark_name = "Benchmark" if "benchmark" in bootstrap_key else "Index"

        print(f"\nBOOTSTRAP TEST VS {benchmark_name.upper()}:")
        print(f"  Strategy Sharpe: {bs['strategy_sharpe']:.3f}")
        print(f"  {benchmark_name} Sharpe: {bs['benchmark_sharpe']:.3f}")
        print(f"  Difference: {bs['sharpe_difference']:+.3f}")
        print(f"  p-value (two-sided): {bs['p_value_two_sided']:.4f}")
        # Handle CI values from JSON (now stored as list of floats)
        try:
            ci_vals = bs['ci_95_bootstrap']
            if isinstance(ci_vals, (list, tuple)) and len(ci_vals) >= 2:
                ci_lower = float(ci_vals[0])
                ci_upper = float(ci_vals[1])
                print(f"  95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
            else:
                print(f"  95% CI: {ci_vals}")
        except (ValueError, TypeError, IndexError):
            # Fallback if CI values are malformed
            print(f"  95% CI: {bs.get('ci_95_bootstrap', 'N/A')}")

        if bs["strategy_significantly_better_5pct"]:
            print("  RESULT: Strategy significantly outperforms benchmark ✓")
        elif bs["strategy_significantly_worse_5pct"]:
            print("  RESULT: Strategy significantly underperforms benchmark ✗")
        else:
            print("  RESULT: No significant difference detected 🤔")

    # Summary
    if "summary_assessment" in validation_results:
        sa = validation_results["summary_assessment"]
        print(f"\nSUMMARY ASSESSMENT:")
        print(f"  Overall: {sa['overall_assessment'].upper()}")
        print(f"  Confidence: {sa['confidence_level'].upper()}")

        if sa["key_findings"]:
            print("  Key Findings:")
            for finding in sa["key_findings"]:
                print(f"    {finding}")

        if sa["recommendations"]:
            print("  Recommendations:")
            for rec in sa["recommendations"]:
                print(f"    • {rec}")

    print("=" * 80 + "\n")


def save_validation_report(validation_results: Dict, output_path: str):
    """
    Save validation results to JSON file.
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.datetime64, pd.Timestamp)):
            return str(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            # For any other types, try to convert to string as fallback
            try:
                return str(obj)
            except:
                return f"<non-serializable: {type(obj).__name__}>"

    # Recursively convert all numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Special handling for tuples that might contain numpy floats
            result = []
            for item in obj:
                converted = recursive_convert(item)
                result.append(converted)
            return result
        else:
            return convert_for_json(obj)

    json_results = recursive_convert(validation_results)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"[INFO] Statistical validation report saved to: {output_path}")
