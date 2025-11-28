# src/baselines.py
"""
Baseline trading strategies for comparison with LLM decisions.

These are deterministic, rule-based strategies that use the same input features
as the LLM. They answer the question: "Does the LLM add value beyond simple rules?"

Baseline Hierarchy:
1. Random        - Pure noise (like dummy_model, but for backtesting)
2. Buy-and-Hold  - Always long, no timing
3. Rule-based    - Simple trading rules using technical features
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List


# =============================================================================
# BASELINE STRATEGY FUNCTIONS
# =============================================================================

def random_baseline(features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Random baseline: uniformly random BUY/HOLD/SELL decisions.
    
    Purpose: Establish the "pure noise" baseline. Any useful strategy should
    significantly outperform random decisions.
    
    Expected return: ~0 (minus transaction costs)
    """
    np.random.seed(seed)
    df = features_df.copy()
    
    decisions = np.random.choice(["BUY", "HOLD", "SELL"], size=len(df))
    df["decision"] = decisions
    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


def buy_and_hold_baseline(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Buy-and-Hold baseline: always long the index.
    
    Purpose: The most basic passive strategy. Any active strategy (including LLM)
    should aim to beat this, otherwise why bother trading?
    
    This is your "benchmark to beat" for any timing strategy.
    """
    df = features_df.copy()
    
    df["decision"] = "BUY"
    df["position"] = 1.0
    df["strategy_return"] = df["next_return_1d"]  # Just the index return
    
    return df


def momentum_baseline(features_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Simple Momentum baseline: BUY when 20-day trend is positive, HOLD otherwise.
    
    Logic: "The trend is your friend" - ride positive momentum, avoid negative.
    
    Parameters:
        threshold: Minimum ma20_pct to trigger BUY (default 0 = any positive)
    
    This is a classic trend-following approach. If the LLM can't beat this,
    it's not learning anything useful about momentum.
    """
    df = features_df.copy()
    
    df["decision"] = np.where(df["ma20_pct"] > threshold, "BUY", "HOLD")
    df["position"] = np.where(df["ma20_pct"] > threshold, 1.0, 0.0)
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


def contrarian_baseline(features_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Contrarian baseline: BUY when 20-day trend is negative (expect mean reversion).
    
    Logic: "Buy the dip" - markets tend to revert, so buy after declines.
    
    This is the opposite of momentum. Useful to see if the LLM has learned
    momentum vs contrarian behavior.
    """
    df = features_df.copy()
    
    df["decision"] = np.where(df["ma20_pct"] < threshold, "BUY", "HOLD")
    df["position"] = np.where(df["ma20_pct"] < threshold, 1.0, 0.0)
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


def mean_reversion_baseline(
    features_df: pd.DataFrame, 
    buy_threshold: float = -2.0,
    sell_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Mean Reversion baseline: BUY after sharp drops, SELL after sharp rallies.
    
    Logic: Short-term overreactions tend to reverse. Buy oversold, sell overbought.
    
    Parameters:
        buy_threshold: BUY if 5-day return below this (default -2%)
        sell_threshold: SELL if 5-day return above this (default +2%)
    """
    df = features_df.copy()
    
    conditions = [
        df["ret_5d"] < buy_threshold,   # Oversold → BUY
        df["ret_5d"] > sell_threshold,  # Overbought → SELL
    ]
    choices = ["BUY", "SELL"]
    df["decision"] = np.select(conditions, choices, default="HOLD")
    
    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


def volatility_timing_baseline(
    features_df: pd.DataFrame, 
    vol_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Volatility Timing baseline: Stay invested in low vol, exit in high vol.
    
    Logic: High volatility often precedes or accompanies market declines.
    Stay in cash during turbulent periods to reduce drawdowns.
    
    Parameters:
        vol_threshold: Exit to cash if annualized vol exceeds this (default 20%)
    """
    df = features_df.copy()
    
    df["decision"] = np.where(df["vol20_annualized"] < vol_threshold, "BUY", "HOLD")
    df["position"] = np.where(df["vol20_annualized"] < vol_threshold, 1.0, 0.0)
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


def combined_momentum_vol_baseline(
    features_df: pd.DataFrame,
    vol_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Combined baseline: Momentum + Volatility filter.
    
    Logic: Only buy when trend is positive AND volatility is low.
    This combines trend-following with risk management.
    
    This is a more sophisticated baseline - if the LLM can't beat this,
    it's not combining signals effectively.
    """
    df = features_df.copy()
    
    # BUY only when: positive trend AND low volatility
    buy_condition = (df["ma20_pct"] > 0) & (df["vol20_annualized"] < vol_threshold)
    
    df["decision"] = np.where(buy_condition, "BUY", "HOLD")
    df["position"] = np.where(buy_condition, 1.0, 0.0)
    df["strategy_return"] = df["position"] * df["next_return_1d"]
    
    return df


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_baseline_metrics(df: pd.DataFrame, baseline_name: str) -> Dict:
    """
    Calculate performance metrics for a baseline strategy.
    
    Returns dict with: total_return, sharpe, max_drawdown, win_rate, etc.
    """
    returns = df["strategy_return"]
    
    # Basic stats
    total_return = returns.sum()
    mean_return = returns.mean()
    volatility = returns.std()
    sharpe = (mean_return / volatility * np.sqrt(252)) if volatility > 0 else 0.0
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Max drawdown
    equity = (1 + returns / 100).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Decision distribution
    decision_counts = df["decision"].value_counts()
    buy_pct = decision_counts.get("BUY", 0) / len(df) * 100
    hold_pct = decision_counts.get("HOLD", 0) / len(df) * 100
    sell_pct = decision_counts.get("SELL", 0) / len(df) * 100
    
    # Position changes (for transaction cost estimation)
    position_changes = (df["position"].diff().abs() > 0).sum()
    
    return {
        "baseline": baseline_name,
        "total_return": round(total_return, 2),
        "mean_daily_return": round(mean_return, 4),
        "volatility": round(volatility, 4),
        "sharpe_annualized": round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100, 2),  # As percentage
        "win_rate": round(win_rate * 100, 2),
        "buy_pct": round(buy_pct, 1),
        "hold_pct": round(hold_pct, 1),
        "sell_pct": round(sell_pct, 1),
        "position_changes": int(position_changes),
        "n_days": len(df),
    }


# =============================================================================
# MAIN BASELINE RUNNER
# =============================================================================

# Registry of all available baselines
BASELINE_REGISTRY = {
    "random": random_baseline,
    "buy_and_hold": buy_and_hold_baseline,
    "momentum": momentum_baseline,
    "contrarian": contrarian_baseline,
    "mean_reversion": mean_reversion_baseline,
    "volatility_timing": volatility_timing_baseline,
    "momentum_vol_combined": combined_momentum_vol_baseline,
}


def run_all_baselines(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all baseline strategies and return a comparison DataFrame.
    
    Args:
        features_df: DataFrame with columns: date, ma20_pct, ret_5d, 
                     vol20_annualized, next_return_1d
    
    Returns:
        DataFrame with one row per baseline, columns for each metric
    """
    results = []
    
    for name, baseline_fn in BASELINE_REGISTRY.items():
        try:
            baseline_df = baseline_fn(features_df)
            metrics = calculate_baseline_metrics(baseline_df, name)
            results.append(metrics)
        except Exception as e:
            print(f"[WARN] Baseline '{name}' failed: {e}")
            continue
    
    return pd.DataFrame(results)


def compare_llm_to_baselines(
    llm_metrics: Dict, 
    baseline_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare LLM performance against all baselines.
    
    Args:
        llm_metrics: Dict with LLM performance (from backtest_model)
        baseline_results: DataFrame from run_all_baselines
    
    Returns:
        DataFrame with baselines + LLM row, sorted by total_return
    """
    # Convert LLM metrics to same format
    llm_row = {
        "baseline": "LLM_STRATEGY",
        "total_return": round(llm_metrics.get("total_return", 0), 2),
        "mean_daily_return": round(llm_metrics.get("mean_return", 0), 4),
        "volatility": round(llm_metrics.get("volatility", 0), 4),
        "sharpe_annualized": round(
            llm_metrics.get("sharpe_like", 0) * np.sqrt(252) 
            if llm_metrics.get("sharpe_like") else 0, 3
        ),
        "max_drawdown": round(llm_metrics.get("max_drawdown", 0) * 100, 2),
        "win_rate": round(llm_metrics.get("hit_rate", 0) * 100, 2),
        "buy_pct": None,  # Would need parsed_df to compute
        "hold_pct": None,
        "sell_pct": None,
        "position_changes": None,
        "n_days": None,
    }
    
    # Create LLM DataFrame with explicit dtypes matching baseline_results
    llm_df = pd.DataFrame([llm_row])
    
    # Ensure columns match and concat without warnings
    for col in baseline_results.columns:
        if col not in llm_df.columns:
            llm_df[col] = pd.NA
    
    # Reorder columns to match baseline_results
    llm_df = llm_df[baseline_results.columns]
    
    all_results = pd.concat([baseline_results, llm_df], ignore_index=True)
    
    # Sort by total return (best at top)
    all_results = all_results.sort_values("total_return", ascending=False)
    
    return all_results


def print_baseline_comparison(comparison_df: pd.DataFrame, model_tag: str = "LLM"):
    """
    Print a formatted comparison table to console.
    """
    print("\n" + "=" * 80)
    print(f"BASELINE COMPARISON - {model_tag}")
    print("=" * 80)
    
    # Find LLM row for highlighting
    llm_return = comparison_df[
        comparison_df["baseline"] == "LLM_STRATEGY"
    ]["total_return"].values
    llm_return = llm_return[0] if len(llm_return) > 0 else None
    
    print(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Win%':>10}")
    print("-" * 65)
    
    for _, row in comparison_df.iterrows():
        name = row["baseline"]
        ret = row["total_return"]
        sharpe = row["sharpe_annualized"]
        maxdd = row["max_drawdown"]
        winrate = row["win_rate"]
        
        # Highlight LLM row
        marker = " ◄" if name == "LLM_STRATEGY" else ""
        
        # Color indicator (text only, no actual ANSI codes for compatibility)
        if llm_return is not None and name != "LLM_STRATEGY":
            if ret < llm_return:
                indicator = " ✓"  # LLM beats this baseline
            else:
                indicator = " ✗"  # LLM loses to this baseline
        else:
            indicator = ""
        
        print(f"{name:<25} {ret:>9.2f}% {sharpe:>10.3f} {maxdd:>9.2f}% {winrate:>9.1f}%{indicator}{marker}")
    
    print("-" * 65)
    
    # Summary
    if llm_return is not None:
        baselines_beaten = (
            comparison_df[comparison_df["baseline"] != "LLM_STRATEGY"]["total_return"] 
            < llm_return
        ).sum()
        total_baselines = len(comparison_df) - 1
        
        print(f"\nLLM beats {baselines_beaten}/{total_baselines} baselines")
        
        # Key comparisons
        bh_return = comparison_df[
            comparison_df["baseline"] == "buy_and_hold"
        ]["total_return"].values
        if len(bh_return) > 0:
            diff = llm_return - bh_return[0]
            status = "OUTPERFORMS" if diff > 0 else "UNDERPERFORMS"
            print(f"vs Buy-and-Hold: {status} by {abs(diff):.2f}%")
        
        momentum_return = comparison_df[
            comparison_df["baseline"] == "momentum"
        ]["total_return"].values
        if len(momentum_return) > 0:
            diff = llm_return - momentum_return[0]
            status = "OUTPERFORMS" if diff > 0 else "UNDERPERFORMS"
            print(f"vs Simple Momentum: {status} by {abs(diff):.2f}%")
    
    print("=" * 80 + "\n")


# =============================================================================
# ENHANCED STATISTICAL ROBUSTNESS FUNCTIONS
# =============================================================================

def robust_random_baseline(features_df: pd.DataFrame, n_runs: int = 30, confidence_level: float = 0.95) -> Dict:
    """
    Run random baseline multiple times to provide statistical robustness.

    Args:
        features_df: Input features DataFrame
        n_runs: Number of random runs (default 30 for good statistics)
        confidence_level: Confidence level for intervals (default 95%)

    Returns:
        Dict with statistical summary of random baseline performance
    """
    results = []

    for seed in range(n_runs):
        # Run random baseline with different seed
        baseline_df = random_baseline(features_df, seed=seed)
        metrics = calculate_baseline_metrics(baseline_df, f"random_{seed}")
        results.append(metrics['total_return'])

    results_array = np.array(results)

    # Calculate statistics
    mean_return = np.mean(results_array)
    std_return = np.std(results_array, ddof=1)  # Sample standard deviation
    median_return = np.median(results_array)

    # Confidence interval
    if n_runs > 1:
        # t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence_level) / 2, n_runs - 1)
        margin_error = t_value * std_return / np.sqrt(n_runs)
        ci_lower = mean_return - margin_error
        ci_upper = mean_return + margin_error
    else:
        ci_lower = ci_upper = mean_return

    # Additional statistics
    min_return = np.min(results_array)
    max_return = np.max(results_array)
    q25, q75 = np.percentile(results_array, [25, 75])

    return {
        'mean': round(mean_return, 2),
        'std': round(std_return, 2),
        'median': round(median_return, 2),
        'ci_lower': round(ci_lower, 2),
        'ci_upper': round(ci_upper, 2),
        'min': round(min_return, 2),
        'max': round(max_return, 2),
        'q25': round(q25, 2),
        'q75': round(q75, 2),
        'n_runs': n_runs,
        'confidence_level': confidence_level,
        'all_results': results_array.tolist()  # For further analysis
    }


def calculate_llm_vs_random_stats(llm_return: float, random_stats: Dict) -> Dict:
    """
    Calculate statistical significance of LLM vs random baseline.

    Args:
        llm_return: LLM strategy total return
        random_stats: Output from robust_random_baseline()

    Returns:
        Dict with statistical test results
    """
    all_random = np.array(random_stats['all_results'])

    # One-sample t-test: is LLM significantly different from random mean?
    t_stat, p_value = stats.ttest_1samp(all_random, llm_return)

    # Effect size (Cohen's d)
    effect_size = (llm_return - random_stats['mean']) / random_stats['std']

    # Percentage of random runs that LLM beats
    beats_random_pct = (llm_return > all_random).mean() * 100

    # Bayesian approach: probability LLM beats random
    # Using normal approximation
    from scipy.stats import norm
    prob_llm_beats_random = 1 - norm.cdf(llm_return, random_stats['mean'], random_stats['std'])

    return {
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 4),
        'effect_size': round(effect_size, 3),
        'beats_random_percent': round(beats_random_pct, 1),
        'prob_llm_beats_random': round(prob_llm_beats_random, 3),
        'significant': p_value < 0.05,
        'llm_vs_random_mean': round(llm_return - random_stats['mean'], 2)
    }


def enhanced_compare_llm_to_baselines(
    llm_metrics: Dict,
    baseline_results: pd.DataFrame,
    features_df: pd.DataFrame,
    n_random_runs: int = 30
) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced comparison with statistical robustness for random baseline.

    Returns:
        Tuple of (comparison_df, random_stats_dict)
    """
    # Get robust random baseline statistics
    random_stats = robust_random_baseline(features_df, n_runs=n_random_runs)

    # Create random baseline row with statistical info
    random_row = {
        "baseline": f"random_mean (n={n_random_runs})",
        "total_return": random_stats['mean'],
        "mean_daily_return": None,  # Not meaningful for aggregate
        "volatility": random_stats['std'],  # Use std as proxy for volatility
        "sharpe_annualized": None,
        "max_drawdown": None,
        "win_rate": None,
        "buy_pct": 33.3,  # Theoretical uniform distribution
        "hold_pct": 33.3,
        "sell_pct": 33.3,
        "position_changes": None,
        "n_days": None,
    }

    # Convert to DataFrame
    random_df = pd.DataFrame([random_row])

    # Remove old random baseline and add new one
    baseline_results = baseline_results[baseline_results['baseline'] != 'random']
    enhanced_baselines = pd.concat([baseline_results, random_df], ignore_index=True)

    # Add LLM and sort
    all_results = compare_llm_to_baselines(llm_metrics, enhanced_baselines)

    return all_results, random_stats


def print_enhanced_baseline_comparison(
    comparison_df: pd.DataFrame,
    random_stats: Dict,
    llm_stats: Dict,
    model_tag: str = "LLM"
):
    """
    Print enhanced comparison with statistical details.
    """
    print("\n" + "=" * 100)
    print(f"ENHANCED BASELINE COMPARISON - {model_tag}")
    print("=" * 100)

    # Standard comparison table
    print(f"\n{'Strategy':<30} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Win%':>10}")
    print("-" * 80)

    for _, row in comparison_df.iterrows():
        name = row["baseline"]
        ret = row["total_return"] if pd.notna(row["total_return"]) else "N/A"
        sharpe = row["sharpe_annualized"] if pd.notna(row["sharpe_annualized"]) else "N/A"
        maxdd = row["max_drawdown"] if pd.notna(row["max_drawdown"]) else "N/A"
        winrate = row["win_rate"] if pd.notna(row["win_rate"]) else "N/A"

        marker = " ◄" if name == "LLM_STRATEGY" else ""
        print(f"{name:<30} {ret:>9} {sharpe:>10} {maxdd:>9} {winrate:>9}{marker}")

    print("-" * 80)

    # Enhanced random statistics
    print(f"\nRANDOM BASELINE STATISTICS (n={random_stats['n_runs']} runs):")
    print(f"  Mean Return: {random_stats['mean']:.2f}%")
    print(f"  95% CI: [{random_stats['ci_lower']:.2f}%, {random_stats['ci_upper']:.2f}%]")
    print(f"  Std Dev: {random_stats['std']:.2f}%")
    print(f"  Range: [{random_stats['min']:.2f}%, {random_stats['max']:.2f}%]")
    print(f"  Quartiles: Q25={random_stats['q25']:.2f}%, Median={random_stats['median']:.2f}%, Q75={random_stats['q75']:.2f}%")

    # Statistical significance
    if llm_stats:
        print(f"\nSTATISTICAL SIGNIFICANCE vs RANDOM:")
        print(f"  LLM outperforms random by: {llm_stats['llm_vs_random_mean']:+.2f}%")
        print(f"  Beats random in: {llm_stats['beats_random_percent']:.1f}% of runs")
        print(f"  Effect size: {llm_stats['effect_size']:+.3f} (Cohen's d)")
        print(f"  p-value: {llm_stats['p_value']:.4f} ({'SIGNIFICANT' if llm_stats['significant'] else 'not significant'})")
        print(f"  Probability LLM > Random: {llm_stats['prob_llm_beats_random']:.1%}")

    print("=" * 100 + "\n")

