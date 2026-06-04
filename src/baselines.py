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

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import get_current_symbol_info
from .constants import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

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


def momentum_baseline(
    features_df: pd.DataFrame, threshold: float = 0.0
) -> pd.DataFrame:
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


def contrarian_baseline(
    features_df: pd.DataFrame, threshold: float = 0.0
) -> pd.DataFrame:
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
    features_df: pd.DataFrame, buy_threshold: float = -2.0, sell_threshold: float = 2.0
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
        df["ret_5d"] < buy_threshold,  # Oversold → BUY
        df["ret_5d"] > sell_threshold,  # Overbought → SELL
    ]
    choices = ["BUY", "SELL"]
    df["decision"] = np.select(conditions, choices, default="HOLD")

    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def volatility_timing_baseline(
    features_df: pd.DataFrame, vol_threshold: float = 20.0
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
    features_df: pd.DataFrame, vol_threshold: float = 20.0
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


def rsi_mean_reversion_baseline(
    features_df: pd.DataFrame, overbought: float = 70, oversold: float = 30
) -> pd.DataFrame:
    """
    RSI Mean Reversion Strategy:
    - BUY when RSI < oversold (expect bounce up from oversold levels)
    - SELL when RSI > overbought (expect pullback down from overbought levels)
    - HOLD otherwise

    This strategy assumes that extreme RSI levels tend to revert to the mean.
    """
    df = features_df.copy()

    conditions = [
        df["rsi_14"] < oversold,  # Oversold → BUY
        df["rsi_14"] > overbought,  # Overbought → SELL
    ]
    choices = ["BUY", "SELL"]
    df["decision"] = np.select(conditions, choices, default="HOLD")

    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def rsi_contrarian_baseline(
    features_df: pd.DataFrame, rsi_threshold: float = 40
) -> pd.DataFrame:
    """
    RSI Contrarian Strategy:
    - BUY when RSI is low (below threshold) - fade the weakness
    - SELL when RSI is high (above 100-threshold) - fade the strength
    - HOLD in neutral RSI territory

    This is the opposite of mean reversion - bets against RSI extremes.
    """
    df = features_df.copy()

    conditions = [
        df["rsi_14"] < rsi_threshold,  # Fade weakness
        df["rsi_14"] > (100 - rsi_threshold),  # Fade strength
    ]
    choices = ["BUY", "SELL"]
    df["decision"] = np.select(conditions, choices, default="HOLD")

    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def macd_momentum_baseline(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD Momentum Strategy: BUY when MACD crosses above signal line, SELL below.

    This is the most basic MACD strategy - trend following based on momentum shifts.
    Classic "MACD crossover" system used by many traders.

    Uses consecutive BUY/SELL signals to maintain positions until exit signal.
    HOLD only used when exiting to cash.

    Logic:
    - BUY when MACD line first crosses above signal line
    - Stay long until MACD line crosses below signal line
    - SELL when MACD line first crosses below signal line
    - Stay short until MACD line crosses above signal line

    This provides trend-following signals based on momentum acceleration.

    Args:
        features_df: DataFrame with macd_line, macd_signal columns

    Returns:
        DataFrame with decision, position, strategy_return columns
    """
    df = features_df.copy()

    # MACD crossover logic: BUY when macd_line > macd_signal, SELL when <
    # Use diff() to detect crossovers (sign changes)
    macd_diff = df["macd_line"] - df["macd_signal"]
    crossover_up = (macd_diff > 0) & (macd_diff.shift(1) <= 0)  # Crossed above
    crossover_down = (macd_diff < 0) & (macd_diff.shift(1) >= 0)  # Crossed below

    # Track signal state: BUY, SELL, or HOLD (cash)
    signal_state = pd.Series("HOLD", index=df.index)  # Start in cash

    for i in range(1, len(df)):
        if crossover_up.iloc[i]:
            signal_state.iloc[i] = "BUY"  # Entry: go long
        elif crossover_down.iloc[i]:
            signal_state.iloc[i] = "SELL"  # Entry: go short
        else:
            signal_state.iloc[i] = signal_state.iloc[i - 1]  # Maintain signal

    # Map signal state to positions
    df["decision"] = signal_state
    df["position"] = signal_state.map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def macd_histogram_baseline(
    features_df: pd.DataFrame, threshold: float = 0.0
) -> pd.DataFrame:
    """
    MACD Histogram Momentum Strategy: BUY/SELL based on histogram strength.

    This strategy uses the MACD histogram to identify momentum acceleration:
    - BUY when histogram > threshold (strong bullish momentum)
    - SELL when histogram < -threshold (strong bearish momentum)
    - HOLD otherwise (neutral momentum)

    Args:
        features_df: DataFrame with macd_histogram column
        threshold: Minimum histogram value for signals (default 0.0)

    Returns:
        DataFrame with decision, position, strategy_return columns
    """
    df = features_df.copy()

    # MACD Histogram signals: momentum strength
    conditions = [
        df["macd_histogram"] > threshold,  # Strong bullish momentum
        df["macd_histogram"] < -threshold,  # Strong bearish momentum
    ]
    choices = ["BUY", "SELL"]
    df["decision"] = np.select(conditions, choices, default="HOLD")

    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def stochastic_baseline(
    features_df: pd.DataFrame, overbought: float = 80, oversold: float = 20
) -> pd.DataFrame:
    """
    Stochastic Oscillator Mean Reversion Strategy.

    This classic momentum strategy buys when the oscillator shows oversold conditions
    and sells when it shows overbought conditions.

    Uses consecutive BUY/SELL signals to maintain positions until exit signal.
    HOLD only used when exiting to cash.

    Logic:
    - BUY when %K first crosses above oversold level (default 20)
    - Stay long until %K crosses below overbought level (default 80)
    - SELL when %K first crosses below overbought level (default 80)
    - Stay short until %K crosses above oversold level (default 20)

    This complements RSI with different calculation method for momentum signals.

    Args:
        features_df: DataFrame with stoch_k, stoch_d columns
        overbought: Overbought threshold (default 80)
        oversold: Oversold threshold (default 20)

    Returns:
        DataFrame with decision, position, strategy_return columns
    """
    df = features_df.copy()

    # Stochastic crossover signals
    stoch_k = df["stoch_k"]

    # BUY: %K crosses above oversold level
    buy_signal = (stoch_k > oversold) & (stoch_k.shift(1) <= oversold)

    # SELL: %K crosses below overbought level
    sell_signal = (stoch_k < overbought) & (stoch_k.shift(1) >= overbought)

    # Track signal state: BUY, SELL, or HOLD (cash)
    signal_state = pd.Series("HOLD", index=df.index)  # Start in cash

    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            signal_state.iloc[i] = "BUY"  # Entry: go long
        elif sell_signal.iloc[i]:
            signal_state.iloc[i] = "SELL"  # Entry: go short
        else:
            signal_state.iloc[i] = signal_state.iloc[i - 1]  # Maintain signal

    # Map signal state to positions
    df["decision"] = signal_state
    df["position"] = signal_state.map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def bollinger_reversion_baseline(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Band Mean Reversion Strategy.

    This classic volatility-based strategy buys when price touches the lower Bollinger Band
    and sells when price touches the upper Bollinger Band, expecting mean reversion.

    Uses consecutive BUY/SELL signals to maintain positions until opposite band touch.
    HOLD only used when exiting to cash.

    Logic:
    - BUY when price first touches lower band (entry long)
    - Stay long until price touches upper band (exit long)
    - SELL when price first touches upper band (entry short)
    - Stay short until price touches lower band (exit short)

    This complements momentum-based strategies (RSI, Stochastic) with volatility signals.
    """
    df = features_df.copy()

    # Bollinger Band signals - detect first touches (crossovers)
    price = df["close"]
    upper_band = df["bb_upper"]
    lower_band = df["bb_lower"]

    # BUY: Price crosses below lower band (first touch)
    buy_signal = (price <= lower_band) & (price.shift(1) > lower_band)

    # SELL: Price crosses above upper band (first touch)
    sell_signal = (price >= upper_band) & (price.shift(1) < upper_band)

    # Track signal state: BUY, SELL, or HOLD (cash)
    signal_state = pd.Series("HOLD", index=df.index)  # Start in cash

    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            signal_state.iloc[i] = "BUY"  # Entry: go long
        elif sell_signal.iloc[i]:
            signal_state.iloc[i] = "SELL"  # Entry: go short
        else:
            signal_state.iloc[i] = signal_state.iloc[i - 1]  # Maintain signal

    # Map signal state to positions
    df["decision"] = signal_state
    df["position"] = signal_state.map({"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def macd_rsi_combined_baseline(
    features_df: pd.DataFrame, rsi_overbought: float = 70, rsi_oversold: float = 30
) -> pd.DataFrame:
    """
    MACD + RSI Combined Confirmation Strategy.

    Requires both MACD trend signal AND RSI momentum confirmation.
    More selective but higher quality signals.

    BUY = long (+1.0) when MACD bullish AND RSI oversold
    SELL = short (-1.0) when MACD bearish AND RSI overbought
    HOLD = cash (0.0) when no dual confirmation

    This combines trend-following with mean reversion for stronger signals.
    """
    df = features_df.copy()

    # MACD bullish/bearish signals (trend confirmation)
    macd_bullish = df["macd_histogram"] > 0
    macd_bearish = df["macd_histogram"] < 0

    # RSI signals (momentum confirmation)
    rsi_oversold = df["rsi_14"] < rsi_oversold
    rsi_overbought = df["rsi_14"] > rsi_overbought

    # Combined confirmation signals
    buy_signal = macd_bullish & rsi_oversold
    sell_signal = macd_bearish & rsi_overbought

    # Apply signals: HOLD (cash) by default, only trade on dual confirmation
    df["decision"] = np.select(
        [buy_signal, sell_signal], ["BUY", "SELL"], default="HOLD"
    )
    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
    df["strategy_return"] = df["position"] * df["next_return_1d"]

    return df


def stochastic_bollinger_combined_baseline(
    features_df: pd.DataFrame, stoch_overbought: float = 80, stoch_oversold: float = 20
) -> pd.DataFrame:
    """
    Stochastic + Bollinger Combined Confirmation Strategy.

    Requires both Stochastic momentum signal AND Bollinger volatility confirmation.
    Combines oscillator extremes with volatility breakouts.

    BUY = long (+1.0) when Stochastic oversold AND price breaks lower Bollinger band
    SELL = short (-1.0) when Stochastic overbought AND price breaks upper Bollinger band
    HOLD = cash (0.0) when no dual confirmation

    This provides high-confidence mean reversion signals.
    """
    df = features_df.copy()

    # Validate required columns
    required_cols = ["stoch_k", "close", "bb_lower", "bb_upper", "next_return_1d"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"stochastic_bollinger_combined_baseline missing required columns: {missing_cols}"
        )

    # Stochastic signals (momentum confirmation)
    # BUY when Stochastic becomes oversold (crosses below oversold level from above)
    stoch_buy_signal = (df["stoch_k"] < stoch_oversold) & (
        df["stoch_k"].shift(1) >= stoch_oversold
    )
    # SELL when Stochastic becomes overbought (crosses above overbought level from below)
    stoch_sell_signal = (df["stoch_k"] > stoch_overbought) & (
        df["stoch_k"].shift(1) <= stoch_overbought
    )

    # Bollinger band breakouts (volatility confirmation)
    bb_buy_signal = (df["close"] <= df["bb_lower"]) & (
        df["close"].shift(1) > df["bb_lower"]
    )
    bb_sell_signal = (df["close"] >= df["bb_upper"]) & (
        df["close"].shift(1) < df["bb_upper"]
    )

    # Combined confirmation - both momentum AND volatility must agree
    buy_signal = stoch_buy_signal & bb_buy_signal
    sell_signal = stoch_sell_signal & bb_sell_signal

    # Apply signals: HOLD (cash) by default, only trade on dual confirmation
    df["decision"] = np.select(
        [buy_signal, sell_signal], ["BUY", "SELL"], default="HOLD"
    )
    df["position"] = df["decision"].map({"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0})
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
    sharpe = (
        (mean_return / volatility * np.sqrt(TRADING_DAYS_PER_YEAR))
        if volatility > 0
        else 0.0
    )

    # Win rate
    win_rate = (returns > 0).mean()

    # Max drawdown
    equity = (1 + returns / 100).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
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
# IMPORTANT: Each strategy should only be registered once to avoid duplication
BASELINE_REGISTRY = {
    "random": random_baseline,
    "buy_and_hold": buy_and_hold_baseline,
    "momentum": momentum_baseline,
    "contrarian": contrarian_baseline,
    "mean_reversion": mean_reversion_baseline,
    "volatility_timing": volatility_timing_baseline,
    "momentum_vol_combined": combined_momentum_vol_baseline,
    "rsi_mean_reversion": rsi_mean_reversion_baseline,
    "rsi_contrarian": rsi_contrarian_baseline,
    # NEW: MACD Momentum baseline
    "macd_momentum": macd_momentum_baseline,
    # NEW: MACD Histogram baseline
    "macd_histogram": macd_histogram_baseline,
    # NEW: Stochastic Oscillator baseline
    "stochastic_oscillator": stochastic_baseline,
    # NEW: Bollinger Band Reversion baseline
    "bollinger_reversion": bollinger_reversion_baseline,
    # NEW: Combined indicator baselines
    "macd_rsi_combined": macd_rsi_combined_baseline,
    "stochastic_bollinger_combined": stochastic_bollinger_combined_baseline,
}


# =============================================================================
# STRATEGY METADATA FOR ENHANCED REPORTING
# =============================================================================

STRATEGY_METADATA = {
    # Original baselines
    "random": {
        "category": "noise",
        "indicators": [],
        "description": "Random noise baseline for statistical comparison",
    },
    "buy_and_hold": {
        "category": "passive",
        "indicators": [],
        "description": f"Buy and hold {get_current_symbol_info()[1]} (benchmark)",
    },
    "momentum": {
        "category": "trend_following",
        "indicators": ["MA20"],
        "description": "20-day moving average momentum strategy",
    },
    "contrarian": {
        "category": "mean_reversion",
        "indicators": ["MA20"],
        "description": "MA20-based contrarian signals",
    },
    "mean_reversion": {
        "category": "mean_reversion",
        "indicators": ["volatility"],
        "description": "Volatility-based mean reversion timing",
    },
    "volatility_timing": {
        "category": "risk_management",
        "indicators": ["volatility"],
        "description": "Volatility threshold risk management",
    },
    "momentum_vol_combined": {
        "category": "multi_factor",
        "indicators": ["MA20", "volatility"],
        "description": "Combined momentum and volatility factors",
    },
    # RSI strategies
    "rsi_mean_reversion": {
        "category": "mean_reversion",
        "indicators": ["RSI"],
        "description": "RSI oversold/overbought mean reversion",
    },
    "rsi_contrarian": {
        "category": "mean_reversion",
        "indicators": ["RSI"],
        "description": "RSI contrarian momentum signals",
    },
    # MACD strategies
    "macd_momentum": {
        "category": "trend_following",
        "indicators": ["MACD"],
        "description": "MACD line crossover trend signals",
    },
    "macd_histogram": {
        "category": "momentum",
        "indicators": ["MACD"],
        "description": "MACD histogram momentum acceleration",
    },
    # Stochastic strategies
    "stochastic_oscillator": {
        "category": "mean_reversion",
        "indicators": ["Stochastic"],
        "description": "Stochastic oscillator mean reversion",
    },
    # Bollinger strategies
    "bollinger_reversion": {
        "category": "mean_reversion",
        "indicators": ["Bollinger"],
        "description": "Bollinger band mean reversion",
    },
    # Combined strategies
    "macd_rsi_combined": {
        "category": "confirmation",
        "indicators": ["MACD", "RSI"],
        "description": "MACD + RSI dual confirmation signals",
    },
    "stochastic_bollinger_combined": {
        "category": "confirmation",
        "indicators": ["Stochastic", "Bollinger"],
        "description": "Stochastic + Bollinger dual confirmation signals",
    },
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
            logger.warning("Baseline '%s' failed: %s", name, e)
            continue

    df = pd.DataFrame(results)

    # Check for duplicates and warn if found
    duplicates = df[df.duplicated(subset=["baseline"], keep=False)]
    if not duplicates.empty:
        logger.warning(
            "Found duplicate baseline entries: %s", duplicates["baseline"].unique()
        )

    return df


def calculate_category_performance(baseline_results: pd.DataFrame) -> pd.DataFrame:
    """
    Group baseline strategies by category and calculate category averages.

    Args:
        baseline_results: DataFrame from run_all_baselines()

    Returns:
        DataFrame with category-level performance metrics
    """
    category_stats = []

    # Get all unique categories from STRATEGY_METADATA
    categories = set()
    for metadata in STRATEGY_METADATA.values():
        categories.add(metadata["category"])

    for category in sorted(categories):
        # Filter baselines in this category
        category_baselines = baseline_results[
            baseline_results["baseline"].map(
                lambda x: STRATEGY_METADATA.get(x, {}).get("category") == category
            )
        ]

        if not category_baselines.empty:
            category_stats.append(
                {
                    "category": category,
                    "avg_return": category_baselines["total_return"].mean(),
                    "avg_sharpe": category_baselines["sharpe_annualized"].mean(),
                    "avg_win_rate": category_baselines["win_rate"].mean(),
                    "strategy_count": len(category_baselines),
                    "best_return": category_baselines["total_return"].max(),
                    "worst_return": category_baselines["total_return"].min(),
                    "return_std": category_baselines["total_return"].std(),
                    "sharpe_std": category_baselines["sharpe_annualized"].std(),
                }
            )

    return pd.DataFrame(category_stats)


def compare_llm_to_baselines(
    llm_metrics: Dict, baseline_results: pd.DataFrame
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
            (
                llm_metrics.get("sharpe_like", 0) * np.sqrt(TRADING_DAYS_PER_YEAR)
                if llm_metrics.get("sharpe_like")
                else 0
            ),
            3,
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


def print_categorized_baseline_comparison(
    comparison_df: pd.DataFrame, model_tag: str = "LLM"
):
    """
    Enhanced baseline comparison with strategy categorization (no LLM case).

    Groups strategies by category and shows performance within each group.
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED BASELINE COMPARISON - {model_tag}")
    print(f"{'='*80}")

    # Find LLM for cross-category comparison
    llm_return = comparison_df[comparison_df["baseline"] == "LLM_STRATEGY"][
        "total_return"
    ].values
    llm_return = llm_return[0] if len(llm_return) > 0 else None

    # Group by category using STRATEGY_METADATA
    categories = {}
    for _, row in comparison_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(row)
        elif baseline == "LLM_STRATEGY":
            # Special handling for LLM - find which category it would fit in
            # For now, put it in a separate "ai" category
            if "ai" not in categories:
                categories["ai"] = []
            categories["ai"].append(row)

    # Display each category
    category_order = [
        "ai",
        "passive",
        "trend_following",
        "mean_reversion",
        "momentum",
        "risk_management",
        "multi_factor",
        "confirmation",
        "noise",
    ]

    for category_name in category_order:
        if category_name in categories:
            strategies = categories[category_name]
            if not strategies:
                continue

            print(f"\n{category_name.upper().replace('_', ' ')} STRATEGIES:")
            print("-" * 70)

            # Sort by total return within category (best first)
            strategies_sorted = sorted(
                strategies, key=lambda x: x["total_return"], reverse=True
            )

            for i, row in enumerate(strategies_sorted):
                baseline = row["baseline"]
                ret = row["total_return"]
                sharpe = row["sharpe_annualized"]
                maxdd = row["max_drawdown"]
                winrate = row["win_rate"]

                # Highlight top performer in category
                rank_marker = " 🥇" if i == 0 and len(strategies) > 1 else ""

                # Highlight LLM
                llm_marker = " ◄" if baseline == "LLM_STRATEGY" else ""

                # LLM vs category indicator
                category_indicator = ""
                if (
                    llm_return is not None
                    and baseline != "LLM_STRATEGY"
                    and category_name != "ai"
                ):
                    if ret < llm_return:
                        category_indicator = " ✓"  # LLM beats this strategy
                    else:
                        category_indicator = " ✗"  # LLM loses to this strategy

                print(
                    f"{baseline:<28} {ret:>8.2f}% {sharpe:>8.3f} {maxdd:>7.2f}% "
                    f"{winrate:>6.1f}%{rank_marker}{category_indicator}{llm_marker}"
                )

    print(f"\n{'='*80}")


def analyze_category_performance(comparison_df: pd.DataFrame) -> Dict:
    """
    Analyze performance by strategy category.

    Returns dict with category-level statistics.
    """
    category_stats = {}

    for _, row in comparison_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]
            if category not in category_stats:
                category_stats[category] = {
                    "strategies": [],
                    "best_return": -999,
                    "worst_return": 999,
                    "avg_return": 0,
                    "avg_sharpe": 0,
                    "avg_win_rate": 0,
                    "count": 0,
                }

            stats = category_stats[category]
            stats["strategies"].append(row)
            stats["best_return"] = max(stats["best_return"], row["total_return"])
            stats["worst_return"] = min(stats["worst_return"], row["total_return"])
            stats["count"] += 1

    # Calculate averages
    for category, stats in category_stats.items():
        strategies = stats["strategies"]
        total_return = sum(row["total_return"] for row in strategies)
        total_sharpe = sum(row["sharpe_annualized"] for row in strategies)
        total_win_rate = sum(row["win_rate"] for row in strategies)

        stats["avg_return"] = total_return / stats["count"]
        stats["avg_sharpe"] = total_sharpe / stats["count"]
        stats["avg_win_rate"] = total_win_rate / stats["count"]

        # Remove strategies list to keep output clean
        del stats["strategies"]

    return category_stats


# =============================================================================
# ENHANCED STATISTICAL ROBUSTNESS FUNCTIONS
# =============================================================================


def robust_random_baseline(
    features_df: pd.DataFrame, n_runs: int = 30, confidence_level: float = 0.95
) -> Dict:
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
        results.append(metrics["total_return"])

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
        "mean": round(mean_return, 2),
        "std": round(std_return, 2),
        "median": round(median_return, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "min": round(min_return, 2),
        "max": round(max_return, 2),
        "q25": round(q25, 2),
        "q75": round(q75, 2),
        "n_runs": n_runs,
        "confidence_level": confidence_level,
        "all_results": results_array.tolist(),  # For further analysis
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
    all_random = np.array(random_stats["all_results"])

    # One-sample t-test: is LLM significantly different from random mean?
    t_stat, p_value = stats.ttest_1samp(all_random, llm_return)

    # Effect size (Cohen's d)
    effect_size = (llm_return - random_stats["mean"]) / random_stats["std"]

    # Percentage of random runs that LLM beats
    beats_random_pct = (llm_return > all_random).mean() * 100

    # Bayesian approach: probability LLM beats random
    # Using normal approximation
    from scipy.stats import norm

    prob_llm_beats_random = 1 - norm.cdf(
        llm_return, random_stats["mean"], random_stats["std"]
    )

    return {
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "effect_size": round(effect_size, 3),
        "beats_random_percent": round(beats_random_pct, 1),
        "prob_llm_beats_random": round(prob_llm_beats_random, 3),
        "significant": p_value < 0.05,
        "llm_vs_random_mean": round(llm_return - random_stats["mean"], 2),
    }


def enhanced_compare_llm_to_baselines(
    llm_metrics: Dict,
    baseline_results: pd.DataFrame,
    features_df: pd.DataFrame,
    n_random_runs: int = 30,
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
        "total_return": random_stats["mean"],
        "mean_daily_return": None,  # Not meaningful for aggregate
        "volatility": random_stats["std"],  # Use std as proxy for volatility
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
    baseline_results = baseline_results[baseline_results["baseline"] != "random"]
    enhanced_baselines = pd.concat([baseline_results, random_df], ignore_index=True)

    # Add LLM and sort
    all_results = compare_llm_to_baselines(llm_metrics, enhanced_baselines)

    return all_results, random_stats


def print_enhanced_baseline_comparison(
    comparison_df: pd.DataFrame,
    random_stats: Dict,
    llm_stats: Dict,
    model_tag: str = "LLM",
):
    """
    Print enhanced comparison with statistical details and categorization.
    """
    print("\n" + "=" * 100)
    print(f"ENHANCED BASELINE COMPARISON - {model_tag}")
    print("=" * 100)

    # Group by category using STRATEGY_METADATA
    categories = {}
    for _, row in comparison_df.iterrows():
        baseline = row["baseline"]
        if baseline in STRATEGY_METADATA:
            category = STRATEGY_METADATA[baseline]["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(row)
        elif baseline == "LLM_STRATEGY":
            # Special handling for LLM - put in "ai" category
            if "ai" not in categories:
                categories["ai"] = []
            categories["ai"].append(row)

    # Display each category
    category_order = [
        "ai",
        "passive",
        "trend_following",
        "mean_reversion",
        "momentum",
        "risk_management",
        "multi_factor",
        "confirmation",
        "noise",
    ]

    for category_name in category_order:
        if category_name in categories:
            strategies = categories[category_name]
            if not strategies:
                continue

            print(f"\n{category_name.upper().replace('_', ' ')} STRATEGIES:")
            print("-" * 85)

            # Sort by total return within category (best first)
            strategies_sorted = sorted(
                strategies,
                key=lambda x: (
                    x["total_return"] if pd.notna(x["total_return"]) else -999
                ),
                reverse=True,
            )

            for i, row in enumerate(strategies_sorted):
                baseline = row["baseline"]
        ret = row["total_return"] if pd.notna(row["total_return"]) else "N/A"
        sharpe = (
            row["sharpe_annualized"] if pd.notna(row["sharpe_annualized"]) else "N/A"
        )
        maxdd = row["max_drawdown"] if pd.notna(row["max_drawdown"]) else "N/A"
        winrate = row["win_rate"] if pd.notna(row["win_rate"]) else "N/A"

        # Highlight top performer in category
        rank_marker = (
            " 🥇" if i == 0 and len(strategies) > 1 and category_name != "ai" else ""
        )

        # Highlight LLM
        llm_marker = " ◄" if baseline == "LLM_STRATEGY" else ""

        print(
            f"{baseline:<30} {str(ret):>9} {str(sharpe):>10} {str(maxdd):>9} {str(winrate):>9}{rank_marker}{llm_marker}"
        )

    print("-" * 85)

    # Enhanced random statistics
    print(f"\nRANDOM BASELINE STATISTICS (n={random_stats['n_runs']} runs):")
    print(f"  Mean Return: {random_stats['mean']:.2f}%")
    print(
        f"  95% CI: [{random_stats['ci_lower']:.2f}%, {random_stats['ci_upper']:.2f}%]"
    )
    print(f"  Std Dev: {random_stats['std']:.2f}%")
    print(f"  Range: [{random_stats['min']:.2f}%, {random_stats['max']:.2f}%]")
    print(
        f"  Quartiles: Q25={random_stats['q25']:.2f}%, Median={random_stats['median']:.2f}%, Q75={random_stats['q75']:.2f}%"
    )

    # Statistical significance
    if llm_stats:
        print("\nSTATISTICAL SIGNIFICANCE vs RANDOM:")
        print(f"  LLM outperforms random by: {llm_stats['llm_vs_random_mean']:+.2f}%")
        print(f"  Beats random in: {llm_stats['beats_random_percent']:.1f}% of runs")
        print(f"  Effect size: {llm_stats['effect_size']:+.3f} (Cohen's d)")
        print(
            f"  p-value: {llm_stats['p_value']:.4f} ({'SIGNIFICANT' if llm_stats['significant'] else 'not significant'})"
        )
        print(f"  Probability LLM > Random: {llm_stats['prob_llm_beats_random']:.1%}")

    print("=" * 100 + "\n")
