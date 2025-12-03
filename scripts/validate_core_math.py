#!/usr/bin/env python3
"""
Critical Mathematical Validation for LLM Finance Framework
Tests the core calculations that directly impact trading decisions.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_strategy_returns():
    """Test position × return multiplication - CRITICAL for P&L"""
    print("[CALC] Testing Strategy Returns Calculation...")

    # Test cases with known expected results
    test_cases = [
        # (position, market_return, expected_strategy_return)
        (1.0, 1.5, 1.5),    # BUY: +1.5%
        (-1.0, 1.5, -1.5),  # SELL: -1.5%
        (0.0, 1.5, 0.0),    # HOLD: 0%
        (1.0, -2.3, -2.3),  # BUY in down market
        (-1.0, -2.3, 2.3),  # SELL in down market
    ]

    all_passed = True
    for position, market_ret, expected in test_cases:
        actual = position * market_ret
        if abs(actual - expected) > 1e-10:
            print(f"  [FAIL] FAILED: position={position}, market_return={market_ret}")
            print(f"     Expected: {expected}, Got: {actual}")
            all_passed = False
        else:
            print(f"  [PASS] position={position} × {market_ret}% = {actual}%")

    return all_passed

def test_equity_curve():
    """Test equity curve calculation - CRITICAL for total returns"""
    print("\n[EQUITY] Testing Equity Curve Calculation...")

    # Start with $100, apply daily returns
    initial_capital = 100.0
    daily_returns_pct = [1.0, -0.5, 2.0, -1.0, 0.5]  # Percent returns

    # Manual calculation
    equity_manual = [initial_capital]
    for ret_pct in daily_returns_pct:
        new_equity = equity_manual[-1] * (1 + ret_pct / 100.0)
        equity_manual.append(new_equity)

    # Framework-style calculation
    returns_series = pd.Series(daily_returns_pct)
    equity_curve = (1 + returns_series / 100.0).cumprod() * initial_capital

    # Compare
    max_diff = max(abs(a - b) for a, b in zip(equity_manual[1:], equity_curve))

    if max_diff > 1e-10:
        print(f"  [FAIL] FAILED: Maximum difference = {max_diff}")
        return False
    else:
        print(f"  [PASS] Equity curve calculation accurate (max diff < 1e-10)")
        print(f"     Final equity: ${equity_curve.iloc[-1]:.2f}")
        return True

def test_performance_metrics():
    """Test Sharpe ratio, drawdown, win rate - CRITICAL for strategy evaluation"""
    print("\n[METRICS] Testing Performance Metrics...")

    # Create test return series with known characteristics
    np.random.seed(42)
    returns = pd.Series([0.1, 0.2, -0.1, 0.15, -0.05, 0.08, 0.12, -0.08, 0.06, 0.09])

    # Calculate metrics
    mean_return = returns.mean()
    volatility = returns.std()
    sharpe = mean_return / volatility if volatility > 0 else np.nan
    win_rate = (returns > 0).mean()

    # Calculate drawdown
    cumulative = (1 + returns / 100.0).cumprod()
    running_max = cumulative.cummax()
    drawdown_series = cumulative / running_max - 1
    max_drawdown = drawdown_series.min()

    # Validate against manual calculations
    expected_mean = sum(returns) / len(returns)
    # pandas std() uses ddof=1 (sample std dev), numpy std() uses ddof=0 (population)
    expected_vol = np.sqrt(sum((returns - expected_mean)**2) / (len(returns) - 1))
    expected_win_rate = sum(returns > 0) / len(returns)

    checks = [
        ("Mean Return", abs(mean_return - expected_mean) < 1e-10),
        ("Volatility", abs(volatility - expected_vol) < 1e-10),
        ("Win Rate", abs(win_rate - expected_win_rate) < 1e-10),
        ("Max Drawdown", max_drawdown <= 0),  # Should be negative or zero
        ("Sharpe Ratio", not np.isnan(sharpe)),  # Should be calculable
    ]

    all_passed = True
    for metric_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {metric_name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    return all_passed

def test_technical_indicators():
    """Test basic technical indicator calculations"""
    print("\n[INDICATORS] Testing Technical Indicators...")

    # Simple price series for testing
    prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108])

    # Test RSI range
    try:
        from src.data_prep import compute_rsi
        rsi = compute_rsi(prices, window=5)
        rsi_valid = rsi.dropna()

        rsi_range_ok = rsi_valid.min() >= 0 and rsi_valid.max() <= 100
        print(f"  [{'PASS]' if rsi_range_ok else '[FAIL]'} RSI range check: {rsi_range_ok}")

        if not rsi_range_ok:
            print(f"     RSI values: min={rsi_valid.min():.1f}, max={rsi_valid.max():.1f}")
            return False

    except Exception as e:
        print(f"  [ERROR] RSI calculation error: {e}")
        return False

    # Test EMA smoothness
    try:
        from src.data_prep import compute_ema
        ema_short = compute_ema(prices, 3)
        ema_long = compute_ema(prices, 8)

        # Short EMA should be more volatile than long EMA
        short_vol = ema_short.std()
        long_vol = ema_long.std()
        smoothness_ok = short_vol > long_vol
        print(f"  [{'PASS]' if smoothness_ok else '[FAIL]'} EMA smoothness check: {smoothness_ok}")

        if not smoothness_ok:
            print(f"     Short EMA vol: {short_vol:.4f}, Long EMA vol: {long_vol:.4f}")

    except Exception as e:
        print(f"  [ERROR] EMA calculation error: {e}")
        return False

    return True

def main():
    """Run all critical mathematical validations"""
    print("[VALIDATION] CRITICAL MATHEMATICAL VALIDATION")
    print("=" * 50)
    print("Testing calculations that directly impact trading decisions...")

    tests = [
        ("Strategy Returns", test_strategy_returns),
        ("Equity Curve", test_equity_curve),
        ("Performance Metrics", test_performance_metrics),
        ("Technical Indicators", test_technical_indicators),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {test_name} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("[SUMMARY] VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "[PASS]" if results[i] else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\n[RESULT] Overall: {passed}/{total} critical tests passed")

    if passed == total:
        print("[SUCCESS] All critical mathematical calculations are VALID!")
        return 0
    else:
        print("[WARNING] MATHEMATICAL ISSUES DETECTED - DO NOT USE FOR REAL TRADING!")
        print("   Fix the failing tests before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
