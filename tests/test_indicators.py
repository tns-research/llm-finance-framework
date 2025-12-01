#!/usr/bin/env python3
"""Unit tests for advanced technical indicators."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_prep import compute_macd, compute_stochastic, compute_bollinger_bands


class TestTechnicalIndicators:
    """Test suite for advanced technical indicators."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Generate realistic price data with trend
        base_price = 100
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.randn(100) * 2
        close = base_price + trend + noise

        # Create OHLC from close with some spread
        high = close + np.abs(np.random.randn(100) * 1.5)
        low = close - np.abs(np.random.randn(100) * 1.5)
        open_price = close + np.random.randn(100) * 0.5

        return pd.DataFrame({
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })

    def test_macd_calculation(self, sample_data):
        """Test MACD computation produces expected outputs."""
        prices = sample_data['close']
        macd_line, signal_line, histogram = compute_macd(prices)

        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

        # Check that we have some valid values (not all NaN)
        assert not macd_line.isna().all()
        assert not signal_line.isna().all()
        assert not histogram.isna().all()

        # MACD should be fast_ema - slow_ema
        fast_ema = prices.ewm(span=12, adjust=False).mean()
        slow_ema = prices.ewm(span=26, adjust=False).mean()
        expected_macd = fast_ema - slow_ema
        pd.testing.assert_series_equal(macd_line, expected_macd, check_names=False)

        # Histogram should be macd_line - signal_line
        expected_hist = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, expected_hist, check_names=False)

    def test_stochastic_calculation(self, sample_data):
        """Test Stochastic Oscillator computation."""
        high = sample_data['high']
        low = sample_data['low']
        close = sample_data['close']

        k_percent, d_percent = compute_stochastic(high, low, close)

        assert len(k_percent) == len(close)
        assert len(d_percent) == len(close)

        # %K and %D should be between 0 and 100 (with some tolerance for floating point)
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()

        if len(valid_k) > 0:
            assert (valid_k >= -1).all() and (valid_k <= 101).all()  # Small tolerance for floating point

        if len(valid_d) > 0:
            assert (valid_d >= -1).all() and (valid_d <= 101).all()

    def test_bollinger_bands_calculation(self, sample_data):
        """Test Bollinger Bands computation."""
        prices = sample_data['close']
        upper, middle, lower = compute_bollinger_bands(prices)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # Check that we have some valid values
        assert not upper.isna().all()
        assert not middle.isna().all()
        assert not lower.isna().all()

        # Upper should be >= middle >= lower for valid values
        valid_idx = ~upper.isna() & ~middle.isna() & ~lower.isna()
        if valid_idx.any():
            assert (upper[valid_idx] >= middle[valid_idx]).all()
            assert (middle[valid_idx] >= lower[valid_idx]).all()

        # Middle band should be simple moving average
        expected_middle = prices.rolling(window=20).mean()
        pd.testing.assert_series_equal(middle, expected_middle, check_names=False)

    def test_macd_with_custom_parameters(self):
        """Test MACD with custom parameters."""
        prices = pd.Series([100, 101, 102, 103, 104, 105] * 10)  # 60 periods

        # Test with different parameters
        macd_line, signal_line, histogram = compute_macd(prices, fast=8, slow=21, signal=5)

        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_stochastic_with_custom_parameters(self, sample_data):
        """Test Stochastic with custom parameters."""
        high = sample_data['high']
        low = sample_data['low']
        close = sample_data['close']

        # Test with different periods
        k_percent, d_percent = compute_stochastic(high, low, close, k_period=10, d_period=5, smooth_k=2)

        assert len(k_percent) == len(close)
        assert len(d_percent) == len(close)

    def test_bollinger_with_custom_parameters(self, sample_data):
        """Test Bollinger Bands with custom parameters."""
        prices = sample_data['close']

        # Test with different parameters
        upper, middle, lower = compute_bollinger_bands(prices, window=10, std_dev=1.5)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # Verify middle is 10-period SMA
        expected_middle = prices.rolling(window=10).mean()
        pd.testing.assert_series_equal(middle, expected_middle, check_names=False)

    def test_edge_cases_constant_prices(self):
        """Test indicators with constant prices."""
        constant_prices = pd.Series([100] * 50)

        # MACD should be 0 for constant prices
        macd_line, signal_line, histogram = compute_macd(constant_prices)
        assert len(macd_line) == len(constant_prices)
        # For constant prices, MACD should eventually be 0
        assert macd_line.iloc[-10:].abs().max() < 1e-10  # Very close to 0

        # Stochastic should be 50 for constant prices (middle of range)
        high_const = pd.Series([100] * 50)
        low_const = pd.Series([100] * 50)
        k_percent, d_percent = compute_stochastic(high_const, low_const, constant_prices)
        assert len(k_percent) == len(constant_prices)
        # Should be NaN initially, then some value (could be 50 or NaN depending on calculation)

    def test_edge_cases_insufficient_data(self):
        """Test indicators handle insufficient data gracefully."""
        short_prices = pd.Series([100, 101, 102])

        # MACD with insufficient data
        macd_line, signal_line, histogram = compute_macd(short_prices)
        assert len(macd_line) == len(short_prices)
        # Should handle gracefully (may produce NaN)

        # Bollinger Bands with insufficient data
        upper, middle, lower = compute_bollinger_bands(short_prices, window=20)
        assert len(upper) == len(short_prices)
        # Should be mostly NaN for short series

    def test_stochastic_extreme_prices(self):
        """Test Stochastic with extreme price movements."""
        # Create data where price hits highs and lows
        high = pd.Series([110, 120, 130, 140, 150] * 10)
        low = pd.Series([90, 80, 70, 60, 50] * 10)
        close = pd.Series([100, 110, 120, 130, 140] * 10)

        k_percent, d_percent = compute_stochastic(high, low, close)

        assert len(k_percent) == len(close)
        assert len(d_percent) == len(close)

        # %K should be calculable and reasonable
        valid_k = k_percent.dropna()
        if len(valid_k) > 0:
            assert (valid_k >= 0).all() and (valid_k <= 100).all()

    def test_indicators_handle_nan_inputs(self):
        """Test that indicators handle NaN inputs gracefully."""
        prices_with_nan = pd.Series([100, np.nan, 102, 103, np.nan, 105] * 10)

        # MACD should handle NaN
        macd_line, signal_line, histogram = compute_macd(prices_with_nan)
        assert len(macd_line) == len(prices_with_nan)

        # Bollinger Bands should handle NaN
        upper, middle, lower = compute_bollinger_bands(prices_with_nan)
        assert len(upper) == len(prices_with_nan)


if __name__ == "__main__":
    pytest.main([__file__])
