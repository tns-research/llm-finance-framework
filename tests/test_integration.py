#!/usr/bin/env python3
"""Integration tests for the full LLM finance framework pipeline with new indicators."""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_prep import prepare_features
from src.prompts import build_prompts
from src.config import (
    ENABLE_MACD, ENABLE_STOCHASTIC, ENABLE_BOLLINGER_BANDS,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K, STOCH_D,
    BB_WINDOW, BB_STD
)


class TestFullPipelineIntegration:
    """Test that the full data processing pipeline works with new indicators."""

    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        n_periods = 200  # Enough for all indicators to calculate

        # Generate realistic price data
        dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')

        # Create a trending price series
        trend = np.linspace(0, 50, n_periods)
        noise = np.random.randn(n_periods) * 2
        close = 100 + trend + noise

        # Create OHLC from close with realistic spreads
        spread = np.abs(np.random.randn(n_periods) * 1.5)
        high = close + spread
        low = close - spread
        open_price = close + np.random.randn(n_periods) * 0.5

        # Ensure high >= close >= low and high >= open >= low
        high = np.maximum(high, np.maximum(close, open_price))
        low = np.minimum(low, np.minimum(close, open_price))

        return pd.DataFrame({
            'Date': dates,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': np.random.randint(100000, 1000000, n_periods)
        })

    def test_full_pipeline_with_new_indicators(self, sample_ohlc_data, tmp_path):
        """Test that the full data processing pipeline works with new indicators."""
        # Save sample data to CSV
        csv_path = tmp_path / "sample_data.csv"
        sample_ohlc_data.to_csv(csv_path, index=False)

        # Process through full pipeline
        features_path = tmp_path / "features.csv"
        prompts_path = tmp_path / "prompts.csv"

        features_df = prepare_features(str(csv_path), str(features_path))
        prompts_df = build_prompts(str(features_path), str(prompts_path))

        # Validate that features DataFrame has expected columns
        expected_base_features = ["date", "close", "return_1d", "ma20_pct", "vol20_annualized", "rsi_14", "ret_5d"]

        for feature in expected_base_features:
            assert feature in features_df.columns, f"Missing base feature: {feature}"

        # Validate new indicator features are present when enabled
        if ENABLE_MACD:
            macd_features = ["macd_line", "macd_signal", "macd_histogram"]
            for feature in macd_features:
                assert feature in features_df.columns, f"Missing MACD feature: {feature}"
                assert not features_df[feature].isna().all(), f"MACD feature {feature} is all NaN"

        if ENABLE_STOCHASTIC:
            stoch_features = ["stoch_k", "stoch_d"]
            for feature in stoch_features:
                assert feature in features_df.columns, f"Missing Stochastic feature: {feature}"
                assert not features_df[feature].isna().all(), f"Stochastic feature {feature} is all NaN"

        if ENABLE_BOLLINGER_BANDS:
            bb_features = ["bb_upper", "bb_middle", "bb_lower", "bb_position"]
            for feature in bb_features:
                assert feature in features_df.columns, f"Missing Bollinger Bands feature: {feature}"
                assert not features_df[feature].isna().all(), f"Bollinger Bands feature {feature} is all NaN"

        # Validate prompts contain new indicators
        assert len(prompts_df) > 0, "No prompts generated"

        # Check a sample prompt from later in the series (after NaN period)
        sample_prompt = prompts_df['prompt_text'].iloc[50]  # Skip initial NaN period

        assert 'RSI(14)' in sample_prompt, "RSI not found in prompt"

        if ENABLE_MACD:
            assert 'MACD(' in sample_prompt, "MACD not found in prompt"
            assert str(MACD_FAST) in sample_prompt, f"MACD fast period {MACD_FAST} not in prompt"
            assert str(MACD_SLOW) in sample_prompt, f"MACD slow period {MACD_SLOW} not in prompt"
            assert str(MACD_SIGNAL) in sample_prompt, f"MACD signal period {MACD_SIGNAL} not in prompt"

        if ENABLE_STOCHASTIC:
            assert 'Stochastic(' in sample_prompt, "Stochastic not found in prompt"
            assert str(STOCH_K) in sample_prompt, f"Stochastic K period {STOCH_K} not in prompt"
            assert str(STOCH_D) in sample_prompt, f"Stochastic D period {STOCH_D} not in prompt"

        if ENABLE_BOLLINGER_BANDS:
            assert 'Bollinger Bands(' in sample_prompt, "Bollinger Bands not found in prompt"
            assert str(BB_WINDOW) in sample_prompt, f"Bollinger Bands window {BB_WINDOW} not in prompt"
            assert str(BB_STD) in sample_prompt, f"Bollinger Bands std {BB_STD} not in prompt"

        # Validate summary contains new indicators
        assert "Summary" in sample_prompt, "Summary section not found in prompt"

        summary_section = sample_prompt.split("Summary")[1]
        assert "RSI(14)" in summary_section, "RSI not in summary"

        # Validate data integrity
        assert not features_df.empty, "Features DataFrame is empty"
        assert not prompts_df.empty, "Prompts DataFrame is empty"

        # Check that dates are preserved correctly
        assert len(features_df) == len(prompts_df), "Features and prompts have different lengths"
        assert all(features_df['date'] == prompts_df['date']), "Dates don't match between features and prompts"

    def test_pipeline_with_minimal_ohlc_data(self, tmp_path):
        """Test pipeline works with minimal OHLC data (just close prices)."""
        # Create minimal CSV with just Date and Close
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = 100 + np.random.randn(100) * 2

        minimal_data = pd.DataFrame({
            'Date': dates,
            'Close': close_prices
        })

        csv_path = tmp_path / "minimal_data.csv"
        minimal_data.to_csv(csv_path, index=False)

        # Process through pipeline - should work with just close prices
        features_path = tmp_path / "features_minimal.csv"
        prompts_path = tmp_path / "prompts_minimal.csv"

        features_df = prepare_features(str(csv_path), str(features_path))
        prompts_df = build_prompts(str(features_path), str(prompts_path))

        # Should still have basic features
        assert "close" in features_df.columns
        assert "rsi_14" in features_df.columns
        assert "return_1d" in features_df.columns

        # Advanced indicators might be NaN but columns should exist
        if ENABLE_MACD:
            assert "macd_line" in features_df.columns
        if ENABLE_STOCHASTIC:
            assert "stoch_k" in features_df.columns
        if ENABLE_BOLLINGER_BANDS:
            assert "bb_upper" in features_df.columns

    def test_feature_flags_disable_indicators(self, sample_ohlc_data, tmp_path, monkeypatch):
        """Test that disabling feature flags removes indicators from pipeline."""
        # Temporarily disable all advanced indicators
        monkeypatch.setattr('src.data_prep.ENABLE_MACD', False)
        monkeypatch.setattr('src.data_prep.ENABLE_STOCHASTIC', False)
        monkeypatch.setattr('src.data_prep.ENABLE_BOLLINGER_BANDS', False)
        monkeypatch.setattr('src.prompts.ENABLE_MACD', False)
        monkeypatch.setattr('src.prompts.ENABLE_STOCHASTIC', False)
        monkeypatch.setattr('src.prompts.ENABLE_BOLLINGER_BANDS', False)

        # Save sample data
        csv_path = tmp_path / "sample_data.csv"
        sample_ohlc_data.to_csv(csv_path, index=False)

        # Process through pipeline
        features_path = tmp_path / "features_no_advanced.csv"
        prompts_path = tmp_path / "prompts_no_advanced.csv"

        features_df = prepare_features(str(csv_path), str(features_path))
        prompts_df = build_prompts(str(features_path), str(prompts_path))

        # Should not have advanced indicator columns
        advanced_features = ["macd_line", "stoch_k", "bb_upper"]
        for feature in advanced_features:
            assert feature not in features_df.columns, f"Advanced feature {feature} should not be present when disabled"

        # Prompts should not contain advanced indicators
        sample_prompt = prompts_df['prompt_text'].iloc[30]
        assert 'MACD(' not in sample_prompt, "MACD should not be in prompt when disabled"
        assert 'Stochastic(' not in sample_prompt, "Stochastic should not be in prompt when disabled"
        assert 'Bollinger Bands(' not in sample_prompt, "Bollinger Bands should not be in prompt when disabled"

    def test_prompt_formatting_with_new_indicators(self, sample_ohlc_data, tmp_path):
        """Test that prompts are properly formatted with new indicators."""
        # Save sample data
        csv_path = tmp_path / "sample_data.csv"
        sample_ohlc_data.to_csv(csv_path, index=False)

        # Process through pipeline
        features_path = tmp_path / "features_formatting.csv"
        prompts_path = tmp_path / "prompts_formatting.csv"

        features_df = prepare_features(str(csv_path), str(features_path))
        prompts_df = build_prompts(str(features_path), str(prompts_path))

        # Get a prompt from the middle (should have valid indicator values)
        middle_idx = len(prompts_df) // 2
        prompt_text = prompts_df['prompt_text'].iloc[middle_idx]

        # Validate prompt structure
        lines = prompt_text.split('\n')

        # Should have basic structure
        assert any('day total return' in line for line in lines), "Return information not found"
        assert any('RSI(14)' in line for line in lines), "RSI information not found"
        assert any('Summary' in line for line in lines), "Summary section not found"

        # Should have advanced indicators if enabled
        if ENABLE_MACD:
            assert any('MACD(' in line for line in lines), "MACD information not found"
        if ENABLE_STOCHASTIC:
            assert any('Stochastic(' in line for line in lines), "Stochastic information not found"
        if ENABLE_BOLLINGER_BANDS:
            assert any('Bollinger Bands(' in line for line in lines), "Bollinger Bands information not found"

        # Summary should be coherent
        summary_start = prompt_text.find('Summary')
        assert summary_start != -1, "Summary section not found"

        summary_text = prompt_text[summary_start:]
        # Summary should be readable and not contain raw NaN values
        assert 'NaN' not in summary_text, "Summary contains NaN values"


if __name__ == "__main__":
    pytest.main([__file__])
