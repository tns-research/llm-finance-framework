"""
Integration tests for dynamic symbol naming functionality.
"""

import pytest

from src.config import get_current_symbol_info
from src.configuration_manager import ConfigurationManager
from src.performance_tracker import PerformanceTracker
from src.prompt_builder import PromptBuilder


class TestDynamicSymbols:
    """Integration tests for dynamic symbol naming across components."""

    def test_config_symbol_info(self):
        """Test that config provides correct symbol information."""
        symbol_code, symbol_name = get_current_symbol_info()

        assert isinstance(symbol_code, str)
        assert isinstance(symbol_name, str)
        assert len(symbol_code) > 0
        assert len(symbol_name) > 0

        # Should map known symbols correctly
        if symbol_code == "SPY":
            assert "SPY ETF" in symbol_name
        elif symbol_code == "QQQ":
            assert "QQQ ETF" in symbol_name

    def test_configuration_manager_symbol_info(self):
        """Test that ConfigurationManager exposes symbol info."""
        cm = ConfigurationManager()
        symbol_code, symbol_name = cm.get_symbol_info()

        assert isinstance(symbol_code, str)
        assert isinstance(symbol_name, str)

    def test_prompt_builder_uses_dynamic_symbols(self):
        """Test that prompt builder includes correct symbol names."""
        cm = ConfigurationManager()
        pb = PromptBuilder(cm)

        prompt = pb.build_system_prompt()
        _, expected_symbol_name = cm.get_symbol_info()

        assert expected_symbol_name in prompt
        assert "S&P500" not in prompt  # Should not contain old hardcoded reference
        assert "S and P 500" not in prompt

    def test_performance_tracker_symbol_integration(self):
        """Test that performance tracker works with different symbols."""
        symbols_to_test = ["SPY ETF", "QQQ ETF", "S&P 500 Index", "Custom Index"]

        for symbol in symbols_to_test:
            tracker = PerformanceTracker(symbol)
            summary = tracker.get_performance_summary()

            assert f"{symbol} cumulative return so far  0.00 percent." in summary

    @pytest.mark.parametrize(
        "symbol,expected_contains",
        [
            ("SPY ETF", "SPY ETF"),
            ("QQQ ETF", "QQQ ETF"),
            ("Apple Stock", "Apple Stock"),
            ("S&P 500 Index", "S&P 500 Index"),
        ],
    )
    def test_symbol_parameterization(self, symbol, expected_contains):
        """Parameterized test for different symbol configurations."""
        tracker = PerformanceTracker(symbol)
        tracker.update_daily_performance("BUY", 1.0, 0.5)

        summary = tracker.get_performance_summary()
        assert expected_contains in summary

    def test_end_to_end_symbol_flow(self):
        """Test complete symbol flow from config to prompts."""
        # Get symbol from config
        symbol_code, symbol_name = get_current_symbol_info()

        # Create components with symbol
        cm = ConfigurationManager()
        pb = PromptBuilder(cm)
        tracker = PerformanceTracker(symbol_name)

        # Test that all components use the same symbol
        prompt = pb.build_system_prompt()
        summary = tracker.get_performance_summary()

        assert symbol_name in prompt
        assert symbol_name in summary

        # Test with trading data
        tracker.update_daily_performance("BUY", 2.0, 1.0)
        summary_with_data = tracker.get_performance_summary()
        assert symbol_name in summary_with_data
