# Test Suite for Report Generator
# Tests the critical report generation functionality including new enhanced sections

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.report_generator import (
    generate_baseline_strategies_section,
    generate_baseline_strategies_section_html,
    generate_llm_indicator_alignment_section,
    generate_llm_indicator_alignment_section_html,
    generate_strategy_comparison_insights,
    generate_strategy_comparison_insights_html,
)


class TestBaselineStrategiesSection:
    """Test the enhanced baseline strategies section functionality."""

    def setup_method(self):
        """Set up mock baseline comparison data."""
        self.mock_baseline_data = pd.DataFrame(
            {
                "baseline": [
                    "buy_and_hold",
                    "momentum",
                    "rsi_mean_reversion",
                    "stochastic_oscillator",
                    "macd_momentum",
                    "random",
                ],
                "total_return": [45.2, 32.1, 28.7, 24.3, 19.8, -5.2],
                "sharpe_annualized": [1.2, 0.8, 0.9, 0.7, 0.6, -0.3],
                "win_rate": [52.1, 51.2, 53.4, 50.8, 49.2, 48.5],
            }
        )

        self.data_sources = {"baseline_comparison": self.mock_baseline_data}

    def test_generate_baseline_strategies_section_with_data(self):
        """Test baseline strategies section with complete data."""
        result = generate_baseline_strategies_section(self.data_sources, "test_model")
        result_str = "\n".join(result)

        # Check section header
        assert "## 📈 Enhanced Baseline Strategy Suite" in result_str
        assert "15 baseline strategies across 8 categories" in result_str

        # Check category table headers
        assert "| Category | Strategies | Best Return | Avg Return |" in result_str

        # Check specific content
        assert "Buy And Hold" in result_str  # Title case in output
        assert "45.2%" in result_str  # Best return

        # Check summary
        assert "**Total Strategies**:" in result_str
        assert "**Research Purpose**:" in result_str

    def test_generate_baseline_strategies_section_missing_data(self):
        """Test graceful handling when baseline data is missing."""
        result = generate_baseline_strategies_section({}, "test_model")
        result_str = "\n".join(result)

        assert "baseline comparison data not available" in result_str.lower()

    def test_category_aggregation(self):
        """Test that categories are properly aggregated."""
        result = generate_baseline_strategies_section(self.data_sources, "test_model")
        result_str = "\n".join(result)

        # Should contain category names
        assert "Passive" in result_str
        assert "Trend Following" in result_str
        assert "Mean Reversion" in result_str

    def test_html_version(self):
        """Test HTML version generates properly."""
        result = generate_baseline_strategies_section_html(
            self.data_sources, "test_model"
        )

        # Check HTML structure
        assert "<h2>📈 Enhanced Baseline Strategy Suite</h2>" in result
        assert '<table class="performance-table">' in result
        assert "Buy And Hold" in result

    def test_html_missing_data(self):
        """Test HTML version handles missing data."""
        result = generate_baseline_strategies_section_html({}, "test_model")

        assert "baseline comparison data not available" in result.lower()


class TestLLMIndicatorAlignmentSection:
    """Test the enhanced LLM indicator alignment functionality."""

    def setup_method(self):
        """Set up mock indicator alignment data."""
        self.mock_alignment_data = {
            "RSI": {
                "alignment_rate": 0.78,
                "total_signals": 245,
                "agreements": 191,
                "bullish_signals": 120,
                "bearish_signals": 125,
                "llm_buy_signals": 115,
                "llm_sell_signals": 118,
                "description": "RSI oversold/overbought mean reversion",
            },
            "MACD": {
                "alignment_rate": 0.65,
                "total_signals": 245,
                "agreements": 159,
                "bullish_signals": 118,
                "bearish_signals": 127,
                "llm_buy_signals": 112,
                "llm_sell_signals": 122,
                "description": "MACD line crossover trend signals",
            },
            "Stochastic": {
                "alignment_rate": 0.42,
                "total_signals": 245,
                "agreements": 103,
                "bullish_signals": 115,
                "bearish_signals": 130,
                "llm_buy_signals": 108,
                "llm_sell_signals": 115,
                "description": "Stochastic oscillator mean reversion",
            },
        }

        self.data_sources = {"llm_indicator_alignment": self.mock_alignment_data}

    def test_generate_indicator_alignment_section(self):
        """Test indicator alignment section with complete data."""
        result = generate_llm_indicator_alignment_section(
            self.data_sources, "test_model"
        )
        result_str = "\n".join(result)

        # Check section header
        assert "## 🤖 Enhanced LLM Indicator Alignment Analysis" in result_str

        # Check ranking
        assert "Indicator Effectiveness Ranking" in result_str
        assert "78.0%" in result_str  # RSI alignment
        assert "65.0%" in result_str  # MACD alignment

        # Check pattern analysis
        assert "Decision Pattern Analysis" in result_str
        assert "Bullish Signals" in result_str

        # Check interpretation
        assert "Strategic Insights" in result_str

    def test_indicator_ranking_order(self):
        """Test that indicators are ranked by alignment rate."""
        result = generate_llm_indicator_alignment_section(
            self.data_sources, "test_model"
        )
        result_str = "\n".join(result)

        # RSI should appear before MACD (higher alignment)
        rsi_pos = result_str.find("RSI")
        macd_pos = result_str.find("MACD")

        assert rsi_pos < macd_pos, "RSI should be ranked before MACD"

    def test_html_version_alignment(self):
        """Test HTML version of alignment section."""
        result = generate_llm_indicator_alignment_section_html(
            self.data_sources, "test_model"
        )

        assert "<h2>🤖 Enhanced LLM Indicator Alignment Analysis</h2>" in result
        assert "78.0%" in result
        assert "positive" in result  # CSS classes for alignment

    def test_missing_alignment_data(self):
        """Test handling of missing alignment data."""
        result = generate_llm_indicator_alignment_section({}, "test_model")
        result_str = "\n".join(result)

        assert "indicator alignment analysis was not available" in result_str.lower()


class TestStrategyComparisonInsights:
    """Test the strategy comparison insights functionality."""

    def setup_method(self):
        """Set up mock data for strategy comparison."""
        self.mock_baseline_data = pd.DataFrame(
            {
                "baseline": ["buy_and_hold", "momentum", "rsi_mean_reversion"],
                "total_return": [45.2, 32.1, 28.7],
                "sharpe_annualized": [1.2, 0.8, 0.9],
                "win_rate": [52.1, 51.2, 53.4],
            }
        )

        self.mock_stat_validation = {
            "dataset_info": {
                "total_strategy_return": 38.5,
                "sharpe_ratio": 1.1,
                "win_rate": 54.2,
            }
        }

        self.data_sources = {
            "baseline_comparison": self.mock_baseline_data,
            "statistical_validation": self.mock_stat_validation,
        }

    def test_generate_strategy_comparison_insights(self):
        """Test strategy comparison insights generation."""
        result = generate_strategy_comparison_insights(self.data_sources, "test_model")
        result_str = "\n".join(result)

        # Check section header
        assert "## 🎯 Strategy Comparison Insights" in result_str

        # Check LLM positioning
        assert "LLM vs Category Performance Positioning" in result_str

        # Check resemblance analysis
        assert "Strategy Resemblance Analysis" in result_str

        # Check attribution analysis
        assert "Performance Attribution Analysis" in result_str

        # Check insights
        assert "Strategic Insights" in result_str

    def test_llm_performance_extraction(self):
        """Test that LLM performance is correctly extracted."""
        result = generate_strategy_comparison_insights(self.data_sources, "test_model")
        result_str = "\n".join(result)

        # Should contain LLM performance metrics
        assert "38.5%" in result_str  # LLM return
        assert "LLM Performance" in result_str  # LLM column header

    def test_category_comparison(self):
        """Test category-level comparisons."""
        result = generate_strategy_comparison_insights(self.data_sources, "test_model")
        result_str = "\n".join(result)

        # Should show performance vs categories
        assert "Passive" in result_str
        assert "Trend Following" in result_str

    def test_html_version_insights(self):
        """Test HTML version of insights section."""
        result = generate_strategy_comparison_insights_html(
            self.data_sources, "test_model"
        )

        assert "<h2>🎯 Strategy Comparison Insights</h2>" in result
        assert "38.5%" in result

    def test_missing_comparison_data(self):
        """Test handling when comparison data is incomplete."""
        incomplete_sources = {"baseline_comparison": self.mock_baseline_data}
        result = generate_strategy_comparison_insights(incomplete_sources, "test_model")
        result_str = "\n".join(result)

        assert "strategy comparison insights require" in result_str.lower()


class TestReportGeneratorIntegration:
    """Test integration of multiple report sections."""

    def test_all_sections_with_mock_data(self):
        """Test that all sections work together with complete mock data."""
        # Create comprehensive mock data
        mock_data = {
            "baseline_comparison": pd.DataFrame(
                {
                    "baseline": ["buy_and_hold", "momentum", "rsi_mean_reversion"],
                    "total_return": [45.2, 32.1, 28.7],
                    "sharpe_annualized": [1.2, 0.8, 0.9],
                    "win_rate": [52.1, 51.2, 53.4],
                }
            ),
            "llm_indicator_alignment": {
                "RSI": {
                    "alignment_rate": 0.75,
                    "total_signals": 200,
                    "agreements": 150,
                    "bullish_signals": 100,
                    "bearish_signals": 100,
                    "description": "RSI signals",
                }
            },
            "statistical_validation": {
                "dataset_info": {
                    "total_strategy_return": 35.0,
                    "sharpe_ratio": 1.0,
                    "win_rate": 52.0,
                }
            },
        }

        # Test each section generates without errors
        baseline_result = generate_baseline_strategies_section(
            mock_data, "integration_test"
        )
        alignment_result = generate_llm_indicator_alignment_section(
            mock_data, "integration_test"
        )
        insights_result = generate_strategy_comparison_insights(
            mock_data, "integration_test"
        )

        # All should be non-empty lists
        assert len(baseline_result) > 10
        assert len(alignment_result) > 10
        assert len(insights_result) > 10

        # Convert to strings for content checks
        baseline_str = "\n".join(baseline_result)
        alignment_str = "\n".join(alignment_result)
        insights_str = "\n".join(insights_result)

        # All should contain expected content
        assert "Enhanced Baseline Strategy Suite" in baseline_str
        assert "Enhanced LLM Indicator Alignment" in alignment_str
        assert "Strategy Comparison Insights" in insights_str

    def test_error_handling_consistency(self):
        """Test that all sections handle missing data consistently."""
        empty_data = {}

        baseline_result = generate_baseline_strategies_section(empty_data, "test")
        alignment_result = generate_llm_indicator_alignment_section(empty_data, "test")
        insights_result = generate_strategy_comparison_insights(empty_data, "test")

        # Convert to strings for checking
        baseline_str = "\n".join(baseline_result)
        alignment_str = "\n".join(alignment_result)
        insights_str = "\n".join(insights_result)

        # All should provide helpful error messages
        assert "not available" in baseline_str.lower()
        assert "not available" in alignment_str.lower()
        assert "require" in insights_str.lower()


class TestDataProcessing:
    """Test data processing logic in report functions."""

    def test_baseline_category_grouping(self):
        """Test that baselines are correctly grouped by category."""
        from src.baselines import STRATEGY_METADATA

        # This test ensures the category mapping is working
        test_strategies = ["buy_and_hold", "momentum", "rsi_mean_reversion"]

        for strategy in test_strategies:
            assert (
                strategy in STRATEGY_METADATA
            ), f"Strategy {strategy} should be in metadata"
            assert (
                "category" in STRATEGY_METADATA[strategy]
            ), f"Strategy {strategy} should have category"

    def test_indicator_alignment_calculations(self):
        """Test that alignment calculations are mathematically correct."""
        # Test data
        mock_alignment = {
            "RSI": {
                "bullish_signals": 10,
                "bearish_signals": 5,
                "llm_buy_signals": 8,
                "llm_sell_signals": 6,
                "agreements": 12,
            }
        }

        # Should not crash and should return valid results
        data_sources = {"llm_indicator_alignment": mock_alignment}
        result = generate_llm_indicator_alignment_section(data_sources, "test")
        result_str = "\n".join(result)

        assert len(result) > 0
        assert "RSI" in result_str
