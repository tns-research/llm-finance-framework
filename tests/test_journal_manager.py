"""
Unit tests for JournalManager class
"""

from datetime import datetime

import pandas as pd
import pytest

from src.journal_manager import JournalManager


class TestJournalManager:
    """Test cases for JournalManager class"""

    def test_initialization_default(self):
        """Test JournalManager initializes with default values"""
        manager = JournalManager()

        assert manager.entries == []
        assert manager.max_entries == 10
        assert manager.is_empty() == True
        assert manager.get_entry_count() == 0

    def test_initialization_custom_max(self):
        """Test JournalManager initializes with custom max_entries"""
        manager = JournalManager(max_entries=5)

        assert manager.max_entries == 5
        assert manager.is_empty() == True

    def test_add_single_entry(self):
        """Test adding a single trade entry"""
        manager = JournalManager()
        trade_data = {"decision": "BUY", "prob": 0.8}

        manager.add_trade_entry(trade_data)

        assert manager.get_entry_count() == 1
        assert manager.is_empty() == False
        assert manager.entries[0] == trade_data

    def test_add_multiple_entries(self):
        """Test adding multiple trade entries"""
        manager = JournalManager()

        trades = [
            {"decision": "BUY", "prob": 0.8},
            {"decision": "HOLD", "prob": 0.6},
            {"decision": "SELL", "prob": 0.9},
        ]

        for trade in trades:
            manager.add_trade_entry(trade)

        assert manager.get_entry_count() == 3
        assert manager.entries == trades

    def test_rolling_window_basic(self):
        """Test that rolling window removes oldest entries"""
        manager = JournalManager(max_entries=3)

        # Add 4 entries (exceeds max of 3)
        trades = [
            {"id": 1, "decision": "BUY"},
            {"id": 2, "decision": "HOLD"},
            {"id": 3, "decision": "SELL"},
            {"id": 4, "decision": "BUY"},  # This should cause removal of id=1
        ]

        for trade in trades:
            manager.add_trade_entry(trade)

        assert manager.get_entry_count() == 3
        assert len(manager.entries) == 3
        # Should contain entries 2, 3, 4 (oldest removed)
        assert manager.entries[0]["id"] == 2
        assert manager.entries[1]["id"] == 3
        assert manager.entries[2]["id"] == 4

    def test_rolling_window_edge_cases(self):
        """Test rolling window edge cases"""
        manager = JournalManager(max_entries=2)

        # Add exactly max_entries
        manager.add_trade_entry({"id": 1})
        manager.add_trade_entry({"id": 2})

        assert manager.get_entry_count() == 2

        # Add one more - should remove oldest
        manager.add_trade_entry({"id": 3})

        assert manager.get_entry_count() == 2
        assert manager.entries[0]["id"] == 2
        assert manager.entries[1]["id"] == 3

    def test_max_entries_one(self):
        """Test with max_entries=1"""
        manager = JournalManager(max_entries=1)

        manager.add_trade_entry({"id": 1})
        assert manager.get_entry_count() == 1

        manager.add_trade_entry({"id": 2})
        assert manager.get_entry_count() == 1
        assert manager.entries[0]["id"] == 2

    def test_large_max_entries(self):
        """Test with large max_entries (no rolling window triggered)"""
        manager = JournalManager(max_entries=100)

        # Add 5 entries (well below max)
        for i in range(5):
            manager.add_trade_entry({"id": i})

        assert manager.get_entry_count() == 5
        assert all(entry["id"] == i for i, entry in enumerate(manager.entries))

    def test_empty_state_methods(self):
        """Test methods work correctly on empty journal"""
        manager = JournalManager()

        assert manager.is_empty() == True
        assert manager.get_entry_count() == 0

    def test_entry_data_preservation(self):
        """Test that complex trade data is preserved correctly"""
        manager = JournalManager()

        complex_trade = {
            "date": datetime(2023, 1, 1),
            "decision": "BUY",
            "prob": 0.75,
            "next_return_1d": 1.23,
            "strategy_return": 1.23,
            "cumulative_return": 5.67,
            "cumulative_index_return": 3.45,
            "rsi_14": 65.4,
            "macd_line": 1.23,
            "macd_signal": 1.15,
            "macd_histogram": 0.08,
            "stoch_k": 75.2,
            "stoch_d": 72.8,
            "bb_position": 0.85,
        }

        manager.add_trade_entry(complex_trade)

        assert manager.get_entry_count() == 1
        assert manager.entries[0] == complex_trade
        # Verify all fields preserved
        assert manager.entries[0]["rsi_14"] == 65.4
        assert manager.entries[0]["macd_histogram"] == 0.08

    def test_get_relative_time_label_today(self):
        """Test relative time label for same day"""
        past = datetime(2023, 1, 1, 10, 0)
        current = datetime(2023, 1, 1, 15, 0)

        label = JournalManager.get_relative_time_label(past, current)
        assert label == "today"

    def test_get_relative_time_label_days(self):
        """Test relative time labels for days"""
        current = datetime(2023, 1, 10)

        # 1 day ago
        past = datetime(2023, 1, 9)
        assert JournalManager.get_relative_time_label(past, current) == "1 day ago"

        # 5 days ago
        past = datetime(2023, 1, 5)
        assert JournalManager.get_relative_time_label(past, current) == "5 days ago"

    def test_get_relative_time_label_weeks(self):
        """Test relative time labels for weeks"""
        current = datetime(2023, 1, 15)

        # 2 weeks ago (14-20 days)
        past = datetime(2023, 1, 1)
        assert JournalManager.get_relative_time_label(past, current) == "2 weeks ago"

        # 4 weeks ago (28-34 days)
        past = datetime(2022, 12, 18)
        assert JournalManager.get_relative_time_label(past, current) == "4 weeks ago"

    def test_get_relative_time_label_months_years(self):
        """Test relative time labels for months and years"""
        current = datetime(2023, 6, 1)

        # 2 months ago (60-89 days)
        past = datetime(2023, 4, 1)
        assert JournalManager.get_relative_time_label(past, current) == "2 months ago"

        # 1 year ago
        past = datetime(2022, 6, 1)
        assert JournalManager.get_relative_time_label(past, current) == "1 year ago"

        # 3 years ago
        past = datetime(2020, 6, 1)
        assert JournalManager.get_relative_time_label(past, current) == "3 years ago"

    def test_format_single_entry_with_dates(self):
        """Test single entry formatting with absolute dates"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 10)

        trade_data = {
            "date": datetime(2023, 1, 5),
            "decision": "BUY",
            "prob": 0.75,
            "next_return_1d": 1.23,
            "strategy_return": 1.23,
            "cumulative_return": 5.67,
            "index_cumulative_return": 3.45,
            "explanation": "Market shows bullish signals",
            "strategic_journal": "Maintaining bullish bias",
            "feeling_log": "Confident in analysis",
        }

        entry = manager.format_single_entry(
            trade_data, current_date, show_dates=True, enable_technical_indicators=False
        )

        assert "Date 2023-01-05:" in entry
        assert "action BUY (prob 0.75)" in entry
        assert "next day index return 1.23 percent" in entry
        assert "strategy return 1.23 percent" in entry
        assert "cumulative strategy return 5.67 percent" in entry
        assert "cumulative index return 3.45 percent" in entry
        assert "Explanation: Market shows bullish signals" in entry
        assert "Strategic journal: Maintaining bullish bias" in entry
        assert "Feeling: Confident in analysis" in entry

    def test_format_single_entry_without_dates(self):
        """Test single entry formatting with relative dates"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 10)

        trade_data = {
            "date": datetime(2023, 1, 5),  # 5 days ago
            "decision": "SELL",
            "prob": 0.82,
            "next_return_1d": -0.45,
            "strategy_return": -0.45,
            "cumulative_return": 2.34,
            "index_cumulative_return": 1.89,
            "explanation": "Risk indicators elevated",
            "strategic_journal": "Taking defensive position",
            "feeling_log": "Cautious approach",
        }

        entry = manager.format_single_entry(
            trade_data,
            current_date,
            show_dates=False,
            enable_technical_indicators=False,
        )

        assert "5 days ago:" in entry
        assert "action SELL (prob 0.82)" in entry
        assert "next day index return -0.45 percent" in entry
        assert "strategy return -0.45 percent" in entry
        assert "Explanation: Risk indicators elevated" in entry

    def test_format_single_entry_with_technical_indicators(self):
        """Test single entry formatting with technical indicators"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 10)

        trade_data = {
            "date": datetime(2023, 1, 5),
            "decision": "HOLD",
            "prob": 0.60,
            "next_return_1d": 0.12,
            "strategy_return": 0.0,
            "cumulative_return": 1.45,
            "index_cumulative_return": 2.10,
            "rsi_14": 65.4,
            "macd_line": 1.23,
            "macd_signal": 1.15,
            "macd_histogram": 0.08,
            "stoch_k": 75.2,
            "stoch_d": 72.8,
            "bb_position": 0.85,
            "explanation": "Mixed signals",
            "strategic_journal": "Waiting for clarity",
            "feeling_log": "Neutral stance",
        }

        entry = manager.format_single_entry(
            trade_data, current_date, show_dates=False, enable_technical_indicators=True
        )

        assert "5 days ago:" in entry
        assert (
            "Technical indicators: RSI(14): 65.4 | MACD: 1.23/1.15/0.080 | Stochastic: 75.2/72.8 | BB Position: 0.85"
            in entry
        )

    def test_format_single_entry_missing_technical_data(self):
        """Test single entry formatting when technical indicators are missing or NaN"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 10)

        trade_data = {
            "date": datetime(2023, 1, 5),
            "decision": "BUY",
            "prob": 0.70,
            "next_return_1d": 1.0,
            "strategy_return": 1.0,
            "cumulative_return": 3.0,
            "index_cumulative_return": 2.0,
            "rsi_14": None,  # Missing data
            "macd_line": 1.0,
            "macd_signal": pd.NA,  # NaN data
            "macd_histogram": 0.1,
            "explanation": "Test",
            "strategic_journal": "Test",
            "feeling_log": "Test",
        }

        entry = manager.format_single_entry(
            trade_data, current_date, show_dates=False, enable_technical_indicators=True
        )

        # Should not include technical indicators when data is missing/NaN
        assert "Technical indicators:" not in entry

    def test_get_journal_block_empty(self):
        """Test journal block when no entries exist"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 10)

        block = manager.get_journal_block(
            current_date, show_dates=False, enable_technical_indicators=False
        )

        assert block == "No past trades yet. You are starting your strategy."

    def test_get_journal_block_with_entries(self):
        """Test journal block with entries"""
        manager = JournalManager()
        current_date = datetime(2023, 1, 15)

        # Add a few entries
        trades = [
            {
                "date": datetime(2023, 1, 10),  # 5 days ago
                "decision": "BUY",
                "prob": 0.8,
                "next_return_1d": 1.5,
                "strategy_return": 1.5,
                "cumulative_return": 1.5,
                "index_cumulative_return": 1.0,
                "explanation": "Bullish",
                "strategic_journal": "Buying",
                "feeling_log": "Optimistic",
            },
            {
                "date": datetime(2023, 1, 12),  # 3 days ago
                "decision": "HOLD",
                "prob": 0.6,
                "next_return_1d": 0.0,
                "strategy_return": 0.0,
                "cumulative_return": 1.5,
                "index_cumulative_return": 1.2,
                "explanation": "Neutral",
                "strategic_journal": "Holding",
                "feeling_log": "Cautious",
            },
        ]

        for trade in trades:
            manager.add_trade_entry(trade)

        block = manager.get_journal_block(
            current_date, show_dates=False, enable_technical_indicators=False
        )

        assert block.startswith("Past trades and results so far:")
        assert "5 days ago: action BUY" in block
        assert "3 days ago: action HOLD" in block
        assert "Bullish" in block
        assert "Neutral" in block

    def test_get_journal_block_rolling_window(self):
        """Test that journal block only shows last 10 entries"""
        manager = JournalManager(max_entries=12)  # Allow more than 10

        current_date = datetime(2023, 1, 20)

        # Add 15 entries
        for i in range(15):
            trade = {
                "date": datetime(2023, 1, i + 1),
                "decision": "BUY",
                "prob": 0.8,
                "next_return_1d": 1.0,
                "strategy_return": 1.0,
                "cumulative_return": float(i),
                "index_cumulative_return": float(i),
                "explanation": f"Trade {i}",
                "strategic_journal": f"Journal {i}",
                "feeling_log": f"Feeling {i}",
            }
            manager.add_trade_entry(trade)

        # Should have all 15 entries stored
        assert manager.get_entry_count() == 12  # Rolling window limit

        # But journal block should only show last 10
        block = manager.get_journal_block(
            current_date, show_dates=False, enable_technical_indicators=False
        )

        lines = block.split("\n")
        # Header + 10 entries = 11 lines
        assert len(lines) == 11
        assert "Past trades and results so far:" in lines[0]
