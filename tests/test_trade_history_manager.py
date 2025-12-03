"""
Unit tests for TradeHistoryManager class
"""

import pytest
from datetime import datetime
from src.trade_history_manager import TradeHistoryManager


class TestTradeHistoryManager:
    """Test cases for TradeHistoryManager class"""

    def test_initialization(self):
        """Test TradeHistoryManager initializes correctly"""
        manager = TradeHistoryManager()

        assert manager.entries == []
        assert manager.get_entry_count() == 0
        assert manager.is_empty() == True

    def test_add_single_entry_with_dates(self):
        """Test adding a single trade entry with date visibility"""
        manager = TradeHistoryManager()
        date = datetime(2023, 1, 1)

        manager.add_trade_entry(date, "BUY", 1.0, 1.5, show_dates=True)

        assert manager.get_entry_count() == 1
        assert manager.is_empty() == False

        entry = manager.entries[0]
        assert entry["date"] == "2023-01-01"
        assert entry["trade_id"] == 1
        assert entry["decision"] == "BUY"
        assert entry["position"] == 1.0
        assert entry["result"] == 1.5

    def test_add_single_entry_without_dates(self):
        """Test adding a single trade entry without date visibility"""
        manager = TradeHistoryManager()
        date = datetime(2023, 1, 1)

        manager.add_trade_entry(date, "SELL", -1.0, -0.8, show_dates=False)

        assert manager.get_entry_count() == 1

        entry = manager.entries[0]
        assert entry["date"] == "2023-01-01"  # Always stored
        assert entry["trade_id"] == 1
        assert entry["decision"] == "SELL"
        assert entry["position"] == -1.0
        assert entry["result"] == -0.8

    def test_add_multiple_entries(self):
        """Test adding multiple trade entries"""
        manager = TradeHistoryManager()

        dates = [datetime(2023, 1, i) for i in range(1, 4)]
        trades = [
            ("BUY", 1.0, 1.5),
            ("HOLD", 0.0, 0.0),
            ("SELL", -1.0, -0.8)
        ]

        for date, (decision, position, result) in zip(dates, trades):
            manager.add_trade_entry(date, decision, position, result, show_dates=True)

        assert manager.get_entry_count() == 3

        # Check sequential trade_ids
        assert manager.entries[0]["trade_id"] == 1
        assert manager.entries[1]["trade_id"] == 2
        assert manager.entries[2]["trade_id"] == 3

        # Check dates are preserved
        assert manager.entries[0]["date"] == "2023-01-01"
        assert manager.entries[1]["date"] == "2023-01-02"
        assert manager.entries[2]["date"] == "2023-01-03"

    def test_result_rounding(self):
        """Test that results are rounded to 6 decimal places"""
        manager = TradeHistoryManager()

        # Test with more precision than 6 decimals
        manager.add_trade_entry(datetime(2023, 1, 1), "BUY", 1.0, 1.123456789, show_dates=True)

        assert manager.entries[0]["result"] == 1.123457  # Rounded to 6 decimals

    def test_get_history_block_with_dates_enabled(self):
        """Test CSV formatting with dates when enabled"""
        manager = TradeHistoryManager()

        manager.add_trade_entry(datetime(2023, 1, 1), "BUY", 1.0, 1.5, show_dates=True)
        manager.add_trade_entry(datetime(2023, 1, 2), "SELL", -1.0, -0.8, show_dates=True)

        block = manager.get_history_block(show_dates=True, enabled=True)

        expected = (
            "TRADING_HISTORY:\n"
            "date,decision,position,result\n"
            "2023-01-01,BUY,1.0,1.5\n"
            "2023-01-02,SELL,-1.0,-0.8"
        )

        assert block == expected

    def test_get_history_block_without_dates_enabled(self):
        """Test CSV formatting with trade_ids when dates disabled"""
        manager = TradeHistoryManager()

        manager.add_trade_entry(datetime(2023, 1, 1), "BUY", 1.0, 1.5, show_dates=False)
        manager.add_trade_entry(datetime(2023, 1, 2), "HOLD", 0.0, 0.0, show_dates=False)

        block = manager.get_history_block(show_dates=False, enabled=True)

        expected = (
            "TRADING_HISTORY:\n"
            "trade_id,decision,position,result\n"
            "1,BUY,1.0,1.5\n"
            "2,HOLD,0.0,0.0"
        )

        assert block == expected

    def test_get_history_block_empty_enabled(self):
        """Test history block when enabled but no entries"""
        manager = TradeHistoryManager()

        block = manager.get_history_block(show_dates=True, enabled=True)

        assert block == "TRADING_HISTORY:\nNo trading history yet."

    def test_get_history_block_disabled(self):
        """Test history block when feature is disabled"""
        manager = TradeHistoryManager()

        # Add some entries
        manager.add_trade_entry(datetime(2023, 1, 1), "BUY", 1.0, 1.5, show_dates=True)

        block = manager.get_history_block(show_dates=True, enabled=False)

        assert block == "TRADING_HISTORY:\nNo trading history yet."

    def test_get_history_block_empty_and_disabled(self):
        """Test history block when both empty and disabled"""
        manager = TradeHistoryManager()

        block = manager.get_history_block(show_dates=True, enabled=False)

        assert block == "TRADING_HISTORY:\nNo trading history yet."

    def test_mixed_decisions_and_positions(self):
        """Test with various decision types and position values"""
        manager = TradeHistoryManager()

        test_cases = [
            ("BUY", 1.0, 2.5),
            ("HOLD", 0.0, 0.0),
            ("SELL", -1.0, -1.2),
        ]

        for decision, position, result in test_cases:
            manager.add_trade_entry(datetime(2023, 1, 1), decision, position, result, show_dates=True)

        block = manager.get_history_block(show_dates=True, enabled=True)

        lines = block.split('\n')
        assert "2023-01-01,BUY,1.0,2.5" in lines
        assert "2023-01-01,HOLD,0.0,0.0" in lines
        assert "2023-01-01,SELL,-1.0,-1.2" in lines

    def test_large_number_of_entries(self):
        """Test with many entries to ensure no performance issues"""
        manager = TradeHistoryManager()

        # Add 50 entries with valid dates
        base_date = datetime(2023, 1, 1)
        for i in range(50):
            # Create valid dates by using day 1-28 to avoid month overflow
            days_to_add = i % 28  # Keep within valid day range
            date = base_date.replace(day=1 + days_to_add)
            manager.add_trade_entry(date, "BUY", 1.0, 1.0, show_dates=True)

        assert manager.get_entry_count() == 50

        block = manager.get_history_block(show_dates=True, enabled=True)
        lines = [line for line in block.split('\n') if line.strip()]  # Remove empty lines

        # Should have: TRADING_HISTORY: + csv_header + 50 data lines = 52 lines
        assert len(lines) == 52
        assert lines[0] == "TRADING_HISTORY:"
        assert lines[1] == "date,decision,position,result"
        assert lines[-1].startswith("2023-")  # Last line should be a data line

    def test_show_dates_parameter_independence(self):
        """Test that show_dates parameter at add time doesn't affect display"""
        manager = TradeHistoryManager()

        # Add entries with different show_dates settings
        manager.add_trade_entry(datetime(2023, 1, 1), "BUY", 1.0, 1.5, show_dates=True)
        manager.add_trade_entry(datetime(2023, 1, 2), "SELL", -1.0, -0.8, show_dates=False)

        # Both should be able to display with dates
        block_with_dates = manager.get_history_block(show_dates=True, enabled=True)
        assert "date,decision,position,result" in block_with_dates
        assert "2023-01-01,BUY,1.0,1.5" in block_with_dates
        assert "2023-01-02,SELL,-1.0,-0.8" in block_with_dates

        # Both should be able to display without dates
        block_without_dates = manager.get_history_block(show_dates=False, enabled=True)
        assert "trade_id,decision,position,result" in block_without_dates
        assert "1,BUY,1.0,1.5" in block_without_dates
        assert "2,SELL,-1.0,-0.8" in block_without_dates
