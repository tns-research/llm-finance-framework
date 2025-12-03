"""
Unit tests for the unified memory management system.

Tests cover all components: MemoryItem, PeriodStats, PeriodConfig,
MemoryManager, and PeriodManager classes.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.memory_classes import MemoryItem, PeriodConfig, PeriodStats
from src.memory_manager import MemoryManager
from src.period_manager import PeriodManager


class TestMemoryItem:
    """Test MemoryItem dataclass functionality"""

    def test_creation(self):
        """Test basic MemoryItem creation"""
        item = MemoryItem(summary="Test summary")
        assert item.summary == "Test summary"
        assert item.technical_stats is None
        assert isinstance(item.timestamp, datetime)

    def test_creation_with_tech_stats(self):
        """Test MemoryItem with technical statistics"""
        tech_stats = {"rsi": 50.0, "macd": 0.5}
        item = MemoryItem(summary="Test", technical_stats=tech_stats)
        assert item.technical_stats == tech_stats

    def test_to_dict(self):
        """Test conversion to dictionary format"""
        tech_stats = {"rsi": 50}
        item = MemoryItem(summary="Test", technical_stats=tech_stats)
        data = item.to_dict()
        assert data["summary"] == "Test"
        assert data["technical_stats"]["rsi"] == 50

    def test_from_dict(self):
        """Test creation from dictionary format"""
        data = {"summary": "Test", "technical_stats": {"rsi": 50}}
        item = MemoryItem.from_dict(data)
        assert item.summary == "Test"
        assert item.technical_stats["rsi"] == 50

    def test_from_string(self):
        """Test creation from legacy string format"""
        text = "Legacy summary text"
        item = MemoryItem.from_string(text)
        assert item.summary == text
        assert item.technical_stats is None


class TestPeriodStats:
    """Test PeriodStats dataclass functionality"""

    def test_creation(self):
        """Test basic PeriodStats creation"""
        stats = PeriodStats()
        assert stats.strategy_return == 0.0
        assert stats.days == 0
        assert stats.wins == 0
        assert stats.buys == 0
        assert stats.holds == 0
        assert stats.sells == 0

    def test_creation_with_values(self):
        """Test PeriodStats with initial values"""
        stats = PeriodStats(strategy_return=100.5, days=10, wins=7)
        assert stats.strategy_return == 100.5
        assert stats.days == 10
        assert stats.wins == 7

    def test_reset(self):
        """Test stats reset functionality"""
        stats = PeriodStats(
            strategy_return=100.0, days=5, wins=3, buys=2, holds=1, sells=2
        )
        stats.reset()
        assert stats.strategy_return == 0.0
        assert stats.days == 0
        assert stats.wins == 0
        assert stats.buys == 0
        assert stats.holds == 0
        assert stats.sells == 0

    def test_to_dict(self):
        """Test conversion to dictionary"""
        stats = PeriodStats(strategy_return=50.0, days=5, wins=3)
        data = stats.to_dict()
        assert data["strategy_return"] == 50.0
        assert data["days"] == 5
        assert data["wins"] == 3

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {"strategy_return": 25.0, "days": 3, "wins": 2}
        stats = PeriodStats.from_dict(data)
        assert stats.strategy_return == 25.0
        assert stats.days == 3
        assert stats.wins == 2


class TestPeriodConfig:
    """Test PeriodConfig dataclass functionality"""

    def test_creation(self):
        """Test basic PeriodConfig creation"""
        config = PeriodConfig("weekly")
        assert config.name == "weekly"
        assert config.max_memory_items == 5
        assert config.date_offset_days == 7

    def test_plural_name_weekly(self):
        """Test plural name for weekly"""
        config = PeriodConfig("weekly")
        assert config.plural_name == "weeks"

    def test_plural_name_monthly(self):
        """Test plural name for monthly"""
        config = PeriodConfig("monthly")
        assert config.plural_name == "months"

    def test_plural_name_quarterly(self):
        """Test plural name for quarterly"""
        config = PeriodConfig("quarterly")
        assert config.plural_name == "quarters"

    def test_plural_name_yearly(self):
        """Test plural name for yearly"""
        config = PeriodConfig("yearly")
        assert config.plural_name == "years"

    def test_custom_values(self):
        """Test PeriodConfig with custom values"""
        config = PeriodConfig("weekly", max_memory_items=3, date_offset_days=6)
        assert config.max_memory_items == 3
        assert config.date_offset_days == 6


class TestMemoryManager:
    """Test MemoryManager functionality"""

    def test_creation(self):
        """Test MemoryManager initialization"""
        manager = MemoryManager()
        # defaultdict starts empty, but we can check the period_configs
        assert len(manager.period_configs) == 4
        assert "weekly" in manager.period_configs
        assert "monthly" in manager.period_configs
        assert "quarterly" in manager.period_configs
        assert "yearly" in manager.period_configs

        # Accessing memories creates entries
        _ = manager.memories["weekly"]
        assert "weekly" in manager.memories

    def test_add_memory_item(self):
        """Test adding memory items"""
        manager = MemoryManager()
        manager.add_memory_item("weekly", "Test summary")
        assert len(manager.memories["weekly"]) == 1
        assert manager.memories["weekly"][0].summary == "Test summary"
        assert manager.memories["weekly"][0].period == "weekly"

    def test_add_memory_item_with_tech_stats(self):
        """Test adding memory items with technical stats"""
        manager = MemoryManager()
        tech_stats = {"rsi": 70.0}
        manager.add_memory_item("monthly", "Monthly summary", tech_stats)
        item = manager.memories["monthly"][0]
        assert item.summary == "Monthly summary"
        assert item.technical_stats == tech_stats

    def test_memory_limit(self):
        """Test memory item limit enforcement"""
        manager = MemoryManager()
        # Add 7 items (limit is 5)
        for i in range(7):
            manager.add_memory_item("weekly", f"Summary {i}")
        assert len(manager.memories["weekly"]) == 5
        # Should keep the most recent 5
        assert manager.memories["weekly"][0].summary == "Summary 2"
        assert manager.memories["weekly"][-1].summary == "Summary 6"

    def test_get_memory_block_empty(self):
        """Test getting memory block when empty"""
        manager = MemoryManager()
        block = manager.get_memory_block("weekly")
        assert block == "No weekly summaries yet."

    def test_get_memory_block_with_items(self):
        """Test getting memory block with items"""
        manager = MemoryManager()
        manager.add_memory_item("weekly", "First summary")
        manager.add_memory_item("weekly", "Second summary")

        block = manager.get_memory_block("weekly")
        assert "Weekly memory (most recent first)" in block
        assert "1 week ago" in block
        assert "2 weeks ago" in block
        assert "First summary" in block
        assert "Second summary" in block

    def test_get_all_memory_blocks(self):
        """Test getting all memory blocks"""
        manager = MemoryManager()
        manager.add_memory_item("weekly", "Week summary")

        blocks = manager.get_all_memory_blocks()
        assert "weekly" in blocks
        assert "monthly" in blocks
        assert "quarterly" in blocks
        assert "yearly" in blocks
        assert "Week summary" in blocks["weekly"]

    def test_has_memory(self):
        """Test memory existence checking"""
        manager = MemoryManager()
        assert not manager.has_memory("weekly")
        manager.add_memory_item("weekly", "Test")
        assert manager.has_memory("weekly")

    def test_clear_memory_specific(self):
        """Test clearing specific period memory"""
        manager = MemoryManager()
        manager.add_memory_item("weekly", "Week summary")
        manager.add_memory_item("monthly", "Month summary")

        manager.clear_memory("weekly")
        assert not manager.has_memory("weekly")
        assert manager.has_memory("monthly")

    def test_clear_memory_all(self):
        """Test clearing all memory"""
        manager = MemoryManager()
        manager.add_memory_item("weekly", "Week summary")
        manager.add_memory_item("monthly", "Month summary")

        manager.clear_memory()
        assert not manager.has_memory("weekly")
        assert not manager.has_memory("monthly")


class TestPeriodManager:
    """Test PeriodManager functionality"""

    def test_creation(self):
        """Test PeriodManager initialization"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        assert len(period_manager.stats) == 4
        assert "weekly" in period_manager.stats
        assert "monthly" in period_manager.stats
        assert "quarterly" in period_manager.stats
        assert "yearly" in period_manager.stats

    def test_update_stats(self):
        """Test updating period statistics"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        period_manager.update_stats("weekly", strategy_return=10.5, days=1)
        assert period_manager.stats["weekly"].strategy_return == 10.5
        assert period_manager.stats["weekly"].days == 1

        # Test incremental updates
        period_manager.update_stats("weekly", strategy_return=5.0, wins=1)
        assert period_manager.stats["weekly"].strategy_return == 15.5
        assert period_manager.stats["weekly"].wins == 1

    def test_should_summarize_period_weekly_same_week(self):
        """Test weekly boundary check - same week"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 1, 2)  # Monday
        date2 = datetime(2023, 1, 3)  # Tuesday
        assert not period_manager.should_summarize_period("weekly", date2, date1)

    def test_should_summarize_period_weekly_different_week(self):
        """Test weekly boundary check - different weeks"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 1, 3)  # Tuesday week 1
        date2 = datetime(2023, 1, 9)  # Monday week 2
        assert period_manager.should_summarize_period("weekly", date2, date1)

    def test_should_summarize_period_monthly_same_month(self):
        """Test monthly boundary check - same month"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 1, 15)
        date2 = datetime(2023, 1, 20)
        assert not period_manager.should_summarize_period("monthly", date2, date1)

    def test_should_summarize_period_monthly_different_month(self):
        """Test monthly boundary check - different months"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 1, 31)
        date2 = datetime(2023, 2, 1)
        assert period_manager.should_summarize_period("monthly", date2, date1)

    def test_should_summarize_period_yearly_same_year(self):
        """Test yearly boundary check - same year"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 6, 15)
        date2 = datetime(2023, 12, 31)
        assert not period_manager.should_summarize_period("yearly", date2, date1)

    def test_should_summarize_period_yearly_different_year(self):
        """Test yearly boundary check - different years"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        date1 = datetime(2023, 12, 31)
        date2 = datetime(2024, 1, 1)
        assert period_manager.should_summarize_period("yearly", date2, date1)

    @patch("src.period_manager.generate_llm_period_summary")
    def test_generate_period_summary(self, mock_generate):
        """Test period summary generation"""
        mock_generate.return_value = "Mocked summary"

        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        # Add some stats
        period_manager.update_stats("weekly", strategy_return=100.0, days=5, wins=3)

        end_date = datetime(2023, 1, 7)
        summary = period_manager.generate_period_summary(
            "weekly", end_date, "router", "model"
        )

        assert summary == "Mocked summary"
        mock_generate.assert_called_once()

        # Check that stats were reset
        assert period_manager.stats["weekly"].days == 0

    def test_get_period_stats(self):
        """Test getting period statistics"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        period_manager.update_stats("monthly", strategy_return=50.0, days=10)
        stats = period_manager.get_period_stats("monthly")

        assert stats.strategy_return == 50.0
        assert stats.days == 10

    def test_reset_all_stats(self):
        """Test resetting all period statistics"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        period_manager.update_stats("weekly", strategy_return=100.0, days=5)
        period_manager.update_stats("monthly", strategy_return=200.0, days=20)

        period_manager.reset_all_stats()

        assert period_manager.stats["weekly"].strategy_return == 0.0
        assert period_manager.stats["weekly"].days == 0
        assert period_manager.stats["monthly"].strategy_return == 0.0
        assert period_manager.stats["monthly"].days == 0

    def test_get_active_periods(self):
        """Test getting periods with accumulated statistics"""
        memory_manager = MemoryManager()
        period_manager = PeriodManager(memory_manager)

        # No active periods initially
        assert period_manager.get_active_periods() == []

        # Add stats to weekly
        period_manager.update_stats("weekly", days=1)
        assert period_manager.get_active_periods() == ["weekly"]

        # Add stats to monthly
        period_manager.update_stats("monthly", days=1)
        active = period_manager.get_active_periods()
        assert "weekly" in active
        assert "monthly" in active
