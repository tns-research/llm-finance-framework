"""
Unit tests for PerformanceTracker class
"""

import pytest
from src.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    """Test cases for PerformanceTracker class"""

    def test_initial_state(self):
        """Test that PerformanceTracker initializes with correct default values"""
        tracker = PerformanceTracker()

        # Performance metrics
        assert tracker.cumulative_return == 0.0
        assert tracker.index_cumulative_return == 0.0

        # Decision counts
        assert tracker.decision_count == 0
        assert tracker.buy_count == 0
        assert tracker.hold_count == 0
        assert tracker.sell_count == 0
        assert tracker.win_count == 0

        # Position tracking
        assert tracker.current_decision is None
        assert tracker.current_position_duration == 0

    def test_first_day_buy_position(self):
        """Test first day with BUY decision"""
        tracker = PerformanceTracker()

        tracker.update_daily_performance("BUY", 1.5, 0.5)

        assert tracker.decision_count == 1
        assert tracker.buy_count == 1
        assert tracker.hold_count == 0
        assert tracker.sell_count == 0
        assert tracker.win_count == 1  # 1.5 > 0
        assert tracker.cumulative_return == 1.5
        assert tracker.index_cumulative_return == 0.5
        assert tracker.current_decision == "BUY"
        assert tracker.current_position_duration == 1

    def test_first_day_sell_position(self):
        """Test first day with SELL decision"""
        tracker = PerformanceTracker()

        tracker.update_daily_performance("SELL", -0.8, 0.3)

        assert tracker.decision_count == 1
        assert tracker.buy_count == 0
        assert tracker.hold_count == 0
        assert tracker.sell_count == 1
        assert tracker.win_count == 0  # -0.8 < 0
        assert tracker.cumulative_return == -0.8
        assert tracker.index_cumulative_return == 0.3
        assert tracker.current_decision == "SELL"
        assert tracker.current_position_duration == 1

    def test_first_day_hold_position(self):
        """Test first day with HOLD decision"""
        tracker = PerformanceTracker()

        tracker.update_daily_performance("HOLD", 0.0, 0.2)

        assert tracker.decision_count == 1
        assert tracker.buy_count == 0
        assert tracker.hold_count == 1
        assert tracker.sell_count == 0
        assert tracker.win_count == 0  # 0.0 is not > 0
        assert tracker.cumulative_return == 0.0
        assert tracker.index_cumulative_return == 0.2
        assert tracker.current_decision == "HOLD"
        assert tracker.current_position_duration == 1

    def test_position_duration_same_decision(self):
        """Test position duration increases when decision stays the same"""
        tracker = PerformanceTracker()

        # Day 1: BUY
        tracker.update_daily_performance("BUY", 1.0, 0.5)
        assert tracker.current_position_duration == 1

        # Day 2: BUY again
        tracker.update_daily_performance("BUY", -0.5, 0.3)
        assert tracker.current_position_duration == 2

        # Day 3: BUY again
        tracker.update_daily_performance("BUY", 2.0, -0.1)
        assert tracker.current_position_duration == 3

    def test_position_duration_change_decision(self):
        """Test position duration resets when decision changes"""
        tracker = PerformanceTracker()

        # Day 1: BUY
        tracker.update_daily_performance("BUY", 1.0, 0.5)
        assert tracker.current_decision == "BUY"
        assert tracker.current_position_duration == 1

        # Day 2: HOLD (different decision)
        tracker.update_daily_performance("HOLD", 0.0, 0.3)
        assert tracker.current_decision == "HOLD"
        assert tracker.current_position_duration == 1  # Reset to 1

        # Day 3: HOLD again
        tracker.update_daily_performance("HOLD", 0.5, 0.2)
        assert tracker.current_decision == "HOLD"
        assert tracker.current_position_duration == 2

    def test_multiple_decisions_mixed(self):
        """Test tracking multiple decisions of different types"""
        tracker = PerformanceTracker()

        # Mix of decisions
        tracker.update_daily_performance("BUY", 1.0, 0.5)   # Win
        tracker.update_daily_performance("HOLD", 0.0, 0.3)  # No win
        tracker.update_daily_performance("SELL", -0.5, 0.2) # Loss
        tracker.update_daily_performance("BUY", 2.0, -0.1)  # Win

        assert tracker.decision_count == 4
        assert tracker.buy_count == 2
        assert tracker.hold_count == 1
        assert tracker.sell_count == 1
        assert tracker.win_count == 2  # Two positive returns

        assert tracker.cumulative_return == 2.5  # 1.0 + 0.0 + (-0.5) + 2.0
        assert tracker.index_cumulative_return == 0.9  # 0.5 + 0.3 + 0.2 + (-0.1)

    def test_performance_summary_no_trades(self):
        """Test performance summary when no trades have been executed"""
        tracker = PerformanceTracker()

        summary = tracker.get_performance_summary()
        expected = (
            "No trades executed yet.\n"
            "Strategy cumulative return so far  0.00 percent.\n"
            "S and P 500 cumulative return so far  0.00 percent.\n"
            "BUY 0, HOLD 0, SELL 0.\n"
            "Win rate undefined."
        )
        assert summary == expected

    def test_performance_summary_with_trades_outperforming(self):
        """Test performance summary when strategy is outperforming"""
        tracker = PerformanceTracker()

        # Add some performance data
        tracker.update_daily_performance("BUY", 2.0, 1.0)
        tracker.update_daily_performance("HOLD", 0.0, 0.5)
        tracker.update_daily_performance("SELL", -1.0, 0.5)
        tracker.update_daily_performance("BUY", 3.0, 1.0)

        summary = tracker.get_performance_summary()

        # Check key elements
        assert "Total strategy return so far  4.00 percent." in summary
        assert "Total S and P 500 return so far  3.00 percent." in summary
        assert "outperforming the index by 1.00 percent." in summary
        assert "Number of decisions so far  4 (BUY 2, HOLD 1, SELL 1)." in summary
        assert "Win rate so far  50.0 percent." in summary

    def test_performance_summary_underperforming(self):
        """Test performance summary when strategy is underperforming"""
        tracker = PerformanceTracker()

        # Strategy underperforms
        tracker.update_daily_performance("BUY", 1.0, 2.0)
        tracker.update_daily_performance("HOLD", 0.0, 1.0)

        summary = tracker.get_performance_summary()

        assert "Total strategy return so far  1.00 percent." in summary
        assert "Total S and P 500 return so far  3.00 percent." in summary
        assert "underperforming the index by -2.00 percent." in summary

    def test_position_duration_info(self):
        """Test getting position duration information"""
        tracker = PerformanceTracker()

        # No decisions yet
        decision, duration = tracker.get_position_duration_info()
        assert decision is None
        assert duration == 0

        # After first decision
        tracker.update_daily_performance("BUY", 1.0, 0.5)
        decision, duration = tracker.get_position_duration_info()
        assert decision == "BUY"
        assert duration == 1

        # After continuing same decision
        tracker.update_daily_performance("BUY", 2.0, 0.3)
        decision, duration = tracker.get_position_duration_info()
        assert decision == "BUY"
        assert duration == 2

    def test_final_metrics(self):
        """Test getting final metrics for reporting"""
        tracker = PerformanceTracker()

        # Add some data
        tracker.update_daily_performance("BUY", 2.0, 1.0)
        tracker.update_daily_performance("HOLD", 0.0, 0.5)
        tracker.update_daily_performance("SELL", -1.0, 0.5)
        tracker.update_daily_performance("BUY", 3.0, 1.0)

        metrics = tracker.get_final_metrics()

        expected = {
            "total_return": 4.0,  # 2.0 + 0.0 + (-1.0) + 3.0
            "index_return": 3.0,  # 1.0 + 0.5 + 0.5 + 1.0
            "total_decisions": 4,
            "buy_decisions": 2,
            "hold_decisions": 1,
            "sell_decisions": 1,
            "win_count": 2,  # Two positive returns: 2.0 and 3.0
            "win_rate": 0.5,  # 2/4
            "current_position": "BUY",
            "current_position_duration": 1,  # Last decision was BUY after previous HOLD
        }

        assert metrics == expected

    def test_final_metrics_no_trades(self):
        """Test final metrics when no trades have been made"""
        tracker = PerformanceTracker()

        metrics = tracker.get_final_metrics()

        expected = {
            "total_return": 0.0,
            "index_return": 0.0,
            "total_decisions": 0,
            "buy_decisions": 0,
            "hold_decisions": 0,
            "sell_decisions": 0,
            "win_count": 0,
            "win_rate": 0.0,
            "current_position": None,
            "current_position_duration": 0,
        }

        assert metrics == expected

    def test_reset(self):
        """Test resetting tracker to initial state"""
        tracker = PerformanceTracker()

        # Add some data
        tracker.update_daily_performance("BUY", 1.0, 0.5)
        tracker.update_daily_performance("HOLD", 0.5, 0.3)

        # Verify data exists
        assert tracker.decision_count == 2
        assert tracker.cumulative_return == 1.5

        # Reset
        tracker.reset()

        # Verify back to initial state
        assert tracker.decision_count == 0
        assert tracker.cumulative_return == 0.0
        assert tracker.current_decision is None
        assert tracker.current_position_duration == 0

    def test_win_rate_edge_cases(self):
        """Test win rate calculation edge cases"""
        tracker = PerformanceTracker()

        # All wins
        tracker.update_daily_performance("BUY", 1.0, 0.5)
        tracker.update_daily_performance("BUY", 2.0, 0.3)
        assert tracker.get_final_metrics()["win_rate"] == 1.0

        # All losses
        tracker.reset()
        tracker.update_daily_performance("SELL", -1.0, 0.5)
        tracker.update_daily_performance("SELL", -2.0, 0.3)
        assert tracker.get_final_metrics()["win_rate"] == 0.0

        # Mixed results
        tracker.reset()
        tracker.update_daily_performance("BUY", 1.0, 0.5)   # Win
        tracker.update_daily_performance("SELL", -1.0, 0.3) # Loss
        tracker.update_daily_performance("HOLD", 0.0, 0.2)  # No win (0.0 not > 0)
        assert tracker.get_final_metrics()["win_rate"] == 0.3333333333333333  # 1/3

