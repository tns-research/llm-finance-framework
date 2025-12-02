"""
Performance Tracker for LLM Finance Framework

Extracted performance tracking logic from trading engine for better testability
and maintainability.
"""

from typing import Optional, Dict, Any, Tuple


class PerformanceTracker:
    """
    Tracks performance metrics for trading simulations.

    Consolidates all performance-related variables and calculations that were
    previously scattered throughout the trading engine.
    """

    def __init__(self):
        """Initialize performance tracking state"""
        # Core performance metrics
        self.cumulative_return = 0.0
        self.index_cumulative_return = 0.0

        # Decision tracking
        self.decision_count = 0
        self.buy_count = 0
        self.hold_count = 0
        self.sell_count = 0
        self.win_count = 0  # days with positive strategy return

        # Position duration tracking
        self.current_decision: Optional[str] = None
        self.current_position_duration = 0

    def update_daily_performance(self, decision: str, daily_return: float, index_return: float) -> None:
        """
        Update performance metrics for a single trading day.

        Args:
            decision: The trading decision ("BUY", "HOLD", or "SELL")
            daily_return: Strategy return for this day (position * next_return_1d)
            index_return: Index return for this day
        """
        # Update position duration tracking
        if self.current_decision is None:
            # First day
            self.current_decision = decision
            self.current_position_duration = 1
        elif decision == self.current_decision:
            # Same position as yesterday
            self.current_position_duration += 1
        else:
            # Position changed
            self.current_decision = decision
            self.current_position_duration = 1

        # Update cumulative returns
        self.cumulative_return += daily_return
        self.index_cumulative_return += index_return

        # Update decision statistics
        self.decision_count += 1
        if decision == "BUY":
            self.buy_count += 1
        elif decision == "HOLD":
            self.hold_count += 1
        elif decision == "SELL":
            self.sell_count += 1

        # Update win count
        if daily_return > 0:
            self.win_count += 1

    def get_performance_summary(self) -> str:
        """
        Generate formatted performance summary for LLM prompts.

        Returns:
            Formatted string showing current performance metrics
        """
        if self.decision_count == 0:
            return (
                "No trades executed yet.\n"
                "Strategy cumulative return so far  0.00 percent.\n"
                "S and P 500 cumulative return so far  0.00 percent.\n"
                "BUY 0, HOLD 0, SELL 0.\n"
                "Win rate undefined."
            )
        else:
            edge = self.cumulative_return - self.index_cumulative_return
            outperform_word = "outperforming" if edge > 0 else "underperforming"
            win_rate_pct = (self.win_count / self.decision_count) * 100.0

            return (
                f"Total strategy return so far  {self.cumulative_return:.2f} percent.\n"
                f"Total S and P 500 return so far  {self.index_cumulative_return:.2f} percent.\n"
                f"You are {outperform_word} the index by {edge:.2f} percent.\n"
                f"Number of decisions so far  {self.decision_count} "
                f"(BUY {self.buy_count}, HOLD {self.hold_count}, SELL {self.sell_count}).\n"
                f"Win rate so far  {win_rate_pct:.1f} percent."
            )

    def get_position_duration_info(self) -> Tuple[Optional[str], int]:
        """
        Get current position and duration information.

        Returns:
            Tuple of (current_decision, current_position_duration)
        """
        return self.current_decision, self.current_position_duration

    def get_final_metrics(self) -> Dict[str, Any]:
        """
        Get final performance metrics for reporting.

        Returns:
            Dictionary with all performance metrics
        """
        return {
            "total_return": self.cumulative_return,
            "index_return": self.index_cumulative_return,
            "total_decisions": self.decision_count,
            "buy_decisions": self.buy_count,
            "hold_decisions": self.hold_count,
            "sell_decisions": self.sell_count,
            "win_count": self.win_count,
            "win_rate": (self.win_count / self.decision_count) if self.decision_count > 0 else 0.0,
            "current_position": self.current_decision,
            "current_position_duration": self.current_position_duration,
        }

    def reset(self) -> None:
        """Reset all metrics to initial state (useful for testing)"""
        self.__init__()

