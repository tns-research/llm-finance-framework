"""
Journal Manager for LLM Finance Framework

Manages the strategic journal entries that provide LLMs with short-term memory
of their recent trading decisions and outcomes.
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


class JournalManager:
    """
    Manages strategic journal entries for LLM memory system.

    The strategic journal provides LLMs with short-term memory by showing
    their recent trading decisions, outcomes, and reasoning. This enables
    self-reflection and consistency in trading behavior.
    """

    def __init__(self, max_entries: int = 10):
        """
        Initialize journal manager.

        Args:
            max_entries: Maximum number of entries to keep (rolling window)
        """
        self.entries: List[Dict[str, Any]] = []
        self.max_entries = max_entries

    def add_trade_entry(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a trade entry to the journal, maintaining rolling window.

        Args:
            trade_data: Dictionary containing trade information
        """
        self.entries.append(trade_data)

        # Maintain rolling window by removing oldest entries
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)  # Remove oldest entry

    def get_entry_count(self) -> int:
        """Get current number of entries in journal."""
        return len(self.entries)

    def is_empty(self) -> bool:
        """Check if journal has any entries."""
        return len(self.entries) == 0

    @staticmethod
    def get_relative_time_label(past_date: datetime, current_date: datetime) -> str:
        """
        Calculate relative time label for journal entries in 'no date' mode.
        Returns labels like '1 week ago', '2 weeks ago', '1 month ago', etc.
        """
        days_diff = (current_date - past_date).days

        if days_diff == 0:
            return "today"
        elif days_diff == 1:
            return "1 day ago"
        elif days_diff < 14:
            return f"{days_diff} days ago"
        elif days_diff < 21:
            weeks = 2
            return f"{weeks} weeks ago"
        elif days_diff < 28:
            weeks = 3
            return f"{weeks} weeks ago"
        elif days_diff < 35:
            weeks = 4
            return f"{weeks} weeks ago"
        elif days_diff < 60:
            return "1 month ago"
        elif days_diff < 90:
            return "2 months ago"
        elif days_diff < 120:
            return "3 months ago"
        elif days_diff < 150:
            return "4 months ago"
        elif days_diff < 180:
            return "5 months ago"
        elif days_diff < 210:
            return "6 months ago"
        elif days_diff < 240:
            return "7 months ago"
        elif days_diff < 270:
            return "8 months ago"
        elif days_diff < 300:
            return "9 months ago"
        elif days_diff < 330:
            return "10 months ago"
        elif days_diff < 365:
            return "11 months ago"
        else:
            years = days_diff // 365
            if years == 1:
                return "1 year ago"
            else:
                return f"{years} years ago"

    def format_single_entry(
        self,
        trade_data: Dict[str, Any],
        current_date: datetime,
        show_dates: bool,
        enable_technical_indicators: bool,
    ) -> str:
        """
        Format a single journal entry with appropriate date labeling.

        Args:
            trade_data: Dictionary containing trade information
            current_date: Current date for relative time calculations
            show_dates: Whether to show absolute dates or relative time
            enable_technical_indicators: Whether to include technical indicators

        Returns:
            Formatted journal entry string
        """
        if show_dates:
            entry_prefix = f"Date {trade_data['date'].strftime('%Y-%m-%d')}: "
        else:
            # Use relative time in 'no date' mode
            relative_label = self.get_relative_time_label(
                trade_data["date"], current_date
            )
            entry_prefix = f"{relative_label}: "

        base_entry = (
            entry_prefix
            + f"action {trade_data['decision']} (prob {trade_data['prob']:.2f}), "
            f"next day index return {trade_data['next_return_1d']:.2f} percent, "
            f"strategy return {trade_data['strategy_return']:.2f} percent, "
            f"cumulative strategy return {trade_data['cumulative_return']:.2f} percent, "
            f"cumulative index return {trade_data['index_cumulative_return']:.2f} percent."
        )

        # Add technical indicators if available and enabled
        if enable_technical_indicators and "rsi_14" in trade_data:
            tech_indicators = []

            if trade_data.get("rsi_14") is not None and not pd.isna(
                trade_data["rsi_14"]
            ):
                tech_indicators.append(f"RSI(14): {trade_data['rsi_14']:.1f}")

            if (
                trade_data.get("macd_line") is not None
                and trade_data.get("macd_signal") is not None
                and trade_data.get("macd_histogram") is not None
                and not any(
                    pd.isna(
                        [
                            trade_data["macd_line"],
                            trade_data["macd_signal"],
                            trade_data["macd_histogram"],
                        ]
                    )
                )
            ):
                tech_indicators.append(
                    f"MACD: {trade_data['macd_line']:.2f}/{trade_data['macd_signal']:.2f}/{trade_data['macd_histogram']:.3f}"
                )

            if (
                trade_data.get("stoch_k") is not None
                and trade_data.get("stoch_d") is not None
                and not any(pd.isna([trade_data["stoch_k"], trade_data["stoch_d"]]))
            ):
                tech_indicators.append(
                    f"Stochastic: {trade_data['stoch_k']:.1f}/{trade_data['stoch_d']:.1f}"
                )

            if trade_data.get("bb_position") is not None and not pd.isna(
                trade_data["bb_position"]
            ):
                tech_indicators.append(f"BB Position: {trade_data['bb_position']:.2f}")

            if tech_indicators:
                base_entry += f" Technical indicators: {' | '.join(tech_indicators)}."

        base_entry += f" Explanation: {trade_data['explanation']} Strategic journal: {trade_data['strategic_journal']} Feeling: {trade_data['feeling_log']}"

        return base_entry

    def get_journal_block(
        self,
        current_date: datetime,
        show_dates: bool,
        enable_technical_indicators: bool,
    ) -> str:
        """
        Get formatted journal block for LLM prompt.

        Args:
            current_date: Current date for relative time calculations
            show_dates: Whether to show absolute dates or relative time
            enable_technical_indicators: Whether to include technical indicators

        Returns:
            Formatted journal block string for LLM prompt
        """
        if self.is_empty():
            return "No past trades yet. You are starting your strategy."

        # Get last 10 entries (or all if less than 10)
        recent_entries = self.entries[-10:]

        formatted_entries = [
            self.format_single_entry(
                entry, current_date, show_dates, enable_technical_indicators
            )
            for entry in recent_entries
        ]

        return "Past trades and results so far:\n" + "\n".join(formatted_entries)
