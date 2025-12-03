"""
Trade History Manager for LLM Finance Framework

Manages the complete trading history that provides LLMs with full chronological
context of all past trades for pattern recognition and learning.
"""

from datetime import datetime
from typing import Any, Dict, List


class TradeHistoryManager:
    """
    Manages trading history entries for LLM memory system.

    The trading history provides LLMs with complete chronological context by storing
    all past trades in a structured format. This enables pattern recognition across
    the entire trading period and supports different output formats (with/without dates).
    """

    def __init__(self):
        """
        Initialize trade history manager.
        """
        self.entries: List[Dict[str, Any]] = []

    def add_trade_entry(
        self,
        date: datetime,
        decision: str,
        position: float,
        daily_return: float,
        show_dates: bool,
    ) -> None:
        """
        Add a trade entry to the history, storing complete information for flexible formatting.

        Args:
            date: Trade date
            decision: Trading decision (BUY/HOLD/SELL)
            position: Position value (-1, 0, 1)
            daily_return: Strategy return for this trade
            show_dates: Whether to include actual dates or use trade_ids (for backward compatibility)
        """
        # Always store complete information for flexible formatting
        entry = {
            "date": str(date.date()),  # Convert to string for JSON serialization
            "trade_id": len(self.entries) + 1,  # Simple sequential ID
            "decision": decision,
            "position": position,
            "result": round(float(daily_return), 6),  # Strategy return for this trade
        }

        self.entries.append(entry)

    def get_entry_count(self) -> int:
        """Get the number of entries in the trading history."""
        return len(self.entries)

    def is_empty(self) -> bool:
        """Check if the trading history is empty."""
        return len(self.entries) == 0

    def get_history_block(self, show_dates: bool, enabled: bool = True) -> str:
        """
        Get formatted trading history block for LLM prompt.

        Args:
            show_dates: Whether to show dates or trade_ids
            enabled: Whether trading history feature is enabled

        Returns:
            Formatted trading history block as CSV string
        """
        if not enabled or self.is_empty():
            return "TRADING_HISTORY:\nNo trading history yet."

        if show_dates:
            # Include dates when date mode is enabled
            header = "date,decision,position,result"
            history_lines = [header]
            for entry in self.entries:
                line = f"{entry['date']},{entry['decision']},{entry['position']},{entry['result']}"
                history_lines.append(line)
        else:
            # Omit dates in anonymized mode
            header = "trade_id,decision,position,result"
            history_lines = [header]
            for entry in self.entries:
                line = f"{entry['trade_id']},{entry['decision']},{entry['position']},{entry['result']}"
                history_lines.append(line)

        return "TRADING_HISTORY:\n" + "\n".join(history_lines)
