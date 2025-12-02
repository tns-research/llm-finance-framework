"""
Memory Management Classes for LLM Finance Framework

This module contains the core data structures for unified memory management,
replacing the duplicated memory logic throughout the codebase.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class MemoryItem:
    """
    Standardized memory item format for all period types.

    Replaces the inconsistent mix of string and dict formats used previously.
    """
    summary: str
    technical_stats: Optional[dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    period: str = ""  # 'weekly', 'monthly', 'quarterly', 'yearly'

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility"""
        return {
            "summary": self.summary,
            "technical_stats": self.technical_stats
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryItem':
        """Create from dict format (backward compatibility)"""
        return cls(
            summary=data["summary"],
            technical_stats=data.get("technical_stats")
        )

    @classmethod
    def from_string(cls, text: str) -> 'MemoryItem':
        """Create from old string format (backward compatibility)"""
        return cls(summary=text)


@dataclass
class PeriodStats:
    """
    Unified period statistics tracking.

    Replaces the duplicated stats objects for each period type.
    """
    strategy_return: float = 0.0
    index_return: float = 0.0
    days: int = 0
    wins: int = 0
    buys: int = 0
    holds: int = 0
    sells: int = 0

    def reset(self):
        """Reset all stats to zero for next period"""
        self.strategy_return = 0.0
        self.index_return = 0.0
        self.days = 0
        self.wins = 0
        self.buys = 0
        self.holds = 0
        self.sells = 0

    def to_dict(self) -> dict:
        """Convert to dict format for compatibility with existing functions"""
        return {
            "strategy_return": self.strategy_return,
            "index_return": self.index_return,
            "days": self.days,
            "wins": self.wins,
            "buys": self.buys,
            "holds": self.holds,
            "sells": self.sells,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PeriodStats':
        """Create from dict format"""
        return cls(**data)


@dataclass
class PeriodConfig:
    """
    Configuration for each period type.

    Centralizes period-specific settings that were previously hardcoded.
    """
    name: str
    max_memory_items: int = 5
    date_offset_days: int = 7  # For computing technical stats

    @property
    def plural_name(self) -> str:
        """Get plural form (weeks, months, etc.)"""
        if self.name == "weekly":
            return "weeks"
        elif self.name == "monthly":
            return "months"
        elif self.name == "quarterly":
            return "quarters"
        elif self.name == "yearly":
            return "years"
        return self.name
