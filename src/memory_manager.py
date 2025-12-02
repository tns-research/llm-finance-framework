"""
Memory Manager for LLM Finance Framework

Unified memory management system that replaces the duplicated memory logic
across weekly, monthly, quarterly, and yearly periods.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import os
import pandas as pd
from datetime import datetime

from .memory_classes import MemoryItem, PeriodConfig
from .reporting import compute_period_technical_stats, format_period_technical_indicators
from .config_compat import ENABLE_TECHNICAL_INDICATORS


class MemoryManager:
    """
    Unified memory management system for all period types.

    This class centralizes all memory operations that were previously
    duplicated across four separate memory lists and logic blocks.
    """

    def __init__(self):
        """Initialize memory manager with period configurations"""
        self.memories: Dict[str, List[MemoryItem]] = defaultdict(list)
        self.period_configs = {
            'weekly': PeriodConfig('weekly', max_memory_items=5, date_offset_days=6),
            'monthly': PeriodConfig('monthly', max_memory_items=5, date_offset_days=30),
            'quarterly': PeriodConfig('quarterly', max_memory_items=5, date_offset_days=90),
            'yearly': PeriodConfig('yearly', max_memory_items=5, date_offset_days=365),
        }

    def add_memory_item(self, period: str, summary: str, technical_stats: Optional[dict] = None):
        """
        Add a memory item for the specified period.

        Args:
            period: 'weekly', 'monthly', 'quarterly', or 'yearly'
            summary: LLM-generated period summary text
            technical_stats: Optional technical indicators data
        """
        config = self.period_configs[period]

        memory_item = MemoryItem(
            summary=summary,
            technical_stats=technical_stats,
            period=period
        )

        self.memories[period].append(memory_item)
        # Keep only the most recent items (circular buffer)
        self.memories[period] = self.memories[period][-config.max_memory_items:]

    def get_memory_block(self, period: str) -> str:
        """
        Generate formatted memory block for prompts.

        This replaces the 60+ lines of duplicated formatting logic
        that existed for each period type.
        """
        config = self.period_configs[period]
        memory_items = self.memories[period]

        if not memory_items:
            return f"No {config.name} summaries yet."

        # Build labeled items (most recent first)
        labeled_items = []
        for i, memory_item in enumerate(reversed(memory_items)):
            units_ago = i + 1
            if units_ago == 1:
                label = f"1 {config.name[:-2]} ago"  # Remove 'ly' from weekly->week
            else:
                label = f"{units_ago} {config.plural_name} ago"

            # Format memory text
            memory_text = f"{label}:\n{memory_item.summary}"

            # Add technical indicators if available
            if memory_item.technical_stats:
                tech_text = format_period_technical_indicators(
                    memory_item.technical_stats,
                    period.capitalize()
                )
                memory_text += tech_text

            labeled_items.append(memory_text)

        return f"{period.capitalize()} memory (most recent first)\n" + "\n\n".join(labeled_items)

    def get_all_memory_blocks(self) -> Dict[str, str]:
        """Get all memory blocks as a dictionary"""
        return {
            period: self.get_memory_block(period)
            for period in self.period_configs.keys()
        }

    def has_memory(self, period: str) -> bool:
        """Check if period has any memory items"""
        return len(self.memories[period]) > 0

    def clear_memory(self, period: Optional[str] = None):
        """
        Clear memory for specific period or all periods.

        Args:
            period: Specific period to clear, or None to clear all
        """
        if period:
            self.memories[period].clear()
        else:
            self.memories.clear()

    def get_memory_count(self, period: str) -> int:
        """Get number of memory items for a period"""
        return len(self.memories[period])

    def get_oldest_memory(self, period: str) -> Optional[MemoryItem]:
        """Get the oldest (first) memory item for a period"""
        if self.memories[period]:
            return self.memories[period][0]
        return None

    def get_newest_memory(self, period: str) -> Optional[MemoryItem]:
        """Get the newest (last) memory item for a period"""
        if self.memories[period]:
            return self.memories[period][-1]
        return None
