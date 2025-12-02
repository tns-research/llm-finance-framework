"""
Period Manager for LLM Finance Framework

Unified period boundary management and statistics tracking that replaces
the duplicated logic for weekly, monthly, quarterly, and yearly periods.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import os
import pandas as pd

from .memory_classes import PeriodStats
from .memory_manager import MemoryManager
from .reporting import generate_llm_period_summary, compute_period_technical_stats
from .configuration_manager import ConfigurationManager


class PeriodManager:
    """
    Unified period boundary management and statistics tracking.

    This class replaces the 100+ lines of duplicated boundary checking
    and period management logic that existed for each period type.
    """

    def __init__(self, memory_manager: MemoryManager, config_manager: ConfigurationManager = None):
        """
        Initialize period manager with memory manager and config manager dependencies.

        Args:
            memory_manager: MemoryManager instance for storing period summaries
            config_manager: ConfigurationManager instance for accessing settings
        """
        self.memory_manager = memory_manager
        self.config_manager = config_manager or ConfigurationManager()
        self.stats: Dict[str, PeriodStats] = {
            'weekly': PeriodStats(),
            'monthly': PeriodStats(),
            'quarterly': PeriodStats(),
            'yearly': PeriodStats(),
        }
        self.last_dates: Dict[str, Optional[datetime]] = {
            'weekly': None,
            'monthly': None,
            'quarterly': None,
            'yearly': None,
        }

    def update_stats(self, period: str, **kwargs):
        """
        Update statistics for a period.

        This replaces the duplicated stats update logic that was repeated
        4 times for each period type.

        Args:
            period: 'weekly', 'monthly', 'quarterly', or 'yearly'
            **kwargs: Statistics to update (strategy_return, index_return, days, wins, etc.)
        """
        stats = self.stats[period]
        for key, value in kwargs.items():
            if hasattr(stats, key):
                current_value = getattr(stats, key)
                if isinstance(current_value, (int, float)):
                    setattr(stats, key, current_value + value)
                else:
                    setattr(stats, key, value)

    def should_summarize_period(self, period: str, current_date: datetime, last_date: Optional[datetime]) -> bool:
        """
        Check if period boundary has been crossed.

        This replaces the duplicated boundary checking logic for each period type.

        Args:
            period: Period type to check
            current_date: Current trading date
            last_date: Previous trading date

        Returns:
            True if period boundary has been crossed
        """
        if last_date is None:
            return False

        if period == 'weekly':
            return self._is_week_boundary(current_date, last_date)
        elif period == 'monthly':
            return current_date.month != last_date.month or current_date.year != last_date.year
        elif period == 'quarterly':
            current_quarter = (current_date.month - 1) // 3 + 1
            last_quarter = (last_date.month - 1) // 3 + 1
            return current_quarter != last_quarter or current_date.year != last_date.year
        elif period == 'yearly':
            return current_date.year != last_date.year

        return False

    def _is_week_boundary(self, current_date: datetime, last_date: datetime) -> bool:
        """
        Check if ISO week boundary has been crossed.

        Uses ISO week numbering which handles year transitions correctly.
        """
        current_iso = current_date.isocalendar()
        last_iso = last_date.isocalendar()
        return current_iso[1] != last_iso[1] or current_iso[0] != last_iso[0]

    def generate_period_summary(self, period: str, end_date: datetime,
                               router_model: str, model_tag: str) -> str:
        """
        Generate period summary and add to memory.

        This replaces the duplicated summary generation logic for each period.

        Args:
            period: Period type ('weekly', 'monthly', etc.)
            end_date: End date of the period
            router_model: Model identifier for LLM calls
            model_tag: Model tag for identification

        Returns:
            Generated summary text
        """
        stats = self.stats[period]

        if stats.days == 0:
            return ""

        # Compute technical stats if enabled
        technical_stats = None
        flags = self.config_manager.get_feature_flags()
        if flags['ENABLE_TECHNICAL_INDICATORS']:
            technical_stats = self._compute_technical_stats(period, end_date)

        # Generate LLM summary
        summary = generate_llm_period_summary(
            period.capitalize(),
            end_date,
            stats.to_dict(),  # Convert to dict for compatibility
            router_model,
            model_tag,
            technical_stats
        )

        # Add to memory
        self.memory_manager.add_memory_item(period, summary, technical_stats)

        # Reset stats for next period
        self.stats[period].reset()

        return summary

    def _compute_technical_stats(self, period: str, end_date: datetime) -> Optional[dict]:
        """
        Compute technical stats for the period.

        Args:
            period: Period type
            end_date: End date of the period

        Returns:
            Technical statistics dict or None if computation fails
        """
        try:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            features_path = os.path.join(base_dir, "data", "processed", "features.csv")
            features_df = pd.read_csv(features_path, parse_dates=['date'])

            config = self.memory_manager.period_configs[period]
            start_date = end_date - pd.Timedelta(days=config.date_offset_days)

            return compute_period_technical_stats(features_df, start_date, end_date)
        except Exception as e:
            print(f"Warning: Could not compute technical stats for {period}: {e}")
            return None

    def check_all_periods(self, current_date: datetime, last_date: Optional[datetime],
                         router_model: str, model_tag: str):
        """
        Check all period boundaries and generate summaries as needed.

        This replaces the 100+ lines of duplicated boundary checking logic.

        Args:
            current_date: Current trading date
            last_date: Previous trading date
            router_model: Model identifier for LLM calls
            model_tag: Model tag for identification
        """
        for period in self.stats.keys():
            if self.should_summarize_period(period, current_date, last_date):
                self.generate_period_summary(period, last_date, router_model, model_tag)
                self.last_dates[period] = last_date

    def get_period_stats(self, period: str) -> PeriodStats:
        """Get statistics for a specific period"""
        return self.stats[period]

    def reset_all_stats(self):
        """Reset all period statistics"""
        for stats in self.stats.values():
            stats.reset()

    def get_active_periods(self) -> list:
        """Get list of periods that have accumulated statistics"""
        return [period for period, stats in self.stats.items() if stats.days > 0]
