#!/usr/bin/env python3
"""
Experiment date-dimension coverage.

The feature-flag synchronization tests exercise the
chain_of_thought / strategic_journal / feeling_log axes but always run with
ACTIVE_EXPERIMENT="baseline", so the SHOW_DATE_TO_LLM (dates_*) axis is never
asserted. These tests close that gap by:

1. Checking every experiment in EXPERIMENT_CONFIGS maps to its documented
   (dates, memory, feeling) triple, and
2. Checking row_to_prompt emits a Date header iff SHOW_DATE_TO_LLM is True.
"""

from unittest.mock import patch

import pandas as pd

from src import config
from src.prompts import row_to_prompt

# (experiment name) -> (SHOW_DATE_TO_LLM, ENABLE_STRATEGIC_JOURNAL, ENABLE_FEELING_LOG)
EXPECTED_FLAGS = {
    "baseline": (False, False, False),
    "memory_only": (False, True, False),
    "memory_feeling": (False, True, True),
    "dates_only": (True, False, False),
    "dates_memory": (True, True, False),
    "dates_full": (True, True, True),
}


def _make_row(show_date: bool) -> pd.Series:
    data = {f"ret_lag_{k}": 0.1 * k for k in range(1, config.PAST_RET_LAGS + 1)}
    data["date"] = pd.Timestamp("2020-03-15")
    data["ma20_pct"] = 1.23
    data["vol20_annualized"] = 15.0
    data["ret_5d"] = 0.45
    return pd.Series(data)


def test_experiment_configs_match_documented_flags():
    """Every documented experiment exists and maps to the right flag triple."""
    for name, (dates, memory, feeling) in EXPECTED_FLAGS.items():
        assert name in config.EXPERIMENT_CONFIGS, f"missing experiment: {name}"
        cfg = config.EXPERIMENT_CONFIGS[name]
        assert cfg["SHOW_DATE_TO_LLM"] == dates, f"{name}: dates flag"
        assert cfg["ENABLE_STRATEGIC_JOURNAL"] == memory, f"{name}: memory flag"
        assert cfg["ENABLE_FEELING_LOG"] == feeling, f"{name}: feeling flag"


def test_no_undocumented_real_experiments():
    """Guard against silently adding a real experiment without documenting it.

    The deprecated 'chain_of_thought' preset is intentionally excluded; it is
    superseded by the ENABLE_CHAIN_OF_THOUGHT toggle.
    """
    real = set(config.EXPERIMENT_CONFIGS) - {"chain_of_thought"}
    assert real == set(EXPECTED_FLAGS), (
        "EXPERIMENT_CONFIGS drifted from documented set; update README "
        "strategy table and EXPECTED_FLAGS together"
    )


def test_date_header_present_only_when_dates_enabled():
    """row_to_prompt emits the Date line iff SHOW_DATE_TO_LLM is True."""
    for name, (dates, _memory, _feeling) in EXPECTED_FLAGS.items():
        with patch("src.config.SHOW_DATE_TO_LLM", dates):
            prompt = row_to_prompt(_make_row(dates))
        has_date = "Date" in prompt and "2020 03 15" in prompt
        assert has_date == dates, f"{name}: date header presence should be {dates}"
