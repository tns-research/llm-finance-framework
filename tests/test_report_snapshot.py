# Golden snapshot harness for the full master report (markdown + HTML).
#
# This is the safety net for the report_generator refactor. It feeds a
# fixed, deterministic ``data_sources`` dict to ``generate_master_report`` and
# ``generate_master_report_html``, neutralises the wall-clock timestamp, and
# compares the output byte-for-byte against committed golden files.
#
# Any pure-move refactor (extracting modules, splitting sections) must keep these
# goldens byte-identical. A deliberate output change must regenerate them in the
# same commit, with justification:
#
#     REGEN_GOLDEN=1 .venv/bin/python -m pytest tests/test_report_snapshot.py
#
# The fixture is hand-built (not captured from a live run or the LLM) so the
# snapshot is reproducible without network access or a model.

import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.report_generator as rg

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXED_NOW = datetime(2024, 1, 2, 9, 30, 0)


def build_data_sources() -> dict:
    """A deterministic, reasonably comprehensive data_sources dict.

    Populates the keys that drive real rendering logic across the high-signal
    sections (executive summary, baselines, indicator alignment, strategy
    comparison, statistical validation, category performance). Sections without
    data render their graceful "not available" branch, which the golden also
    locks in.
    """
    baseline_comparison = pd.DataFrame(
        {
            "baseline": [
                "buy_and_hold",
                "momentum",
                "rsi_mean_reversion",
                "stochastic_oscillator",
                "macd_momentum",
                "bollinger_bands",
                "moving_average_crossover",
                "random",
            ],
            "total_return": [45.2, 32.1, 28.7, 24.3, 19.8, 22.5, 30.0, -5.2],
            "sharpe_annualized": [1.2, 0.8, 0.9, 0.7, 0.6, 0.65, 0.85, -0.3],
            "win_rate": [52.1, 51.2, 53.4, 50.8, 49.2, 50.1, 51.9, 48.5],
            "max_drawdown": [-18.2, -22.1, -15.3, -19.8, -25.0, -20.2, -17.5, -35.0],
        }
    )

    llm_indicator_alignment = {
        "RSI": {
            "alignment_rate": 0.78,
            "total_signals": 245,
            "agreements": 191,
            "bullish_signals": 120,
            "bearish_signals": 125,
            "llm_buy_signals": 115,
            "llm_sell_signals": 118,
            "description": "RSI oversold/overbought mean reversion",
        },
        "MACD": {
            "alignment_rate": 0.65,
            "total_signals": 245,
            "agreements": 159,
            "bullish_signals": 118,
            "bearish_signals": 127,
            "llm_buy_signals": 112,
            "llm_sell_signals": 122,
            "description": "MACD line crossover trend signals",
        },
        "Stochastic": {
            "alignment_rate": 0.42,
            "total_signals": 245,
            "agreements": 103,
            "bullish_signals": 115,
            "bearish_signals": 130,
            "llm_buy_signals": 108,
            "llm_sell_signals": 115,
            "description": "Stochastic oscillator mean reversion",
        },
    }

    statistical_validation = {
        "dataset_info": {
            "total_strategy_return": 38.5,
            "total_index_return": 30.0,
            "n_periods": 245,
            "sharpe_ratio": 1.1,
            "win_rate": 54.2,
            "date_range": {"start": "2022-01-03", "end": "2022-12-30"},
        },
        "bootstrap_vs_index": {
            "strategy_sharpe": 1.10,
            "benchmark_sharpe": 0.95,
            "sharpe_difference": 0.15,
            "significant_difference_5pct": True,
            "strategy_significantly_better_5pct": True,
            "strategy_significantly_worse_5pct": False,
            "effect_size": 0.42,
            "p_value_two_sided": 0.031,
            "ci_95_bootstrap": [0.02, 0.28],
        },
        "summary_assessment": {
            "overall_assessment": "Good",
            "confidence_level": "Moderate",
            "key_findings": [
                "Strategy beats the index on a risk-adjusted basis.",
                "Bootstrap test is significant at the 5% level.",
            ],
            "recommendations": [
                "Validate on an out-of-sample window before deployment.",
            ],
        },
    }

    parsed_data = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=12, freq="D"),
            "decision": [
                "BUY",
                "HOLD",
                "SELL",
                "HOLD",
                "BUY",
                "BUY",
                "HOLD",
                "SELL",
                "BUY",
                "HOLD",
                "SELL",
                "HOLD",
            ],
            "strategy_return": [
                0.012,
                0.0,
                -0.008,
                0.0,
                0.015,
                0.006,
                0.0,
                -0.004,
                0.009,
                0.0,
                -0.011,
                0.0,
            ],
            "next_return_1d": [
                0.010,
                0.002,
                -0.006,
                0.001,
                0.013,
                0.005,
                -0.001,
                -0.003,
                0.008,
                0.000,
                -0.009,
                0.002,
            ],
            "regime": [
                "bull",
                "bull",
                "bear",
                "bear",
                "bull",
                "bull",
                "sideways",
                "bear",
                "bull",
                "sideways",
                "bear",
                "sideways",
            ],
        }
    )

    return {
        "baseline_comparison": baseline_comparison,
        "llm_indicator_alignment": llm_indicator_alignment,
        "statistical_validation": statistical_validation,
        "parsed_data": parsed_data,
    }


class _FrozenDatetime(datetime):
    """datetime subclass whose now() is pinned, leaving the rest intact."""

    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


def _datetime_modules():
    """Modules whose ``datetime.now()`` must be frozen for a deterministic render.

    Covers report_generator (in case it ever emits one) plus the split-out
    section modules that emit a timestamp. The owning module is resolved from
    the bound function so the patch lands on the exact object report_generator
    calls into, regardless of which import path (``report.*`` vs ``src.report.*``)
    bound it. Modules without a module-level ``datetime`` (e.g. report_generator,
    now a pure facade) have nothing to freeze and are skipped.
    """
    modules = {id(rg): rg}
    for fn in (
        rg.generate_master_report,
        rg.generate_master_report_html,
        rg.generate_technical_details,
        rg.generate_technical_details_html,
    ):
        mod = sys.modules.get(fn.__module__)
        if mod is not None:
            modules[id(mod)] = mod
    return [mod for mod in modules.values() if hasattr(mod, "datetime")]


def _render(kind: str) -> str:
    data_sources = build_data_sources()
    analysis_dir = Path("/tmp/snapshot_analysis")
    plots_dir = Path("/tmp/snapshot_plots")
    with ExitStack() as stack:
        for mod in _datetime_modules():
            stack.enter_context(patch.object(mod, "datetime", _FrozenDatetime))
        if kind == "md":
            return rg.generate_master_report(
                "snapshot_model", data_sources, analysis_dir, plots_dir
            )
        return rg.generate_master_report_html(
            "snapshot_model", data_sources, analysis_dir, plots_dir
        )


def _check(kind: str, filename: str):
    rendered = _render(kind)
    golden_path = GOLDEN_DIR / filename
    if os.environ.get("REGEN_GOLDEN"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(rendered, encoding="utf-8")
        return
    assert golden_path.exists(), (
        f"Golden {golden_path} missing. Generate it once with "
        f"REGEN_GOLDEN=1 pytest tests/test_report_snapshot.py"
    )
    expected = golden_path.read_text(encoding="utf-8")
    assert rendered == expected, (
        f"{filename} drifted from golden. If intentional, regenerate with "
        f"REGEN_GOLDEN=1 and justify the change in the commit."
    )


def test_master_report_markdown_matches_golden():
    _check("md", "master_report.md")


def test_master_report_html_matches_golden():
    _check("html", "master_report.html")


def test_render_is_deterministic():
    """Same fixture twice must yield identical output (no hidden nondeterminism)."""
    assert _render("md") == _render("md")
    assert _render("html") == _render("html")
