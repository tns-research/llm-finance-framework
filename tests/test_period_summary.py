# Characterization tests for generate_llm_period_summary in
# src/reporting/period_summary.py.
#
# This function is the largest in the module and is only ever *mocked* in the
# rest of the suite, so its real output text was previously unguarded. These
# tests pin the exact string produced by every branch -- days==0 early return,
# the dummy/no-router fallback template (with and without technical_stats, and
# both SHOW_DATE_TO_LLM modes), the real-LLM happy path, and the error fallback
# -- so the upcoming decomposition (extracting _build_period_prompt and
# _fallback_period_summary) can be proven byte-identical.

import datetime as dt

import pytest

import src.reporting.period_summary as ps

END = dt.datetime(2022, 3, 25)

STATS = {
    "strategy_return": 4.20,
    "index_return": 1.80,
    "days": 20,
    "wins": 12,
    "buys": 7,
    "holds": 9,
    "sells": 4,
}

TECH = {
    "rsi_avg": 47.5,
    "rsi_overbought_pct": 10.0,
    "rsi_oversold_pct": 5.0,
    "rsi_min": 22.0,
    "rsi_max": 78.0,
    "macd_bullish_pct": 60.0,
    "macd_avg_histogram": 0.123,
    "macd_crossovers": 3,
    "stoch_overbought_pct": 25.0,
    "stoch_oversold_pct": 8.0,
    "stoch_min": 12.0,
    "stoch_max": 90.0,
    "bb_upper_touch_pct": 6.0,
    "bb_lower_touch_pct": 7.0,
    "bb_avg_position": 0.55,
}


@pytest.fixture
def patch_flags(monkeypatch):
    """Pin the module-level config flags so tests are independent of the
    active experiment. Returns a setter for SHOW_DATE_TO_LLM / USE_DUMMY_MODEL."""

    def _set(*, show_date, use_dummy):
        monkeypatch.setattr(ps, "SHOW_DATE_TO_LLM", show_date)
        monkeypatch.setattr(ps, "USE_DUMMY_MODEL", use_dummy)

    return _set


def test_days_zero_early_return():
    out = ps.generate_llm_period_summary(
        "Week", END, {**STATS, "days": 0}, "anything", "mtag"
    )
    assert out == "Week ending 2022-03-25  no trading activity recorded."


def test_fallback_with_date_no_tech(patch_flags):
    patch_flags(show_date=True, use_dummy=True)
    out = ps.generate_llm_period_summary("Month", END, STATS, None, "mtag")
    assert out == (
        "Month ending 2022-03-25\n"
        "Explanation: Month ending 2022-03-25. Market total return  1.80 percent. "
        "Strategy total return  4.20 percent. The strategy outperformed the index "
        "by 2.40 percent over 20 days.\n"
        "Strategic journal: During this period you traded BUY 7 times, HOLD 9 times, "
        "SELL 4 times, with a win rate of 60.0 percent on daily returns. Reflect on "
        "whether your positioning matched the prevailing trend and volatility, and "
        "whether your risk management was consistent.\n"
        "Feeling log: Feeling cautiously reflective about this period. Use the results "
        "to refine your process without becoming overconfident or discouraged."
    )


def test_fallback_no_date_no_tech(patch_flags):
    patch_flags(show_date=False, use_dummy=True)
    out = ps.generate_llm_period_summary("Month", END, STATS, None, "mtag")
    assert out == (
        "Month summary (date hidden)\n"
        "Explanation: Month summary. Market total return  1.80 percent. "
        "Strategy total return  4.20 percent. The strategy outperformed the index "
        "by 2.40 percent over 20 days.\n"
        "Strategic journal: During this period you traded BUY 7 times, HOLD 9 times, "
        "SELL 4 times, with a win rate of 60.0 percent on daily returns. Reflect on "
        "whether your positioning matched the prevailing trend and volatility, and "
        "whether your risk management was consistent.\n"
        "Feeling log: Feeling cautiously reflective about this period. Use the results "
        "to refine your process without becoming overconfident or discouraged."
    )


def test_fallback_with_date_and_tech(patch_flags):
    patch_flags(show_date=True, use_dummy=True)
    out = ps.generate_llm_period_summary("Quarter", END, STATS, None, "mtag", TECH)
    assert out == (
        "Quarter ending 2022-03-25\n"
        "Explanation: Quarter ending 2022-03-25. Market total return  1.80 percent. "
        "Strategy total return  4.20 percent. The strategy outperformed the index "
        "by 2.40 percent over 20 days. RSI averaged 47.5 with 10.0% overbought days. "
        "MACD was bullish 60.0% of the time. Stochastic showed 25.0% overbought "
        "conditions. Price touched Bollinger upper band on 6.0% of days.\n"
        "Strategic journal: During this period you traded BUY 7 times, HOLD 9 times, "
        "SELL 4 times, with a win rate of 60.0 percent on daily returns. Technical "
        "indicators provided mixed signals: RSI bullish signals, MACD showing mostly "
        "bullish momentum, Stochastic with frequent overbought conditions, Bollinger "
        "Bands with frequent band touches. Reflect on whether your positioning matched "
        "the prevailing trend and volatility, and whether your risk management was "
        "consistent.\n"
        "Feeling log: Feeling cautiously reflective about this period. Use the results "
        "to refine your process without becoming overconfident or discouraged."
    )


def test_llm_happy_path_with_date(patch_flags, monkeypatch):
    patch_flags(show_date=True, use_dummy=False)
    monkeypatch.setattr(
        ps, "generate_response", lambda model, sysp, usr: "  MODEL JOURNAL BODY.  "
    )
    out = ps.generate_llm_period_summary("Year", END, STATS, "router/x", "mtag")
    assert out == (
        "Year ending 2022-03-25\n"
        "Stats  strategy 4.20 percent, index 1.80 percent, edge (strategy minus index) "
        "2.40 percent, days 20, wins 12, BUY 7, HOLD 9, SELL 4.\n\n"
        "MODEL JOURNAL BODY."
    )


def test_llm_prompt_with_date_no_tech(patch_flags, monkeypatch):
    """Pins the exact user message handed to generate_response (date on, no tech)."""
    patch_flags(show_date=True, use_dummy=False)
    captured = {}

    def spy(model, sysp, usr):
        captured["model"] = model
        captured["usr"] = usr
        return "BODY"

    monkeypatch.setattr(ps, "generate_response", spy)
    ps.generate_llm_period_summary("Year", END, STATS, "router/x", "mtag")
    assert captured["model"] == "router/x"
    assert captured["usr"] == (
        "You are summarizing a completed Year.\n\n"
        "Period information\n"
        "- End date  2022-03-25\n"
        "- Trading days in period  20\n"
        "- Strategy total return over the period  4.20 percent\n"
        "- Index total return over the period  1.80 percent\n"
        "- Difference strategy minus index  2.40 percent\n"
        "- Winning days (positive strategy return)  12 out of 20\n"
        "- Number of BUY decisions  7\n"
        "- Number of HOLD decisions  9\n"
        "- Number of SELL decisions  4\n"
        "- Daily win rate  60.0 percent\n\n"
        "Write a reflection journal for this period. Do not include any dates or "
        "calendar references. Use only the numerical information provided."
    )


def test_llm_prompt_with_tech_and_date(patch_flags, monkeypatch):
    patch_flags(show_date=True, use_dummy=False)
    captured = {}
    monkeypatch.setattr(
        ps, "generate_response", lambda m, s, u: captured.__setitem__("u", u) or "BODY"
    )
    ps.generate_llm_period_summary("Quarter", END, STATS, "router/x", "mtag", TECH)
    assert captured["u"] == (
        "You are summarizing a completed Quarter.\n\n"
        "Period information\n"
        "- End date  2022-03-25\n"
        "- Trading days in period  20\n"
        "- Strategy total return over the period  4.20 percent\n"
        "- Index total return over the period  1.80 percent\n"
        "- Difference strategy minus index  2.40 percent\n"
        "- Winning days (positive strategy return)  12 out of 20\n"
        "- Number of BUY decisions  7\n"
        "- Number of HOLD decisions  9\n"
        "- Number of SELL decisions  4\n"
        "- Daily win rate  60.0 percent\n"
        "Technical indicators summary for this period:\n"
        "- RSI(14): Average 47.5, 10.0% overbought days (>70), 5.0% oversold days "
        "(<30), range 22.0-78.0\n"
        "- MACD(12,26,9): 60.0% bullish periods, avg histogram 0.123, 3 signal "
        "crossovers\n"
        "- Stochastic(14,3): 25.0% overbought days (>80), 8.0% oversold days (<20), "
        "range 12.0-90.0\n"
        "- Bollinger Bands(20,2): 6.0% days touched upper band, 7.0% touched lower "
        "band, avg position 0.55\n\n\n"
        "Write a reflection journal for this period. Do not include any dates or "
        "calendar references. Use only the numerical information provided."
    )


def test_llm_prompt_no_tech_date_hidden(patch_flags, monkeypatch):
    patch_flags(show_date=False, use_dummy=False)
    captured = {}
    monkeypatch.setattr(
        ps, "generate_response", lambda m, s, u: captured.__setitem__("u", u) or "BODY"
    )
    ps.generate_llm_period_summary("Quarter", END, STATS, "router/x", "mtag")
    assert captured["u"] == (
        "You are summarizing a completed Quarter (time period anonymized).\n\n"
        "Period information\n"
        "- Trading days in period  20\n"
        "- Strategy total return over the period  4.20 percent\n"
        "- Index total return over the period  1.80 percent\n"
        "- Difference strategy minus index  2.40 percent\n"
        "- Winning days (positive strategy return)  12 out of 20\n"
        "- Number of BUY decisions  7\n"
        "- Number of HOLD decisions  9\n"
        "- Number of SELL decisions  4\n"
        "- Daily win rate  60.0 percent\n\n"
        "Write a reflection journal for this period. Do not include any dates or "
        "calendar references. Use only the numerical information provided."
    )


def test_llm_error_fallback_with_date(patch_flags, monkeypatch):
    patch_flags(show_date=True, use_dummy=False)

    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(ps, "generate_response", boom)
    out = ps.generate_llm_period_summary("Week", END, STATS, "router/x", "mtag")
    assert out == (
        "Week ending 2022-03-25\n"
        "Explanation: Week ending 2022-03-25. Market total return  1.80 percent. "
        "Strategy total return  4.20 percent over 20 days.\n"
        "Strategic journal: BUY 7, HOLD 9, SELL 4, win rate 60.0 percent. LLM journal "
        "generation failed, using fallback summary.\n"
        "Feeling log: Feeling neutral due to technical issues."
    )


def test_llm_error_fallback_date_hidden(patch_flags, monkeypatch):
    patch_flags(show_date=False, use_dummy=False)

    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(ps, "generate_response", boom)
    out = ps.generate_llm_period_summary("Week", END, STATS, "router/x", "mtag")
    assert out == (
        "Week summary (date hidden)\n"
        "Explanation: Week summary. Market total return  1.80 percent. "
        "Strategy total return  4.20 percent over 20 days.\n"
        "Strategic journal: BUY 7, HOLD 9, SELL 4, win rate 60.0 percent. LLM journal "
        "generation failed, using fallback summary.\n"
        "Feeling log: Feeling neutral due to technical issues."
    )
