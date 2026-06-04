# =============================================================================
# 🤖 LLM FINANCE EXPERIMENT CONFIGURATION
# =============================================================================
# This file contains all settings for the LLM trading strategy experiments.

import logging
import os  # For file existence checks
import sys  # For CLI arg checks during validation

import pandas as pd  # For date calculations

# Validation runs at import (before the app configures logging), so warnings go
# through logging.warning -- visible by default via the last-resort handler and
# capturable once logging is set up. Informational startup banners stay on
# print() because logging.info would be suppressed at import time.
logger = logging.getLogger(__name__)

#
# 🚀 QUICK START GUIDE:
#   1. Choose your data source: DATA_SOURCE = "vendored" (default, offline) or "csv" or "stooq"
#   2. Pick how decisions are generated: LLM_PROVIDER = "dummy" (free, no API, for testing)
#   3. For real experiments: LLM_PROVIDER = "openrouter" (+ OPENROUTER_API_KEY) or "claude_code"
#   4. Choose experiment type with ACTIVE_EXPERIMENT
#   5. Choose trader personality with ACTIVE_PERSONALITY
#   6. Enable chain of thought reasoning with ENABLE_CHAIN_OF_THOUGHT = True (works with any experiment)
#   7. Set TEST_MODE = False for full analysis of the data range or True for testing
#
# 📋 COMMON USE CASES (USE_DUMMY_MODEL is derived from LLM_PROVIDER, never set it by hand):
#   • First time user: DATA_SOURCE = "vendored", LLM_PROVIDER = "dummy", ACTIVE_EXPERIMENT = "baseline", ACTIVE_PERSONALITY = "cautious"
#   • Memory testing: DATA_SOURCE = "vendored", LLM_PROVIDER = "openrouter", ACTIVE_EXPERIMENT = "memory_feeling", ACTIVE_PERSONALITY = "balanced"
#   • Chain of thought: DATA_SOURCE = "vendored", LLM_PROVIDER = "claude_code", ACTIVE_EXPERIMENT = "memory_feeling", ENABLE_CHAIN_OF_THOUGHT = True, ACTIVE_PERSONALITY = "balanced"
#   • Full research: DATA_SOURCE = "vendored", LLM_PROVIDER = "openrouter", TEST_MODE = False, ACTIVE_PERSONALITY = "aggressive"
# =============================================================================

# =============================================================================
# 🔑 ESSENTIAL USER CONFIGURATION - START HERE!
# =============================================================================

# MODEL / PROVIDER SELECTION
# --------------------------
# Pick how trading decisions are generated. This is the single knob to switch
# between providers.
#
#   "dummy"                 -> random decisions, no API/CLI, no cost. For testing.
#   "openrouter"            -> any model via OpenRouter HTTP API.
#                              Needs OPENROUTER_API_KEY. Pay per token.
#                              Best for large backtests (parallel, model variety).
#   "claude_code"           -> Claude Code via your SUBSCRIPTION (no per-token cost),
#                              single-shot. Needs the claude CLI logged in on this
#                              machine. Best for small / high-quality experiments.
#   "claude_code_subagents" -> Claude Code subscription, multi-agent: a lead PM
#                              consults specialist analysts (see ANALYST_AGENTS)
#                              before deciding. Richer, slower, costs more.
#
# Note on intended use: the Claude Code providers use the official headless CLI
# against your subscription. Keep them for interactive-scale research (short
# windows, few tickers). For high-volume automated backtests, use "openrouter".
LLM_PROVIDER = "dummy"

# Claude model used by the claude_code providers ("sonnet", "opus", "haiku",
# or a full model id like "claude-sonnet-4-6").
CLAUDE_CODE_MODEL = "sonnet"

# Backward compatibility: legacy code and tests still read USE_DUMMY_MODEL.
# It is now derived from LLM_PROVIDER (do not set it directly).
USE_DUMMY_MODEL = LLM_PROVIDER == "dummy"

# Specialist analysts for the "claude_code_subagents" provider. Each entry is a
# subagent the lead PM can consult via the Task tool. Edit freely (add macro,
# sentiment, ...). They reason only from the prompt data; web/tools are disabled.
ANALYST_AGENTS = {
    "technical_analyst": {
        "description": "Reads price action, moving averages, RSI, MACD. Use for the technical picture.",
        "prompt": (
            "You are a technical analyst. From the market data given, assess trend, "
            "momentum, and overbought/oversold conditions, then state a clear bias: "
            "bullish, neutral, or bearish. Be concise (3 sentences max). Reason only "
            "from the data provided; do not look anything up."
        ),
    },
    "risk_analyst": {
        "description": "Assesses volatility regime and downside risk. Use for the risk picture.",
        "prompt": (
            "You are a risk analyst. From the market data given, assess the "
            "volatility regime and downside risk, then state a clear posture: "
            "risk-on, neutral, or risk-off. Be concise (3 sentences max). Reason "
            "only from the data provided; do not look anything up."
        ),
    },
}

# EXPERIMENT TYPE
# ---------------
ACTIVE_EXPERIMENT = "memory_feeling"  # Choose experiment configuration:
# - "baseline": No memory, no feeling, no dates
# - "memory_only": Memory/journal enabled
# - "memory_feeling": Memory + emotional tracking
# - "dates_only": Calendar dates shown
# - "dates_memory": Dates + memory
# - "dates_full": Full context (dates + memory + feeling)
# Note: Chain of thought reasoning is controlled by ENABLE_CHAIN_OF_THOUGHT toggle (independent of experiment)

# TRADER PERSONALITY
# ------------------
ACTIVE_PERSONALITY = "balanced"  # Choose trader personality:
# - "cautious": Conservative, risk-averse trading
# - "aggressive": Bold, opportunity-focused trading
# - "balanced": Systematic, balanced approach
# - "momentum": Trend-following strategies
# - "contrarian": Counter-trend strategies

# CHAIN OF THOUGHT REASONING
# --------------------------
ENABLE_CHAIN_OF_THOUGHT = True  # Enable structured analytical reasoning in prompts

# =================================================================================
# 🎯  DATA SOURCE SELECTION (Important Choice!)
# =================================================================================
DATA_SOURCE = (
    "vendored"  # "vendored" (default: committed frozen snapshot, offline + deterministic)
    # Other options: "stooq" (live fetch, needs apikey) or "csv" (your own local file)
)

# DATE RANGE (applies to both data sources)
# -----------------------------------------
# ⚠️  IMPORTANT: These dates set your RAW data range, but trading starts LATER!
#
# Due to technical indicator requirements, the system automatically removes
# the first ~40 trading days where indicators are incomplete (contain NaN values).
# This creates an automatic "warm-up buffer" before trading begins.
#
# Examples with DATA_START = "2015-01-01":
#   Raw data starts: 2015-01-01 (first day of data)
#   Trading actually starts: ~2015-03-01 (after ~40 trading days of buffer)
#   Why? RSI needs 14+ days, MACD needs 35+ days, Bollinger Bands need 20+ days
#
# To start trading earlier, you could:
#   Option A: Set DATA_START earlier (e.g., "2014-01-01")
#   Option B: Use START_ROW to skip additional days beyond the buffer
#   Option C: Accept that ~40-day buffer is required for reliable indicators
#
DATA_START = "2015-03-03"  # Raw data start (not actual trading start!)
DATA_END = "2023-12-31"  # Raw data end

# =================================================================================
# 📈  STOOQ HISTORICAL DATA (live refresh option, needs an apikey)
# =================================================================================
# Symbol for Stooq API (only used when DATA_SOURCE = "stooq")
STOOQ_SYMBOL = "SPY"  # SPY, QQQ, AAPL, MSFT, GOOGL, BTC-USD

# =================================================================================
# 📁  CSV DATA FILES (your own local dataset)
# =================================================================================
# Symbol for CSV data (only used when DATA_SOURCE = "csv")
SYMBOL = "^GSPC"  # Stock symbol in CSV file
CSV_DATA_PATH = "data/raw/sp500.csv"  # Path to your CSV file

# =================================================================================
# 📦  VENDORED SNAPSHOT (Default - frozen, offline, checksummed)
# =================================================================================
# Committed real SPY data so a clean clone runs offline and deterministically.
# Verified against MANIFEST.json on load. See data/raw/PROVENANCE.md.
VENDORED_DATA_PATH = "data/raw/spy_daily.csv"
VENDORED_MANIFEST_PATH = "data/raw/MANIFEST.json"

# =================================================================================
# 🧪  TEST vs FULL RUN
# =================================================================================
TEST_MODE = True  # Set to True for quick tests, False for full experiments
TEST_LIMIT = 5  # Number of days to run when TEST_MODE = True (test on at least 3 days)
# Set TEST_MODE = False for complete ~2700 day analysis

# =================================================================================
# 🎲  REPRODUCIBILITY
# =================================================================================
# Seed for all stochastic components (the dummy model, statistical bootstraps).
# Seeded once at the start of each run so a backtest replays identically. It is
# recorded in results/RUN_MANIFEST.json alongside data checksum and dep versions.
# See docs/REPRODUCIBILITY.md.
RANDOM_SEED = 42


# =================================================================================
# 🎛️  EXPERIMENT FEATURES - EASY TOGGLES
# =================================================================================

# TECHNICAL ANALYSIS
# ------------------
# Master toggle for technical indicators in LLM prompts
# When enabled: LLM sees RSI + MACD + Stochastic + Bollinger Bands
# When disabled: LLM sees no indicator values
# Note: Indicators are always calculated for baselines and analysis
ENABLE_TECHNICAL_INDICATORS = True

# TRADING HISTORY CONTEXT
# -----------------------
# Include complete historical trading performance in LLM prompts
# Enable: Better long-term learning and pattern recognition
# Disable: Token efficiency and reduced context length
ENABLE_FULL_TRADING_HISTORY = True

# DEBUGGING & DEVELOPMENT
# -----------------------
# Show full LLM prompts during execution (helpful for understanding what the AI sees)
DEBUG_SHOW_FULL_PROMPT = True

# DATA SUBSET FOR TESTING
# -----------------------
# Start from a specific row in the PROCESSED dataset (after automatic cleaning)
# ⚠️  NOTE: Even START_ROW = 0 starts ~40 days after DATA_START!
#
# The system automatically removes ~40 trading days due to technical indicator
# warm-up requirements. START_ROW is applied AFTER this automatic cleaning.
#
# Examples (assuming DATA_START = "2015-01-01"):
# START_ROW = 0  → Trading starts ~2015-03-01 (after automatic 40-day buffer)
# START_ROW = 30 → Trading starts ~2015-04-15 (buffer + 30 additional days)
# START_ROW = 333 → Start mid-dataset for testing different market conditions
#
# WARNING: Effective start = DATA_START + ~40 trading days + START_ROW
START_ROW = 0

# LLM MODELS TO TEST
# ------------------
# Configure which LLM models to benchmark via OpenRouter API
# Uncomment/comment models as needed. Requires USE_DUMMY_MODEL = False
LLM_MODELS = [
    # {
    #    "tag": "bert",  # Model no longer available
    #    "router_model": "openrouter/bert-nebulon-alpha",
    # },
    # {
    #    "tag": "chimera",  # Model identifier (used in filenames)
    #    "router_model": "tngtech/tng-r1t-chimera:free",  # OpenRouter model ID
    # },
    # Add more models here:
    # {
    #    "tag": "olmo-32b",
    #    "router_model": "allenai/olmo-3-32b-think",
    # },
    # {
    #    "tag": "gpt-oss-120b",
    #    "router_model": "openai/gpt-oss-120b:free",
    # },
    # {
    #    "tag": "gpt-oss-20b",
    #    "router_model": "openai/gpt-oss-20b:free",
    # },
    {
        "tag": "deepseek-r1-0528",
        "router_model": "deepseek/deepseek-r1-0528:free",
    },
    # {
    #     "tag": "claude",
    #     "router_model": "anthropic/claude-3-sonnet",
    # },
]

# TECHNICAL PARAMETERS
# --------------------
# Advanced settings for data processing and trading features
PAST_RET_LAGS = 20  # Number of past return lags for features
RET_5D_WINDOW = 5  # 5-day return window (configurable via config.py)
MA20_WINDOW = 20  # 20-day moving average window (configurable via config.py)
VOL20_WINDOW = 20  # 20-day volatility window (configurable via config.py)
RSI_WINDOW = 14  # RSI period (14 days standard)
RSI_OVERBOUGHT = 70  # RSI overbought threshold
RSI_OVERSOLD = 30  # RSI oversold threshold

# Advanced Technical Indicators
# ----------------------------
MACD_FAST = 12  # Fast EMA period for MACD
MACD_SLOW = 26  # Slow EMA period for MACD
MACD_SIGNAL = 9  # Signal line EMA period for MACD
STOCH_K = 14  # %K period for Stochastic Oscillator
STOCH_D = 3  # %D smoothing period for Stochastic Oscillator
STOCH_SMOOTH_K = 3  # %K smoothing period (optional)
BB_WINDOW = 20  # Bollinger Bands period
BB_STD = 2  # Standard deviations for Bollinger Bands
STOCH_OVERBOUGHT = 80  # Stochastic overbought threshold
STOCH_OVERSOLD = 20  # Stochastic oversold threshold

# Note: Technical indicators (RSI, MACD, Stochastic, Bollinger Bands)
# are always calculated for baselines and analysis, regardless of the toggle above.
# The toggle above only controls whether they appear in LLM prompts.

# =============================================================================
# ✅ END OF ESSENTIAL CONFIGURATION
# =============================================================================
# You probably don't need to change anything below this line!
# Advanced users can modify experiment configurations and prompts below.
# =============================================================================

# =============================================================================
# 🔧 EXPERT CONFIGURATION (rarely need to change these)
# =============================================================================

# =============================================================================
# 🎯 EXPERIMENT CONFIGURATIONS
# =============================================================================
# Pre-configured experiment setups to test different aspects of LLM trading:
#
# NO DATES (anonymized time series - prevents historical pattern overfitting):
#   • baseline        - Minimal context: no memory, no feeling, no dates
#   • memory_only     - Strategic journal: learns from past decisions
#   • memory_feeling  - Memory + emotional tracking for self-reflection
#
# WITH DATES (real calendar dates - tests for data leakage):
#   • dates_only      - Calendar awareness: dates shown, no memory
#   • dates_memory    - Full learning: dates + strategic adaptation
#   • dates_full      - Complete context: dates + memory + emotional state
#


EXPERIMENT_CONFIGS = {
    # --- NO DATES (anonymized) ---
    "baseline": {
        "description": "Minimal context: no dates, no memory, no feeling",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": False,
        "ENABLE_FEELING_LOG": False,
    },
    "memory_only": {
        "description": "Memory/journal only: no dates, no feeling",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": False,
    },
    "memory_feeling": {
        "description": "Memory + feeling: no dates",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": True,
    },
    # --- WITH DATES (real calendar) ---
    "dates_only": {
        "description": "Dates only: no memory, no feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": False,
        "ENABLE_FEELING_LOG": False,
    },
    "dates_memory": {
        "description": "Dates + memory: no feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": False,
    },
    "dates_full": {
        "description": "Full context: dates + memory + feeling",
        "SHOW_DATE_TO_LLM": True,
        "ENABLE_STRATEGIC_JOURNAL": True,
        "ENABLE_FEELING_LOG": True,
    },
    # --- CHAIN OF THOUGHT EXPERIMENTS ---
    "chain_of_thought": {
        "description": "Chain of thought reasoning: structured analytical steps (deprecated preset - use ENABLE_CHAIN_OF_THOUGHT toggle instead)",
        "SHOW_DATE_TO_LLM": False,
        "ENABLE_STRATEGIC_JOURNAL": False,
        "ENABLE_FEELING_LOG": False,
    },
}

# (ACTIVE_EXPERIMENT moved to top of file for easy access)

# MANUAL EXPERIMENT SETTINGS
# -------------------------
# Only used if ACTIVE_EXPERIMENT = None above
# Allows custom configuration not covered by preset experiments
_MANUAL_SHOW_DATE_TO_LLM = False
_MANUAL_ENABLE_STRATEGIC_JOURNAL = False
_MANUAL_ENABLE_FEELING_LOG = True
_MANUAL_ENABLE_CHAIN_OF_THOUGHT = False
_MANUAL_ENABLE_TECHNICAL_INDICATORS = False

# Apply experiment config or use manual settings
if ACTIVE_EXPERIMENT and ACTIVE_EXPERIMENT in EXPERIMENT_CONFIGS:
    _config = EXPERIMENT_CONFIGS[ACTIVE_EXPERIMENT]
    SHOW_DATE_TO_LLM = _config["SHOW_DATE_TO_LLM"]
    ENABLE_STRATEGIC_JOURNAL = _config["ENABLE_STRATEGIC_JOURNAL"]
    ENABLE_FEELING_LOG = _config["ENABLE_FEELING_LOG"]
    # ENABLE_CHAIN_OF_THOUGHT is now a master toggle - don't override it from experiments
    print(f"[CONFIG] Active experiment: {ACTIVE_EXPERIMENT}")
    print(f"         {_config['description']}")
    print(
        f"         dates={SHOW_DATE_TO_LLM}, memory={ENABLE_STRATEGIC_JOURNAL}, "
        f"feeling={ENABLE_FEELING_LOG}, chain_of_thought={ENABLE_CHAIN_OF_THOUGHT} (master toggle), technical={ENABLE_TECHNICAL_INDICATORS}"
    )
else:
    # Use manual settings
    SHOW_DATE_TO_LLM = _MANUAL_SHOW_DATE_TO_LLM
    ENABLE_STRATEGIC_JOURNAL = _MANUAL_ENABLE_STRATEGIC_JOURNAL
    ENABLE_FEELING_LOG = _MANUAL_ENABLE_FEELING_LOG
    ENABLE_CHAIN_OF_THOUGHT = _MANUAL_ENABLE_CHAIN_OF_THOUGHT
    ENABLE_TECHNICAL_INDICATORS = _MANUAL_ENABLE_TECHNICAL_INDICATORS
    if ACTIVE_EXPERIMENT:
        logger.warning(
            "Unknown experiment '%s', using manual settings", ACTIVE_EXPERIMENT
        )


# (USE_DUMMY_MODEL moved to top of file)
# (TEST_MODE and TEST_LIMIT moved to top of file)

# START_ROW validation - check against dataset size
if START_ROW is not None:
    estimated_dataset_days = 2700  # Approximate full dataset size
    if START_ROW >= estimated_dataset_days:
        logger.warning(
            "START_ROW (%s) >= estimated dataset size (%s); this may cause no data "
            "to be processed in full experiments",
            START_ROW,
            estimated_dataset_days,
        )

    # Additional check for TEST_MODE
    if TEST_MODE and START_ROW + TEST_LIMIT > estimated_dataset_days:
        logger.warning(
            "START_ROW (%s) + TEST_LIMIT (%s) > estimated dataset size; test may "
            "not have enough data",
            START_ROW,
            TEST_LIMIT,
        )

    # Warning about automatic buffer even with START_ROW = 0
    if START_ROW == 0:
        print(
            "NOTE: START_ROW = 0 will still skip ~40 days due to technical indicator requirements"
        )
        print(
            "    Effective trading start will be approximately 40 trading days after DATA_START"
        )
        print(
            f"    With DATA_START = '{DATA_START}', expect trading to begin around {pd.to_datetime(DATA_START) + pd.Timedelta(days=40):%Y-%m-%d}"
        )

# Decision mapping
# BUY  +1, HOLD 0, SELL -1
POSITION_MAP = {
    "BUY": 1.0,
    "HOLD": 0.0,
    "SELL": -1.0,
}


# Create a single ConfigurationManager instance seeded with this module's live values.
from .configuration_manager import ConfigurationManager

_config_manager = ConfigurationManager(config_values=globals().copy())


def get_current_symbol_info():
    """
    Get the current symbol and user-friendly name based on data source.

    Returns:
        tuple: (symbol_code, user_friendly_name)
        Examples:
        - ("SPY", "SPY ETF (S&P 500 tracker)")
        - ("^GSPC", "S&P 500 Index")
        - ("AAPL", "Apple Inc. stock")
    """
    if DATA_SOURCE == "stooq":
        symbol = STOOQ_SYMBOL
        # Map common symbols to user-friendly names
        symbol_names = {
            "SPY": "SPY ETF (S&P 500 tracker)",
            "QQQ": "QQQ ETF (Nasdaq 100 tracker)",
            "AAPL": "Apple Inc. stock",
            "MSFT": "Microsoft Corp. stock",
            "GOOGL": "Alphabet Inc. stock",
            "BTC-USD": "Bitcoin USD",
        }
        name = symbol_names.get(symbol, f"{symbol} (via Stooq)")
    else:  # CSV data source
        symbol = SYMBOL
        name = "S&P 500 Index" if symbol == "^GSPC" else f"Custom index ({symbol})"

    return symbol, name


def validate_symbol_config():
    """
    Validate that symbol configuration is consistent and provide warnings.
    Called during config validation.
    """
    symbol, name = get_current_symbol_info()

    # Log current configuration for debugging
    print(f"[CONFIG] 📊 Symbol: {symbol}")
    print(f"[CONFIG] 🏷️  Display name: {name}")

    # Warn about potential confusion
    if DATA_SOURCE == "stooq" and symbol == "SPY":
        print(
            "[CONFIG] ℹ️  Using SPY as S&P 500 proxy - ensure this matches your research intent"
        )
    elif DATA_SOURCE == "csv" and symbol != "^GSPC":
        logger.warning(
            "Using custom symbol '%s' - verify data file contains this symbol", symbol
        )


# Make SYSTEM_PROMPT a function that returns the current prompt
def SYSTEM_PROMPT():
    """Get the current system prompt based on active personality."""
    from .prompt_builder import PromptBuilder

    return PromptBuilder(_config_manager).build_system_prompt()


def _build_journal_system_prompt():
    # Get current personality directly from ACTIVE_PERSONALITY setting
    personality_name = ACTIVE_PERSONALITY
    _personality = _config_manager._config.personality.personalities.get(
        personality_name
    )
    if not _personality:
        # Fallback to cautious if personality not found
        _personality = _config_manager._config.personality.personalities["cautious"]

    _, symbol_name = get_current_symbol_info()
    base_prompt = f"""You are a {_personality.description}. Your role is to trade the {symbol_name}.

Instead of deciding a trading action, your task now is to write a reflection journal for a completed period
(one Week, Month, Quarter, or Year) based only on the numerical information provided in the user message.

The user message will give you:
- the type of period
- the end date of the period
- the number of trading days
- the strategy total return for that period
- the index total return for that period
- the number of winning days
- the number of BUY, HOLD, and SELL decisions"""

    # Add technical indicators info when enabled
    if ENABLE_TECHNICAL_INDICATORS:
        base_prompt += """
- Technical indicators summary for this period (averages and key statistics)"""

    base_prompt += """

Write your answer in"""

    if ENABLE_FEELING_LOG:
        sections = """ three clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was."""

        if ENABLE_TECHNICAL_INDICATORS:
            sections += """
Analyze how technical indicators behaved during this period."""

        sections += """

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period."""

        if ENABLE_TECHNICAL_INDICATORS:
            sections += """
Consider whether technical indicators provided useful signals or conflicting information."""

        sections += """
Mention any biases, patterns, or adjustments that you should consider for future periods.

Feeling log:
Describe how you "feel" about this period (for example confident, cautious, frustrated, relieved),
linking these feelings to the performance and the quality of your decisions."""

        if ENABLE_TECHNICAL_INDICATORS:
            sections += """
Consider how technical indicator behavior influenced your emotional state."""

    else:
        sections = """ two clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was."""

        if ENABLE_TECHNICAL_INDICATORS:
            sections += """
Analyze how technical indicators behaved during this period."""

        sections += """

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period."""

        if ENABLE_TECHNICAL_INDICATORS:
            sections += """
Consider whether technical indicators provided useful signals or conflicting information."""

        sections += """
Mention any biases, patterns, or adjustments that you should consider for future periods."""

    # Add technical indicator interpretation guidelines when enabled
    if ENABLE_TECHNICAL_INDICATORS:
        sections += """

When analyzing technical indicators in your reflection:
- RSI(14): Values >70 indicate overbought conditions, <30 indicate oversold conditions
- MACD: Positive histogram suggests bullish momentum, negative suggests bearish
- Stochastic(14,3): Values >80 indicate overbought, <20 indicate oversold
- Bollinger Bands: Price near upper band suggests potential reversal, near lower band suggests potential bounce

Consider how these indicators performed during the period and whether they aligned with your trading decisions."""

    closing = """

Do not output any trading actions such as BUY, HOLD, or SELL as commands.
Do not talk about future periods as if you know the outcomes.
Do not include disclaimers or meta commentary."""

    return (base_prompt + sections + closing).strip()


# Make JOURNAL_SYSTEM_PROMPT a function that returns the current prompt
def JOURNAL_SYSTEM_PROMPT():
    """Get the current journal system prompt based on active personality."""
    return _build_journal_system_prompt()


# (LLM_MODELS moved to top of file)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"


# =============================================================================
# EXPERIMENT HELPER FUNCTIONS
# =============================================================================


def list_experiments():
    """Print all available experiment configurations."""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENT CONFIGURATIONS")
    print("=" * 70)
    print(f"\n{'Name':<20} {'Dates':<8} {'Memory':<8} {'Feeling':<8} Description")
    print("-" * 70)
    for name, cfg in EXPERIMENT_CONFIGS.items():
        dates = "Yes" if cfg["SHOW_DATE_TO_LLM"] else "No"
        memory = "Yes" if cfg["ENABLE_STRATEGIC_JOURNAL"] else "No"
        feeling = "Yes" if cfg["ENABLE_FEELING_LOG"] else "No"
        cot = "Yes" if cfg.get("ENABLE_CHAIN_OF_THOUGHT", False) else "No"
        desc = (
            cfg["description"][:30] + "..."
            if len(cfg["description"]) > 30
            else cfg["description"]
        )
        marker = " ◄ ACTIVE" if name == ACTIVE_EXPERIMENT else ""
        print(f"{name:<20} {dates:<8} {memory:<8} {feeling:<8} {cot:<8} {desc}{marker}")
    print("=" * 70 + "\n")


def get_experiment_suffix():
    """
    Return a suffix for model tags based on active experiment.
    Example: 'grok_fast' becomes 'grok_fast_baseline' or 'grok_fast_dates_full'
    """
    if ACTIVE_EXPERIMENT and ACTIVE_EXPERIMENT in EXPERIMENT_CONFIGS:
        return f"_{ACTIVE_EXPERIMENT}"
    else:
        # Build suffix from manual settings
        parts = []
        if SHOW_DATE_TO_LLM:
            parts.append("dates")
        if ENABLE_STRATEGIC_JOURNAL:
            parts.append("mem")
        if ENABLE_FEELING_LOG:
            parts.append("feel")
        if ENABLE_CHAIN_OF_THOUGHT:
            parts.append("cot")
        if not parts:
            parts.append("minimal")
        return "_" + "_".join(parts)


def get_current_config_summary():
    """Return a dict summarizing current experiment settings."""
    return {
        "experiment": ACTIVE_EXPERIMENT or "manual",
        "show_dates": SHOW_DATE_TO_LLM,
        "strategic_journal": ENABLE_STRATEGIC_JOURNAL,
        "feeling_log": ENABLE_FEELING_LOG,
        "chain_of_thought": ENABLE_CHAIN_OF_THOUGHT,
        "description": EXPERIMENT_CONFIGS.get(ACTIVE_EXPERIMENT, {}).get(
            "description", "Manual configuration"
        ),
    }


# Data source validation
VALID_DATA_SOURCES = ["vendored", "csv", "stooq"]


def validate_data_source_config():
    """Validate configuration and provide clear user feedback."""
    if DATA_SOURCE not in VALID_DATA_SOURCES:
        raise ValueError(
            f"Invalid DATA_SOURCE '{DATA_SOURCE}'. Must be one of: {VALID_DATA_SOURCES}"
        )

    if DATA_SOURCE == "vendored":
        if not os.path.exists(VENDORED_DATA_PATH):
            logger.warning("Vendored dataset not found: %s", VENDORED_DATA_PATH)
        if not os.path.exists(VENDORED_MANIFEST_PATH):
            logger.warning("Vendored manifest not found: %s", VENDORED_MANIFEST_PATH)
        print("[CONFIG] [OK] Primary: vendored frozen snapshot (offline, checksummed)")
        print(f"[CONFIG] 📦 File: {VENDORED_DATA_PATH}")
        print("[CONFIG] 🔄 Fallback: None (vendored is offline-by-design)")

    elif DATA_SOURCE == "stooq":
        # Validate Stooq settings
        if not STOOQ_SYMBOL or not isinstance(STOOQ_SYMBOL, str):
            raise ValueError("STOOQ_SYMBOL must be a non-empty string")

        # Only print when not in math validation subprocess
        is_math_validation = "validate_core_math" in str(sys.argv)

        if not is_math_validation:
            # Warn about ignored settings
            if SYMBOL != "^GSPC":  # Check if user changed from default
                logger.warning(
                    "SYMBOL='%s' ignored when using Stooq. Using STOOQ_SYMBOL='%s' instead.",
                    SYMBOL,
                    STOOQ_SYMBOL,
                )

            # Check for old variable name
            if hasattr(__import__("src.config"), "CSV_FALLBACK_PATH"):
                print(
                    "[INFO] CSV_FALLBACK_PATH variable detected. Consider using CSV_DATA_PATH instead."
                )

            print("[CONFIG] [OK] Primary: Stooq historical data")
            print(
                f"[CONFIG] [DATA] Symbol: {STOOQ_SYMBOL} (historical: {DATA_START} to {DATA_END})"
            )
            print(
                f"[CONFIG] [FALLBACK] CSV file at {getattr(__import__('src.config'), 'CSV_DATA_PATH', getattr(__import__('src.config'), 'CSV_FALLBACK_PATH', 'data/raw/sp500.csv'))}"
            )

    elif DATA_SOURCE == "csv":
        # Validate CSV settings
        csv_path = getattr(
            __import__("src.config"),
            "CSV_DATA_PATH",
            getattr(
                __import__("src.config"), "CSV_FALLBACK_PATH", "data/raw/sp500.csv"
            ),
        )
        if not os.path.exists(csv_path):
            logger.warning("CSV file not found: %s", csv_path)

        print("[CONFIG] 📁 Primary: CSV file data")
        print(f"[CONFIG] 📄 File: {csv_path}")
        print("[CONFIG] 🔄 Fallback: None (CSV is primary)")

    # Always show date range (but not during math validation)
    if not "validate_core_math" in str(sys.argv):
        print(f"[CONFIG] [DATES] Date range: {DATA_START} to {DATA_END}")
        print(
            f"[CONFIG] 🧪 Test mode: {'ON' if TEST_MODE else 'OFF'} ({TEST_LIMIT if TEST_MODE else 'full'} days)"
        )

    # Validate symbol configuration
    if not "validate_core_math" in str(sys.argv):
        validate_symbol_config()


# Call validation during config load
validate_data_source_config()
