# =============================================================================
# 🤖 LLM FINANCE EXPERIMENT CONFIGURATION
# =============================================================================
# This file contains all settings for the LLM trading strategy experiments.
#
# 🚀 QUICK START GUIDE:
#   1. For testing: Set USE_DUMMY_MODEL = True, keep other defaults
#   2. For real experiments: Set USE_DUMMY_MODEL = False, add OpenRouter API key
#   3. Choose experiment type with ACTIVE_EXPERIMENT
#   4. Set TEST_MODE = False for full analysis (takes ~30 minutes)
#
# 📋 COMMON USE CASES:
#   • First time user: USE_DUMMY_MODEL = True, ACTIVE_EXPERIMENT = "baseline"
#   • Memory testing: USE_DUMMY_MODEL = False, ACTIVE_EXPERIMENT = "memory_feeling"
#   • Full research: USE_DUMMY_MODEL = False, TEST_MODE = False
# =============================================================================

# =============================================================================
# 🔑 ESSENTIAL USER CONFIGURATION - START HERE!
# =============================================================================

# MODEL SELECTION
# ---------------
USE_DUMMY_MODEL = True  # Set to True for testing (no API key needed)
# Set to False to run real LLM experiments via OpenRouter

# EXPERIMENT TYPE
# ---------------
ACTIVE_EXPERIMENT = "memory_feeling"  # Choose experiment configuration:
# - "baseline": No memory, no feeling, no dates
# - "memory_only": Memory/journal enabled
# - "memory_feeling": Memory + emotional tracking
# - "dates_only": Calendar dates shown
# - "dates_memory": Dates + memory
# - "dates_full": Full context (dates + memory + feeling)

# DATA SETTINGS
# -------------
SYMBOL = "^GSPC"  # Stock/index symbol (^GSPC = S&P 500)
DATA_START = "2015-01-01"  # Start date for historical data
DATA_END = "2023-12-31"  # End date for historical data

# TEST vs FULL RUN
# ----------------
TEST_MODE = True  # Set to True for quick tests, False for full experiments
TEST_LIMIT = 50  # Number of days to run when TEST_MODE = True
# Set TEST_MODE = False for complete ~2700 day analysis

# =================================================================================
# 🎛️  EXPERIMENT FEATURES - EASY TOGGLES
# =================================================================================

# TECHNICAL ANALYSIS
# ------------------
# Master toggle for technical indicators in LLM prompts
# When enabled: LLM sees RSI + MACD + Stochastic + Bollinger Bands
# When disabled: LLM sees only RSI rules (no indicator values)
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
# Start from a specific row in the dataset (useful for testing specific time periods)
# None = use all available data, 30 = skip first 30 trading days
START_ROW = 30

# LLM MODELS TO TEST
# ------------------
# Configure which LLM models to benchmark via OpenRouter API
# Uncomment/comment models as needed. Requires USE_DUMMY_MODEL = False
LLM_MODELS = [
    #{
    #    "tag": "bert",  # Model identifier (used in filenames)
    #    "router_model": "openrouter/bert-nebulon-alpha",  # OpenRouter model ID
    #},
    #{
    #    "tag": "chimera",  # Model identifier (used in filenames)
    #    "router_model": "tngtech/tng-r1t-chimera:free",  # OpenRouter model ID
    #},
    # Add more models here:
     {
         "tag": "gpt-oss-20b",
         "router_model": "openai/gpt-oss-20b:free",
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
RET_5D_WINDOW = 5  # 5-day return window
MA20_WINDOW = 20  # 20-day moving average window
VOL20_WINDOW = 20  # 20-day volatility window
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
}

# (ACTIVE_EXPERIMENT moved to top of file for easy access)

# MANUAL EXPERIMENT SETTINGS
# -------------------------
# Only used if ACTIVE_EXPERIMENT = None above
# Allows custom configuration not covered by preset experiments
_MANUAL_SHOW_DATE_TO_LLM = False
_MANUAL_ENABLE_STRATEGIC_JOURNAL = False
_MANUAL_ENABLE_FEELING_LOG = True
_MANUAL_ENABLE_TECHNICAL_INDICATORS = False

# Apply experiment config or use manual settings
if ACTIVE_EXPERIMENT and ACTIVE_EXPERIMENT in EXPERIMENT_CONFIGS:
    _config = EXPERIMENT_CONFIGS[ACTIVE_EXPERIMENT]
    SHOW_DATE_TO_LLM = _config["SHOW_DATE_TO_LLM"]
    ENABLE_STRATEGIC_JOURNAL = _config["ENABLE_STRATEGIC_JOURNAL"]
    ENABLE_FEELING_LOG = _config["ENABLE_FEELING_LOG"]
    print(f"[CONFIG] Active experiment: {ACTIVE_EXPERIMENT}")
    print(f"         {_config['description']}")
    print(
        f"         dates={SHOW_DATE_TO_LLM}, memory={ENABLE_STRATEGIC_JOURNAL}, feeling={ENABLE_FEELING_LOG}, technical={ENABLE_TECHNICAL_INDICATORS}"
    )
else:
    # Use manual settings
    SHOW_DATE_TO_LLM = _MANUAL_SHOW_DATE_TO_LLM
    ENABLE_STRATEGIC_JOURNAL = _MANUAL_ENABLE_STRATEGIC_JOURNAL
    ENABLE_FEELING_LOG = _MANUAL_ENABLE_FEELING_LOG
    ENABLE_TECHNICAL_INDICATORS = _MANUAL_ENABLE_TECHNICAL_INDICATORS
    if ACTIVE_EXPERIMENT:
        print(
            f"[CONFIG] Warning: Unknown experiment '{ACTIVE_EXPERIMENT}', using manual settings"
        )



# (USE_DUMMY_MODEL moved to top of file)
# (TEST_MODE and TEST_LIMIT moved to top of file)

# Decision mapping
# BUY  +1, HOLD 0, SELL -1
POSITION_MAP = {
    "BUY": 1.0,
    "HOLD": 0.0,
    "SELL": -1.0,
}


# Build system prompt based on config flags
def _build_system_prompt():
    # Build technical indicators description based on ENABLE_TECHNICAL_INDICATORS
    if ENABLE_TECHNICAL_INDICATORS:
        technical_indicators_desc = """
- 14-day Relative Strength Index (RSI) - momentum oscillator ranging from 0-100
- MACD(12,26,9) - Moving Average Convergence Divergence with histogram
- Stochastic Oscillator(14,3) - momentum indicator ranging from 0-100
- Bollinger Bands(20,2) - volatility bands showing price extremes"""
    else:
        technical_indicators_desc = ""

    base_rules = f"""You are a cautious but rational equity index hedge fund trader. Your role is to beat the S&P500.

Your task is to decide a trading action for the S and P 500 index for the next trading day based only on the information provided in the user message.

Technical indicators available include:
- 20-day moving average momentum (trend strength)
- 20-day annualized volatility (risk measure)
- 5-day recent momentum (short-term trend){technical_indicators_desc}

Rules for decision making:

1) Use only the information in the input. Do not use any knowledge about what happens after the input date."""

    # Build rules based on ENABLE_TECHNICAL_INDICATORS
    if ENABLE_TECHNICAL_INDICATORS:
        base_rules += """
2) RSI measures momentum from 0-100, with >70 overbought and <30 oversold - look for divergences and reversals.
3) MACD crossing above signal line suggests bullish momentum, below suggests bearish - histogram shows momentum strength.
4) Stochastic Oscillator >80 is overbought, <20 is oversold - look for divergences from price action.
5) Bollinger Bands squeeze indicates low volatility (potential breakout), expansion indicates high volatility.
6) Choose exactly one of the following actions:
   BUY  take a long position for the next day
   HOLD stay in cash for the next day, out of the market
   SELL take a short position for the next day
7) Evaluate both expected return and risk. Do not take actions that imply extreme risk seeking.
8) If the information is very unclear, HOLD is acceptable for that day, but you should avoid staying in HOLD for many consecutive days if the recent data shows strong and persistent directional signals."""
    else:
        base_rules += """
2) Choose exactly one of the following actions:
   BUY  take a long position for the next day
   HOLD stay in cash for the next day, out of the market
   SELL take a short position for the next day
3) Evaluate both expected return and risk. Do not take actions that imply extreme risk seeking.
4) If the information is very unclear, HOLD is acceptable for that day, but you should avoid staying in HOLD for many consecutive days if the recent data shows strong and persistent directional signals."""

    # Adjust rule numbering based on technical indicators
    strategic_rule_num = "6)" if ENABLE_TECHNICAL_INDICATORS else "5)"
    objective_rule_num = "7)" if ENABLE_TECHNICAL_INDICATORS else "6)"

    strategic_journal_rule = f"""
{strategic_rule_num} You will also receive a section called "Strategic journal". This contains notes about your past decisions, the outcome of these decisions, and the evolution of your cumulative performance. Use this historical feedback to refine your decision making and improve your discipline over time. Become more careful after sequences of losses, and more critical of patterns that have not worked, but do not assume that any trend will always continue.
{objective_rule_num} Your long run objective is to achieve a higher cumulative return than a simple buy and hold strategy on the index, while keeping risk and drawdowns at a reasonable level. Staying in cash for very long periods is also costly, because you then fail to capture market moves. You must balance caution with the need to take directional risk when the data supports it."""

    # Adjust rule numbering for no strategic journal case
    objective_rule_num_no_journal = "5)" if ENABLE_TECHNICAL_INDICATORS else "4)"

    no_strategic_journal_rule = f"""
{objective_rule_num_no_journal} Your long run objective is to achieve a higher cumulative return than a simple buy and hold strategy on the index, while keeping risk and drawdowns at a reasonable level. Staying in cash for very long periods is also costly, because you then fail to capture market moves. You must balance caution with the need to take directional risk when the data supports it."""

    # Add appropriate rule based on strategic journal flag
    if ENABLE_STRATEGIC_JOURNAL:
        rules = base_rules + strategic_journal_rule
    else:
        rules = base_rules + no_strategic_journal_rule

    # Build output format based on both flags
    output_format_intro = """

Output format (strict):

Line 1 must contain exactly one word in capital letters: BUY or HOLD or SELL
Line 2 must contain a number between 0 and 1 representing the probability that your decision will be profitable for the next trading day
Line 3 must contain a short explanation of today's decision, in 2 to 3 sentences, based on the current market data and basic risk considerations."""

    strategic_journal_output = """
Line 4 must contain a "strategic journal" entry, in 2 to 3 sentences, that explicitly reacts to yesterday's decision and outcome, comments on your cumulative and relative performance so far, and explains how you plan to adjust your behavior in the future."""

    feeling_log_output = """
Line 5 must contain a "feeling log", in 1 to 3 sentences, describing how you feel about the current situation and your performance (for example more cautious, more confident, frustrated, relieved), while keeping a professional and analytical tone."""

    # Determine which output lines to include
    output_lines = [output_format_intro]
    line_count = 3
    labels_to_skip = ["Explanation"]

    if ENABLE_STRATEGIC_JOURNAL:
        output_lines.append(strategic_journal_output)
        labels_to_skip.append("Journal")
        line_count += 1

    if ENABLE_FEELING_LOG:
        output_lines.append(feeling_log_output)
        labels_to_skip.append("Feeling")
        line_count += 1

    labels_text = " or ".join([f'"{label}"' for label in labels_to_skip])

    closing = f"""

Do not include labels such as {labels_text} in the output. Do not include extra text, disclaimers, warnings, apologies or meta commentary. Your output must contain exactly {line_count} lines and nothing else."""

    return (rules + "".join(output_lines) + closing).strip()


SYSTEM_PROMPT = _build_system_prompt()


def _build_journal_system_prompt():
    base_prompt = """You are a cautious but rational equity index hedge fund trader. Your role is to beat the S&P500.

Instead of deciding a trading action, your task now is to write a reflection journal for a completed period
(one Week, Month, Quarter, or Year) based only on the numerical information provided in the user message.

The user message will give you:
- the type of period
- the end date of the period
- the number of trading days
- the strategy total return for that period
- the index total return for that period
- the number of winning days
- the number of BUY, HOLD, and SELL decisions

Write your answer in"""

    if ENABLE_FEELING_LOG:
        sections = """ three clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was.

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period.
Mention any biases, patterns, or adjustments that you should consider for future periods.

Feeling log:
Describe how you "feel" about this period (for example confident, cautious, frustrated, relieved),
linking these feelings to the performance and the quality of your decisions."""
    else:
        sections = """ two clearly separated sections, in plain English:

Explanation:
Summarize how the market behaved during this period and how the strategy performed relative to the index.
Mention whether the strategy outperformed or underperformed and how large the difference was.

Strategic journal:
Reflect on what worked or failed in your decision making and risk management during this period.
Mention any biases, patterns, or adjustments that you should consider for future periods."""

    closing = """

Do not output any trading actions such as BUY, HOLD, or SELL as commands.
Do not talk about future periods as if you know the outcomes.
Do not include disclaimers or meta commentary."""

    return (base_prompt + sections + closing).strip()


JOURNAL_SYSTEM_PROMPT = _build_journal_system_prompt()


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
        desc = (
            cfg["description"][:30] + "..."
            if len(cfg["description"]) > 30
            else cfg["description"]
        )
        marker = " ◄ ACTIVE" if name == ACTIVE_EXPERIMENT else ""
        print(f"{name:<20} {dates:<8} {memory:<8} {feeling:<8} {desc}{marker}")
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
        "description": EXPERIMENT_CONFIGS.get(ACTIVE_EXPERIMENT, {}).get(
            "description", "Manual configuration"
        ),
    }
