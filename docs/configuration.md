# Configuration Guide

This guide explains all configuration options available in the LLM Finance Experiment framework.

## 🔧 Configuration System Architecture

The framework uses a **hybrid configuration system** that combines simplicity for users with modern infrastructure benefits.

### For Users (You)
- **Single configuration file**: Edit `src/config.py` - all settings work
- **Familiar interface**: Simple variable assignments as before
- **No changes needed**: Existing configurations continue to work

### System Status
- **[OK] Fully Configurable**: `DEBUG_SHOW_FULL_PROMPT`, `START_ROW`, `TEST_MODE`, `TEST_LIMIT`, `OPENROUTER_API_BASE`, `ACTIVE_EXPERIMENT`, `SHOW_DATE_TO_LLM`, `MA20_WINDOW`, `RET_5D_WINDOW`, `VOL20_WINDOW`
- **[!] Legacy System**: Other technical constants (RSI_WINDOW, MACD_FAST, etc.) - still work but less robust
- **[*] Future**: Additional constants can be migrated if needed for research

### Why This Approach?
- **Stability**: Proven working system with minimal complexity
- **User Experience**: No disruption to existing workflows
- **Research Focus**: Prioritizes actual research over infrastructure perfection

## 🚀 Execution & Runtime

### Main Execution Command
The framework runs via the main module:
```bash
python -m src.main
```

### Model Selection (Provider)
`LLM_PROVIDER` is the single knob that decides how trading decisions are generated:

```python
LLM_PROVIDER = "dummy"  # Random decisions, no API/CLI, no cost. For testing.
# LLM_PROVIDER = "openrouter"            # Any model via OpenRouter HTTP API (needs OPENROUTER_API_KEY)
# LLM_PROVIDER = "claude_code"           # Claude Code via your subscription, single-shot (needs the claude CLI logged in)
# LLM_PROVIDER = "claude_code_subagents" # Claude Code subscription, multi-agent (lead PM + specialist analysts)

# Claude model used by the claude_code providers
CLAUDE_CODE_MODEL = "sonnet"  # "sonnet", "opus", "haiku", or a full model id
```

> **Note**: `USE_DUMMY_MODEL` is derived automatically from `LLM_PROVIDER` (`USE_DUMMY_MODEL = LLM_PROVIDER == "dummy"`). Never set it by hand; it only exists for backward compatibility.

### API Key Configuration
The `openrouter` provider requires an OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your_key_here"
```
- **Cost**: ~1-5 cents per API call depending on model and prompt length
- **Free Option**: Some models like `deepseek/deepseek-r1-0528:free` are available
- **No cost / no key**: Use `LLM_PROVIDER = "dummy"` for development, or the `claude_code` providers to run on your Claude subscription (no per-token cost). See [docs/providers.md](providers.md).

## 📊 Data Configuration

### Step 1: Choose Data Source
Select your data source first:
```python
# 🎯 DATA SOURCE SELECTION
DATA_SOURCE = "vendored"  # Default: committed frozen SPY snapshot, offline + deterministic
# DATA_SOURCE = "csv"     # Your own local CSV file
# DATA_SOURCE = "stooq"   # Live fetch (now requires an apikey; no longer the default)
```

### Step 2: Set Date Range
Configure the date range (applies to all data sources):
```python
# DATE RANGE (applies to all data sources)
DATA_START = "2015-01-01"  # Start date for historical data
DATA_END = "2023-12-31"    # End date for historical data
```

### Step 3: Configure Chosen Data Source

#### Vendored Snapshot (Default - offline, deterministic)
When `DATA_SOURCE = "vendored"` (the default), the framework loads a committed real SPY snapshot, verified by SHA-256 against its manifest. No network, no key, reproducible out of the box:
```python
# Committed real SPY data so a clean clone runs offline and deterministically
VENDORED_DATA_PATH = "data/raw/spy_daily.csv"
VENDORED_MANIFEST_PATH = "data/raw/MANIFEST.json"
```
To refresh the snapshot from a live source, see `scripts/refresh_data.py` and `data/raw/PROVENANCE.md`.

#### CSV Data Files (your own local dataset)
When `DATA_SOURCE = "csv"`:
```python
# Symbol for CSV data (only used when DATA_SOURCE = "csv")
SYMBOL = "^GSPC"  # Stock symbol in CSV file
CSV_DATA_PATH = "data/raw/sp500.csv"  # Path to your CSV file
# Must contain columns: Date, Open, High, Low, Close, Volume
```

#### Stooq Historical Data (live refresh option, needs an apikey)
When `DATA_SOURCE = "stooq"`. Note: Stooq now requires an apikey (and may present a captcha), so it is no longer the default nor a zero-setup option:
```python
# Symbol selection (Stooq format - no special characters needed)
STOOQ_SYMBOL = "SPY"     # S&P 500 ETF (recommended for broad market)
# STOOQ_SYMBOL = "QQQ"   # NASDAQ 100 ETF
# STOOQ_SYMBOL = "AAPL"  # Individual stocks
# STOOQ_SYMBOL = "BTC-USD"  # Cryptocurrencies
```

### Trading Delay Warning
**⚠️ Important**: Trading doesn't begin immediately after `DATA_START`!

Due to technical indicator warm-up requirements, the system automatically skips the first ~40 trading days of your dataset. Even with `START_ROW = 0`, you'll start trading approximately 40 trading days after your configured `DATA_START` date.

**Example:** If `DATA_START = "2015-01-01"`, actual trading begins around March 2015.

This delay ensures reliable technical indicators (RSI needs 14+ days, MACD needs 35+ days, etc.).

### Configuration Validation
The framework automatically validates your configuration and provides helpful guidance:

**Successful Vendored Configuration (default):**
```
[CONFIG] [OK] Primary: vendored frozen snapshot (offline, checksummed)
[CONFIG] 📦 File: data/raw/spy_daily.csv
[CONFIG] 🔄 Fallback: None (vendored is offline-by-design)
[CONFIG] [DATES] Date range: 2015-03-03 to 2023-12-31
[CONFIG] 🧪 Test mode: ON (5 days)
```

**Helpful Warnings:**
```
[WARNING] CSV file not found: data/raw/sp500.csv
[INFO] CSV_FALLBACK_PATH variable detected. Consider using CSV_DATA_PATH instead.
```

### Technical Indicators
```python
# [CONFIG] FULLY CONFIGURABLE (new system - can be modified for research)
MA20_WINDOW = 20        # 20-day moving average window (configurable)
RET_5D_WINDOW = 5       # 5-day return calculation window (configurable)
VOL20_WINDOW = 20       # 20-day volatility window (configurable)

# [WARN] LEGACY SYSTEM (still works, defined in config.py)
PAST_RET_LAGS = 20      # Number of lagged return features
RSI_WINDOW = 14         # RSI period (14 days standard)
RSI_OVERBOUGHT = 70     # RSI overbought threshold
RSI_OVERSOLD = 30       # RSI oversold threshold

# MACD (Moving Average Convergence Divergence)
MACD_FAST = 12          # Fast EMA period for MACD
MACD_SLOW = 26          # Slow EMA period for MACD
MACD_SIGNAL = 9         # Signal line EMA period for MACD

# Stochastic Oscillator
STOCH_K = 14            # %K period for Stochastic Oscillator
STOCH_D = 3             # %D smoothing period for Stochastic Oscillator
STOCH_SMOOTH_K = 3      # %K smoothing period (optional)
STOCH_OVERBOUGHT = 80   # Stochastic overbought threshold
STOCH_OVERSOLD = 20     # Stochastic oversold threshold

# Bollinger Bands
BB_WINDOW = 20          # Bollinger Bands period
BB_STD = 2              # Standard deviations for Bollinger Bands

# Technical indicators control
ENABLE_TECHNICAL_INDICATORS = True  # Show indicators to LLM
```

### Technical Indicators Memory System
```python
# Historical technical indicators (automatically enabled with ENABLE_TECHNICAL_INDICATORS)
# These create 20-day lagged series for enhanced pattern recognition in daily prompts:
# RSI_LAG_1 through RSI_LAG_20: Historical RSI values for trend analysis
# MACD_HIST_LAG_1 through MACD_HIST_LAG_20: Historical MACD histogram values for momentum analysis
# STOCH_K_LAG_1 through STOCH_K_LAG_20: Historical Stochastic %K values for cycle analysis
# BB_POSITION_LAG_1 through BB_POSITION_LAG_20: Historical Bollinger Band positions for volatility analysis

# Memory system aggregates these for weekly/monthly/quarterly/yearly summaries:
# - Weekly RSI: Average RSI with overbought/oversold percentages and range
# - Monthly MACD: Bullish percentage and average histogram strength
# - Quarterly Stochastic: Overbought/oversold conditions over period
# - Yearly Bollinger Bands: Average position and band touch frequency
```

## 🧪 Experiment Configurations

The framework provides 6 predefined experiment configurations to systematically test different aspects of LLM trading:

### Experiment Types

#### No Dates (Anonymized Time Series)
```python
"baseline": {
    "description": "Minimal context: no dates, no memory, no feeling",
    "SHOW_DATE_TO_LLM": False,        # [CONFIG] Configurable via ACTIVE_EXPERIMENT
    "ENABLE_STRATEGIC_JOURNAL": False, # [CONFIG] Configurable via ACTIVE_EXPERIMENT
    "ENABLE_FEELING_LOG": False,      # [CONFIG] Configurable via ACTIVE_EXPERIMENT
}
```
**Use Case**: Test pure technical analysis capability without temporal context.

### ⚠️ **Why "No Dates" Mode is Critical for Research Integrity**

**Date anonymization is the preferred experimental setup** because LLMs trained on internet-scale data may have acquired knowledge of major historical events, market crashes, and economic cycles. This creates a significant methodological confound:

#### **Data Leakage Risk:**
- **Historical Recognition**: LLMs might identify dates like "2008-09" and recall the Global Financial Crisis
- **Event-Based Trading**: Models could trade based on known historical patterns rather than technical analysis
- **Knowledge Contamination**: Pre-trained knowledge of events like 9/11, COVID-19, or Federal Reserve actions
- **Unfair Advantage**: Models effectively have access to "future" information from their training data

#### **Research Validity:**
- **Pure Logic Testing**: Without dates, we test genuine pattern recognition and trading logic
- **Contamination Control**: Eliminates historical knowledge as a confounding variable
- **Fair Comparison**: All models operate under the same informational constraints
- **Scientific Rigor**: Ensures results reflect algorithmic capabilities, not data leakage

#### **Experimental Best Practice:**
```python
# Always start with date-anonymized experiments
ACTIVE_EXPERIMENT = "memory_feeling"  # No dates by default
# Only use date-enabled configs after establishing baseline performance
```

**Date-enabled experiments should only be used after establishing that the LLM demonstrates genuine trading skill in anonymized conditions.** This prevents false positives from historical pattern recognition rather than actual trading acumen.

```python
"memory_only": {
    "description": "Memory/journal only: no dates, no feeling",
    "SHOW_DATE_TO_LLM": False,
    "ENABLE_STRATEGIC_JOURNAL": True,
    "ENABLE_FEELING_LOG": False,
}
```
**Use Case**: Test if LLMs can learn from performance feedback.

```python
"memory_feeling": {
    "description": "Memory + feeling: no dates",
    "SHOW_DATE_TO_LLM": False,
    "ENABLE_STRATEGIC_JOURNAL": True,
    "ENABLE_FEELING_LOG": True,
}
```
**Use Case**: Test if emotional self-reflection improves decision quality.

#### With Dates (Real Calendar Context)
```python
"dates_only": {
    "description": "Dates only: no memory, no feeling",
    "SHOW_DATE_TO_LLM": True,
    "ENABLE_STRATEGIC_JOURNAL": False,
    "ENABLE_FEELING_LOG": False,
}
```
**Use Case**: Test if calendar awareness affects decision patterns.

```python
"dates_memory": {
    "description": "Dates + memory: no feeling",
    "SHOW_DATE_TO_LLM": True,
    "ENABLE_STRATEGIC_JOURNAL": True,
    "ENABLE_FEELING_LOG": False,
}
```
**Use Case**: Test interaction between temporal awareness and learning.

```python
"dates_full": {
    "description": "Full context: dates + memory + feeling",
    "SHOW_DATE_TO_LLM": True,
    "ENABLE_STRATEGIC_JOURNAL": True,
    "ENABLE_FEELING_LOG": True,
}
```
**Use Case**: Maximum context scenario for comprehensive evaluation.

### Selecting Experiments

#### Using Predefined Configurations
```python
# ✅ FULLY CONFIGURABLE (new system)
ACTIVE_EXPERIMENT = "memory_feeling"  # One of: baseline, memory_only, memory_feeling,
                                     #         dates_only, dates_memory, dates_full
```

#### Manual Configuration
```python
# Or set manually (when ACTIVE_EXPERIMENT = None)
ACTIVE_EXPERIMENT = None

_MANUAL_SHOW_DATE_TO_LLM = True
_MANUAL_ENABLE_STRATEGIC_JOURNAL = True
_MANUAL_ENABLE_FEELING_LOG = False
```

## 🎭 Trader Personality Configuration

### Overview
The framework supports **5 distinct trader personalities** that influence how LLMs approach trading decisions. Each personality provides a different behavioral framework without dictating specific rules.

### Personality Selection
```python
# 🤖 LLM PERSONALITY SETTINGS
ACTIVE_PERSONALITY = "cautious"  # Choose trader personality:
# - "cautious": Conservative, risk-averse trading
# - "aggressive": Bold, opportunity-focused trading
# - "balanced": Systematic, balanced approach
# - "momentum": Trend-following strategies
# - "contrarian": Counter-trend strategies
```

### Personality Descriptions

#### Cautious Conservative
- **Risk Tolerance**: Low
- **Decision Style**: Defensive
- **Behavioral Framework**: Prioritizes capital preservation, requires strong conviction for directional trades. Prefers HOLD when signals are mixed and takes quick action to exit losing positions.

#### Aggressive Growth
- **Risk Tolerance**: High
- **Decision Style**: Offensive
- **Behavioral Framework**: Seeks alpha through active positioning, tolerates higher volatility for potential returns. More willing to take directional risk based on market momentum.

#### Balanced Professional
- **Risk Tolerance**: Medium
- **Decision Style**: Systematic
- **Behavioral Framework**: Balances risk and reward systematically, follows structured decision criteria. Makes decisions based on comprehensive analysis rather than instinct.

#### Momentum Trader
- **Risk Tolerance**: High
- **Decision Style**: Reactive
- **Behavioral Framework**: Capitalizes on market trends, quick to cut losses and ride winners. Emphasizes timing and market direction over fundamental valuation.

#### Contrarian Value
- **Risk Tolerance**: Medium
- **Decision Style**: Contrarian
- **Behavioral Framework**: Fades market sentiment, buys fear and sells greed when fundamentals suggest. Goes against prevailing market psychology when indicators show extremes.

### Impact on Research
- **Behavioral Consistency**: Study whether LLMs maintain personality-consistent decision patterns
- **Market Regime Performance**: Analyze which personalities excel in different market conditions
- **Decision Pattern Analysis**: Compare how different behavioral frameworks influence actual trading decisions
- **LLM Psychology Research**: Understand how behavioral prompts affect AI decision-making

## 🧠 Chain of Thought Reasoning

### Overview
The framework supports structured analytical reasoning in LLM prompts, enabling step-by-step decision processes independent of experiment configuration.

### Configuration
```python
# 🧠 CHAIN OF THOUGHT REASONING
ENABLE_CHAIN_OF_THOUGHT = True  # Enable structured analytical reasoning
```

### Important: Breaking Change
**⚠️ BREAKING CHANGE**: As of recent updates, `ENABLE_CHAIN_OF_THOUGHT` works **independently** of experiment selection. This toggle enables reasoning regardless of which experiment type is active (`ACTIVE_EXPERIMENT`).

Previously, chain of thought was tied to specific experiment configurations. Now it functions as a master toggle that can be combined with any experiment type.

### Impact on Prompts
When enabled, LLMs receive additional structured reasoning prompts that encourage:
- Step-by-step market analysis
- Systematic evaluation of technical indicators
- Logical decision justification
- Risk-reward assessment frameworks

### Research Applications
- **Decision Quality**: Study if structured reasoning improves trading decisions
- **Process Transparency**: Analyze LLM thought processes and decision logic
- **Methodological Rigor**: Compare intuitive vs. analytical decision-making approaches

## 🤖 Model Configuration

### Available Models
```python
LLM_MODELS = [
    {
        "tag": "deepseek-r1-0528",
        "router_model": "deepseek/deepseek-r1-0528:free",
    },
    # {
    #     "tag": "bert",  # Model no longer available
    #     "router_model": "openrouter/bert-nebulon-alpha",
    # },
    # {
    #     "tag": "chimera",
    #     "router_model": "tngtech/tng-r1t-chimera:free",
    # },
    # {
    #     "tag": "olmo-32b",
    #     "router_model": "allenai/olmo-3-32b-think",
    # },
    # {
    #     "tag": "gpt-oss-120b",
    #     "router_model": "openai/gpt-oss-120b:free",
    # },
    # {
    #     "tag": "gpt-oss-20b",
    #     "router_model": "openai/gpt-oss-20b:free",
    # },
    # {
    #     "tag": "claude",
    #     "router_model": "anthropic/claude-3-sonnet",
    # },
]
```

### Model Selection
The `LLM_MODELS` list above only applies when `LLM_PROVIDER = "openrouter"`. Models are selected based on the `tag` field, and the framework automatically appends the experiment configuration to create unique identifiers (e.g., `deepseek-r1-0528_memory_feeling`).

For the other providers the model list is not used:
- `LLM_PROVIDER = "dummy"` runs the deterministic stub model (no API, no cost).
- `LLM_PROVIDER = "claude_code"` / `"claude_code_subagents"` use the local Claude Code session (no OpenRouter key, no per-call cost).

## ⚙️ Runtime Configuration

### Testing and Development
```python
# Select the model backend (the single knob you set by hand)
LLM_PROVIDER = "dummy"  # "dummy" | "openrouter" | "claude_code" | "claude_code_subagents"
# USE_DUMMY_MODEL is DERIVED from LLM_PROVIDER (== "dummy"); never set it by hand.

# ✅ FULLY CONFIGURABLE (new system)
DEBUG_SHOW_FULL_PROMPT = False  # Show complete prompts for debugging

# ✅ FULLY CONFIGURABLE (new system)
TEST_MODE = True      # Enable test mode (limits data processing)
TEST_LIMIT = 5         # Number of days to process in test mode
```

### Starting Position
```python
# ✅ FULLY CONFIGURABLE (new system)
# Override automatic start position (for testing specific periods)
START_ROW = 30  # None = automatic, or specific row number
```

## 🎯 Decision Framework

### Position Mapping
```python
POSITION_MAP = {
    "BUY": 1.0,   # Long position
    "HOLD": 0.0,  # Cash position
    "SELL": -1.0, # Short position
}
```

## 🕐 Full Trading History

### Overview
The full trading history feature provides the LLM with complete historical trading data in every prompt, enabling long-term pattern recognition and learning.

**When to Enable:**
- ✅ Long-term pattern recognition needed
- ✅ Comprehensive historical context required
- ✅ Advanced learning capabilities desired

**When to Disable:**
- ❌ Token efficiency critical (saves ~10-20 tokens per trade)
- ❌ Prevent potential date identification through performance patterns
- ❌ Simpler experimental setup preferred

### Configuration
```python
# Include complete historical trades in prompts as CSV data
ENABLE_FULL_TRADING_HISTORY = True  # Set to False for token efficiency
```

### Format
Trading history format adapts based on date configuration:

#### With Dates Enabled (`SHOW_DATE_TO_LLM = True`)
```
TRADING_HISTORY:
date,decision,position,result
2024-01-01,BUY,1.0,0.0234
2024-01-02,HOLD,0.0,0.0000
2024-01-03,SELL,-1.0,-0.0156
...
```

#### With Dates Disabled (`SHOW_DATE_TO_LLM = False`)
```
TRADING_HISTORY:
trade_id,decision,position,result
1,BUY,1.0,0.0234
2,HOLD,0.0,0.0000
3,SELL,-1.0,-0.0156
...
```

Where:
- `date`/`trade_id`: Date (when enabled) or sequential trade ID (when anonymized)
- `decision`: BUY/HOLD/SELL choice
- `position`: Position value (-1.0, 0.0, 1.0)
- `result`: Strategy return for that day

### Benefits
- **Long-term Memory**: Access to complete trading history
- **Pattern Recognition**: Identify recurring market patterns
- **Performance Analysis**: Track decision-outcome relationships
- **Learning Evolution**: Observe strategy adaptation over time

### Token Considerations
- **Data Volume**: Adds ~4 tokens per historical trade
- **Cumulative Growth**: Day 100 adds ~400 tokens to each prompt
- **Format Efficiency**: CSV structure minimizes token usage
- **Optional Feature**: Can be disabled for token-constrained experiments
- **Pure Data**: No verbose text, only CSV-structured information

## 🔧 System Prompts

The framework automatically builds appropriate system prompts based on configuration:

### Base Rules (Always Included)
- Decision must be exactly one of: BUY, HOLD, SELL
- Use only provided information (no external knowledge)
- Balance return potential with risk management
- Avoid extreme risk-seeking behavior

### Strategic Journal Rules (When Enabled)
- Use historical performance feedback to refine decisions
- Learn from past mistakes and successes
- Adapt risk tolerance based on recent performance
- Maintain long-term objective of beating buy-and-hold

### Feeling Log Integration (When Enabled)
- Self-reflection on decision confidence
- Emotional state tracking
- Risk perception assessment

## 📊 API Configuration

### OpenRouter Settings
```python
# ✅ FULLY CONFIGURABLE (new system)
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
```

### API Interface
The framework uses OpenRouter's OpenAI-compatible API:

**Request Format:**
```json
{
  "model": "deepseek/deepseek-r1-0528:free",
  "temperature": 0.0,
  "max_tokens": 50000,
  "messages": [
    {"role": "system", "content": "system_prompt"},
    {"role": "user", "content": "user_prompt"}
  ]
}
```

**Response Format:**
```json
{
  "choices": [
    {
      "message": {
        "content": "LLM response text"
      }
    }
  ]
}
```

### Authentication
API keys are read from environment variables:
```bash
export OPENROUTER_API_KEY="your_key_here"
```

## 🛠️ Helper Functions

### Configuration Inspection
```python
# List all available experiments
from src.config import list_experiments
list_experiments()

# Get current configuration summary
from src.config import get_current_config_summary
config = get_current_config_summary()
```

### Experiment Naming
```python
# Get experiment suffix for file naming
from src.config import get_experiment_suffix
suffix = get_experiment_suffix()  # Returns "_memory_feeling" etc.
```

## 🎯 Best Practices

### Research Workflow
1. **Start with baseline**: Establish performance floor
2. **Add memory**: Test learning capability
3. **Add dates**: Test temporal awareness
4. **Add feelings**: Test self-reflection impact

### Performance Optimization
- Use `TEST_MODE = True` for development
- Set `LLM_PROVIDER = "dummy"` for prompt testing (no API, no cost)
- Limit `TEST_LIMIT` for quick iterations

### Model Selection
- Start with smaller, faster models for testing
- Use larger models for final research results
- Compare multiple models for robustness

## 🔄 Configuration System Status

### What Works
- **All settings are functional**: Modify `src/config.py` and they take effect
- **Backward compatibility**: Existing configurations continue to work
- **Research flexibility**: Core window sizes are now configurable for experiments

### Current Architecture
```
src/config.py (User Interface - Single Source of Truth)
    ↓ Legacy reading
src/configuration_manager.py (New System Bridge)
    ↓ Clean imports
src/*.py (Application Code)
```

### Future Considerations
The system is **stable and functional**. Additional constants (RSI_WINDOW, MACD_FAST, etc.) can be migrated to the new system if specific research needs arise, but the current hybrid approach provides the best balance of simplicity and capability.

This configuration system enables systematic, reproducible research into LLM financial decision-making capabilities.
