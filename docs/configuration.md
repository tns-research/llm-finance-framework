# Configuration Guide

This guide explains all configuration options available in the LLM Finance Experiment framework.

## 🔧 Configuration System Architecture

The framework uses a **hybrid configuration system** that combines simplicity for users with modern infrastructure benefits.

### For Users (You)
- **Single configuration file**: Edit `src/config.py` - all settings work
- **Familiar interface**: Simple variable assignments as before
- **No changes needed**: Existing configurations continue to work

### System Status
- **✅ Fully Configurable**: `DEBUG_SHOW_FULL_PROMPT`, `START_ROW`, `TEST_MODE`, `TEST_LIMIT`, `OPENROUTER_API_BASE`, `ACTIVE_EXPERIMENT`, `SHOW_DATE_TO_LLM`, `MA20_WINDOW`, `RET_5D_WINDOW`, `VOL20_WINDOW`
- **⚠️ Legacy System**: Other technical constants (RSI_WINDOW, MACD_FAST, etc.) - still work but less robust
- **🎯 Future**: Additional constants can be migrated if needed for research

### Why This Approach?
- **Stability**: Proven working system with minimal complexity
- **User Experience**: No disruption to existing workflows
- **Research Focus**: Prioritizes actual research over infrastructure perfection

## 📊 Data Configuration

### Basic Data Settings
```python
# Data time range
DATA_START = "2015-01-01"  # Start date for historical data
DATA_END = "2023-12-31"    # End date for historical data

# Asset selection
SYMBOL = "^GSPC"  # S&P 500 index ticker (Yahoo Finance format)
```

### Technical Indicators
```python
# ✅ FULLY CONFIGURABLE (new system - can be modified for research)
MA20_WINDOW = 20        # 20-day moving average window (configurable)
RET_5D_WINDOW = 5       # 5-day return calculation window (configurable)
VOL20_WINDOW = 20       # 20-day volatility window (configurable)

# ⚠️ LEGACY SYSTEM (still works, defined in config.py)
PAST_RET_LAGS = 20      # Number of lagged return features
RSI_WINDOW = 14         # RSI period (14 days standard)
RSI_OVERBOUGHT = 70     # RSI overbought threshold
RSI_OVERSOLD = 30       # RSI oversold threshold

MACD_FAST = 12          # Fast EMA period for MACD
MACD_SLOW = 26          # Slow EMA period for MACD
MACD_SIGNAL = 9         # Signal line EMA period for MACD

STOCH_K = 14            # %K period for Stochastic Oscillator
STOCH_D = 3             # %D smoothing period for Stochastic Oscillator
STOCH_SMOOTH_K = 3      # %K smoothing period (optional)
STOCH_OVERBOUGHT = 80   # Stochastic overbought threshold
STOCH_OVERSOLD = 20     # Stochastic oversold threshold

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
    "SHOW_DATE_TO_LLM": False,        # ✅ Configurable via ACTIVE_EXPERIMENT
    "ENABLE_STRATEGIC_JOURNAL": False, # ✅ Configurable via ACTIVE_EXPERIMENT
    "ENABLE_FEELING_LOG": False,      # ✅ Configurable via ACTIVE_EXPERIMENT
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

## 🤖 Model Configuration

### Available Models
```python
LLM_MODELS = [
    {
        "tag": "bert",
        "router_model": "openrouter/bert-nebulon-alpha",
    },
    # Add more models as needed
]
```

### Model Selection
Models are selected based on the `tag` field. The framework automatically appends the experiment configuration to create unique identifiers (e.g., `bert_memory_feeling`).

## ⚙️ Runtime Configuration

### Testing and Development
```python
# Use dummy model for development/testing
USE_DUMMY_MODEL = True  # Set to False for real LLM experiments

# ✅ FULLY CONFIGURABLE (new system)
DEBUG_SHOW_FULL_PROMPT = False  # Show complete prompts for debugging

# ✅ FULLY CONFIGURABLE (new system)
TEST_MODE = True      # Enable test mode (limits data processing)
TEST_LIMIT = 500       # Number of days to process in test mode
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

API keys are read from environment variables:
```bash
export OPENROUTER_API_KEY="your_key_here"
```

## 🛠️ Helper Functions

### Configuration Inspection
```python
# List all available experiments
from src.config_compat import list_experiments
list_experiments()

# Get current configuration summary
from src.config_compat import get_current_config_summary
config = get_current_config_summary()
```

### Experiment Naming
```python
# Get experiment suffix for file naming
from src.config_compat import get_experiment_suffix
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
- Set `USE_DUMMY_MODEL = True` for prompt testing
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
    ↓ Modern access
src/config_compat.py (Backward Compatibility Layer)
    ↓ Clean imports
src/*.py (Application Code)
```

### Future Considerations
The system is **stable and functional**. Additional constants (RSI_WINDOW, MACD_FAST, etc.) can be migrated to the new system if specific research needs arise, but the current hybrid approach provides the best balance of simplicity and capability.

This configuration system enables systematic, reproducible research into LLM financial decision-making capabilities.
