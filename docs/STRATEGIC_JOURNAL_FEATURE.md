# Experiment Configuration Guide

## Overview

This project supports **6 predefined experiment configurations** to systematically test the impact of different context types on LLM trading decisions:

- **Dates**: Real calendar dates vs. anonymized time series
- **Memory**: Strategic journal with past decisions and performance feedback
- **Feeling**: Emotional self-reflection log

## Quick Start

Set your experiment in `src/config.py`:

```python
ACTIVE_EXPERIMENT = "baseline"  # or: memory_only, memory_feeling, 
                                 #     dates_only, dates_memory, dates_full
```

Then run: `python -m src.main`

## Available Experiments

### Without Dates (Anonymized Time Series)

| Experiment | Memory | Feeling | Description |
|------------|--------|---------|-------------|
| `baseline` | ❌ | ❌ | Minimal context - pure market data only |
| `memory_only` | ✅ | ❌ | Strategic journal enabled, no feeling log |
| `memory_feeling` | ✅ | ✅ | Full memory with emotional reflection |

### With Dates (Real Calendar Dates)

| Experiment | Memory | Feeling | Description |
|------------|--------|---------|-------------|
| `dates_only` | ❌ | ❌ | Dates shown, no memory or feeling |
| `dates_memory` | ✅ | ❌ | Dates + strategic journal |
| `dates_full` | ✅ | ✅ | Maximum context (dates + memory + feeling) |

## Research Questions

Each experiment helps answer specific questions:

| Comparison | Research Question |
|------------|-------------------|
| `baseline` vs `memory_only` | Does strategic memory improve learning from past decisions? |
| `baseline` vs `dates_only` | Does showing dates cause data leakage / overfitting? |
| `memory_only` vs `memory_feeling` | Does emotional reflection add value or noise? |
| `baseline` vs `dates_full` | What's the combined effect of all context types? |
| Any no-dates vs with-dates | Can the LLM exploit historical calendar patterns? |

## Response Format by Configuration

The LLM output format adapts to the experiment:

| Configuration | Lines | Format |
|--------------|-------|--------|
| Both `False` | 3 | Decision, Probability, Explanation |
| Memory only | 4 | Decision, Probability, Explanation, Strategic Journal |
| Feeling only | 4 | Decision, Probability, Explanation, Feeling Log |
| Both `True` | 5 | Decision, Probability, Explanation, Strategic Journal, Feeling Log |

## Output Organization

Results are automatically tagged with the experiment name:

```
results/
├── parsed/
│   ├── grok_fast_baseline_parsed.csv
│   ├── grok_fast_memory_only_parsed.csv
│   ├── grok_fast_dates_full_parsed.csv
│   └── ...
├── plots/
│   ├── grok_fast_baseline_calibration.png
│   ├── grok_fast_baseline_baseline_comparison.png
│   └── ...
└── analysis/
    ├── grok_fast_baseline_pattern_analysis.md
    └── ...
```

## Running Multiple Experiments

To run all 6 experiments systematically:

```python
# In a script or notebook
experiments = ["baseline", "memory_only", "memory_feeling", 
               "dates_only", "dates_memory", "dates_full"]

for exp in experiments:
    # Edit config.py to set ACTIVE_EXPERIMENT = exp
    # Then run: python -m src.main
    pass
```

Or create a batch runner (future enhancement).

## Helper Functions

```python
from src.config import list_experiments, get_current_config_summary

# Show all available experiments
list_experiments()

# Get current settings as dict
config = get_current_config_summary()
print(config)
```

## Manual Configuration

If you need custom settings not covered by the 6 presets:

```python
# In src/config.py
ACTIVE_EXPERIMENT = None  # Disable preset selection

# Then set manual flags
_MANUAL_SHOW_DATE_TO_LLM = True
_MANUAL_ENABLE_STRATEGIC_JOURNAL = True
_MANUAL_ENABLE_FEELING_LOG = False
```

## Implementation Details

### Config Flags

| Flag | Purpose |
|------|---------|
| `SHOW_DATE_TO_LLM` | Include real dates in prompts (e.g., "Date: 2018-05-21") |
| `ENABLE_STRATEGIC_JOURNAL` | Include past trade history and performance in prompts |
| `ENABLE_FEELING_LOG` | Request emotional self-reflection in LLM output |

### Files Involved

- **`src/config.py`**: Experiment definitions and active selection
- **`src/main.py`**: Applies experiment suffix to model tags
- **`src/trading_engine.py`**: Conditional prompt building
- **`src/backtest.py`**: Flexible response parsing
- **`src/dummy_model.py`**: Respects config flags for testing

## Notes

- The dummy model respects all experiment flags for testing
- Baseline comparisons are computed separately for each experiment run
- When memory/feeling is disabled, placeholder text is stored in parsed results
- Longer experiments (500+ days) recommended for statistical significance
