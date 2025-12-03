# Experiment Design Guide

This guide helps researchers choose the right configuration for their study and understand what outputs they'll get. It bridges the gap between research questions and technical settings.

## 🎯 Research Question → Experiment Type

Choose your experiment based on what you want to study:

| Experiment Type | Research Question | Key Settings | Master Toggles | What It Tests | Best For |
|---|---|---|---|---|---|
| **`baseline`** | How does LLM perform with no context? | `ACTIVE_EXPERIMENT = "baseline"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = False`<br>`ENABLE_FEELING_LOG = False`<br>`SHOW_DATE_TO_LLM = False` | Pure algorithmic capability<br>Technical indicator usage | Establishing baseline performance<br>Model capability assessment |
| **`memory_only`** | How does LLM learn from experience? | `ACTIVE_EXperiment = "memory_only"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = True`<br>`ENABLE_FEELING_LOG = False`<br>`SHOW_DATE_TO_LLM = False` | Self-reflection and adaptation<br>Pattern recognition from history | Learning dynamics research<br>Memory system evaluation |
| **`memory_feeling`** | How do emotions affect trading? | `ACTIVE_EXPERIMENT = "memory_feeling"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = True`<br>`ENABLE_FEELING_LOG = True`<br>`SHOW_DATE_TO_LLM = False` | Emotional intelligence<br>Confidence vs performance<br>Behavioral biases | Behavioral finance studies<br>LLM psychology research |
| **`dates_only`** | Do LLMs use calendar patterns? | `ACTIVE_EXPERIMENT = "dates_only"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = False`<br>`ENABLE_FEELING_LOG = False`<br>`SHOW_DATE_TO_LLM = True` | Historical knowledge usage<br>Pattern recognition ability<br>Potential data leakage ⚠️ | Data contamination studies<br>Calendar effect research |
| **`dates_memory`** | How do dates + memory interact? | `ACTIVE_EXPERIMENT = "dates_memory"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = True`<br>`ENABLE_FEELING_LOG = False`<br>`SHOW_DATE_TO_LLM = True` | Context integration<br>Historical + experiential learning | Advanced learning research<br>Context utilization studies |
| **`dates_full`** | What's the maximum LLM capability? | `ACTIVE_EXPERIMENT = "dates_full"` | `ENABLE_TECHNICAL_INDICATORS = True`<br>`ENABLE_FULL_TRADING_HISTORY = True`<br>`ENABLE_STRATEGIC_JOURNAL = True`<br>`ENABLE_FEELING_LOG = True`<br>`SHOW_DATE_TO_LLM = True` | Peak performance assessment<br>All context utilization | Benchmarking studies<br>Capability demonstration |

## ⚙️ Configuration Impact Guide

### Master Toggles (Fundamental Controls)

| Toggle | Default | What It Controls | Research Impact |
|---|---|---|---|
| **`ENABLE_TECHNICAL_INDICATORS`** | `True` | RSI, MACD, Stochastic, Bollinger in prompts | **Quantitative vs Qualitative**: `True` = technical analysis, `False` = fundamental reasoning only |
| **`ENABLE_FULL_TRADING_HISTORY`** | `True` | Complete trading record in context | **Memory Depth**: `True` = full history, `False` = limited context window |
| **`ENABLE_STRATEGIC_JOURNAL`** | Varies | LLM's own trading notes and reasoning | **Self-Reflection**: `True` = learns from past decisions and outcomes |
| **`ENABLE_FEELING_LOG`** | Varies | Emotional state descriptions | **Behavioral Analysis**: `True` = tracks confidence, frustration, etc. |
| **`SHOW_DATE_TO_LLM`** | Varies | Calendar dates in prompts | **Data Leakage Testing**: `True` = potential hindsight bias ⚠️ |

### Window Sizes (Now Configurable!)

| Setting | Default | Research Use | Impact on Strategy |
|---|---|---|---|
| `MA20_WINDOW` | 20 | Trend sensitivity testing | Smaller (10) = responsive/noisy<br>Larger (50) = smooth/slow |
| `RET_5D_WINDOW` | 5 | Momentum horizon adjustment | Shorter (3) = fast signals<br>Longer (10) = confirmed trends |
| `VOL20_WINDOW` | 20 | Risk assessment period | Shorter = volatile signals<br>Longer = stable risk view |

### Scale Settings

| Setting | Development | Research | Impact |
|---|---|---|---|
| `TEST_MODE` | `True` | `False` | Quick iteration vs comprehensive analysis |
| `TEST_LIMIT` | 100-500 | 2700+ | 6 months vs 10+ years of data |
| `USE_DUMMY_MODEL` | `True` | `False` | Synthetic vs real LLM responses |

## 📊 Expected Outputs Guide

### Standard Outputs (All Experiments)

| Output Type | Location | Content | Primary Use |
|---|---|---|---|
| **Parsed Results** | `results/parsed/model_experiment_parsed.csv` | Raw decisions + explanations + confidence | Decision pattern analysis |
| **Performance Plots** | `results/plots/` | Equity curves, risk charts, calibration plots | Visual performance assessment |
| **Statistical Analysis** | `results/analysis/` JSON files | Bootstrap tests, significance, effect sizes | Rigorous validation |
| **HTML Report** | `results/reports/comprehensive_report.html` | Executive summary + all metrics | Publication-ready overview |

### Experiment-Specific Outputs

| Experiment Type | Bonus Outputs | Unique Insights |
|---|---|---|
| **`memory_only`**<br>**`memory_feeling`** | Strategic journal evolution<br>Decision adaptation analysis | How LLM strategy changes over time<br>Learning from wins/losses |
| **`dates_only`**<br>**`dates_memory`**<br>**`dates_full`** | Calendar pattern analysis<br>Performance by year/month | Historical knowledge usage<br>Potential data leakage detection |
| **`memory_feeling`** | Emotional state tracking<br>Confidence vs outcomes correlation | Behavioral biases in AI<br>Emotional intelligence assessment |

## 🚀 Research Workflow

### Quick Start (5 minutes)
1. **Choose research question** → Pick experiment type from table above
2. **Set basic config** → Modify `src/config.py` with chosen settings
3. **Run experiment** → `python -m src.main`

### Full Research Process (30 min - 2 hours)
1. **Design** (5 min): Select experiment type + adjust parameters
2. **Execute** (5-120 min): Run `python -m src.main`
3. **Analyze** (15 min): Review outputs in `results/` directory
4. **Iterate** (5 min): Adjust settings, compare results

### Output Analysis Checklist
- ✅ **Check `results/plots/`**: Visual performance overview
- ✅ **Read `comprehensive_report.html`**: Executive summary
- ✅ **Review `analysis/` JSON**: Statistical significance
- ✅ **Examine `parsed/` CSV**: Raw decision patterns

## 🎯 Common Research Scenarios

### Scenario 1: "Does this LLM actually learn?"
```
Configuration:
├── ACTIVE_EXPERIMENT = "memory_feeling"
├── ENABLE_STRATEGIC_JOURNAL = True
├── ENABLE_FEELING_LOG = True
├── SHOW_DATE_TO_LLM = False  # Clean learning assessment

Expected Outputs:
├── Strategic journal evolution over time
├── Decision pattern changes after wins/losses
├── Emotional state correlation with performance

Analysis Focus:
├── Learning curve progression
├── Adaptation to market conditions
├── Self-correction capabilities
```

### Scenario 2: "Is there unfair data leakage?"
```
Configuration:
├── Compare: dates_only vs memory_only
├── ENABLE_TECHNICAL_INDICATORS = True (both)
├── SHOW_DATE_TO_LLM = True (dates_only) / False (memory_only)

Expected Outputs:
├── Performance comparison charts
├── Calendar-aware decision patterns
├── Data leakage statistical analysis

Analysis Focus:
├── Performance jumps at known events
├── Unfair advantage quantification
├── Clean vs contaminated learning comparison
```

### Scenario 3: "How sensitive are results to parameters?"
```
Configuration:
├── ACTIVE_EXPERIMENT = "memory_feeling"
├── Vary: MA20_WINDOW = [10, 20, 50]
├── Vary: RET_5D_WINDOW = [3, 5, 10]
├── ENABLE_TECHNICAL_INDICATORS = True

Expected Outputs:
├── Parameter sensitivity analysis
├── Sharpe ratio comparison across settings
├── Risk metric stability assessment

Analysis Focus:
├── Robustness to configuration changes
├── Optimal parameter identification
├── Strategy consistency evaluation
```

### Scenario 4: "How does LLM compare to traditional strategies?"
```
Configuration:
├── Any experiment type
├── USE_DUMMY_MODEL = False
├── Multiple models in LLM_MODELS
├── TEST_MODE = False (full analysis)

Expected Outputs:
├── Baseline comparison plots
├── Risk-adjusted return analysis
├── Statistical significance vs benchmarks

Analysis Focus:
├── Outperformance quantification
├── Risk profile comparison
├── Model capability ranking
```

## 🔍 Troubleshooting Common Issues

### "Results seem too good to be true"
- **Check**: `SHOW_DATE_TO_LLM = True` → Possible data leakage
- **Solution**: Compare with `dates_only` vs date-free experiments

### "No learning visible in results"
- **Check**: `ENABLE_STRATEGIC_JOURNAL = False`
- **Solution**: Switch to `memory_only` or `memory_feeling`

### "Want to test quickly"
- **Check**: `TEST_MODE = False` (running full dataset)
- **Solution**: Set `TEST_MODE = True` and `TEST_LIMIT = 100`

### "LLM not using technical indicators"
- **Check**: `ENABLE_TECHNICAL_INDICATORS = False`
- **Solution**: Set to `True` and compare results

---

## 📋 Quick Reference

**For Learning Research**: Use `memory_feeling` with `ENABLE_STRATEGIC_JOURNAL = True`

**For Calendar Effects**: Use `dates_only` with caution ⚠️ (potential data leakage)

**For Baseline Comparison**: Any experiment type with `USE_DUMMY_MODEL = False`

**For Parameter Sensitivity**: Vary `MA20_WINDOW`, `RET_5D_WINDOW`, `VOL20_WINDOW`

**For Publication Quality**: `TEST_MODE = False` + multiple models + full statistical analysis

This guide helps you design experiments that answer your specific research questions while understanding exactly what outputs and insights you'll get from each configuration.
