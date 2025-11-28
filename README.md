# 🤖 Large Language Models in Financial Decision-Making

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Research](https://img.shields.io/badge/Type-Research_Framework-purple.svg)](#research-questions)

**An empirical framework for evaluating AI agents in financial markets**

This research framework provides rigorous methodological tools to assess Large Language Model performance against established quantitative trading strategies. The framework enables systematic investigation of memory adaptation, probabilistic calibration, and behavioral patterns in AI-driven financial decision-making.

---

## 🎯 What This Does

### The Trading Process
Each day, the LLM analyzes market data and **chooses one action**:
- **BUY**: Take a long position (+1.0) - profit if market goes up
- **HOLD**: Stay in cash (0.0) - no profit/loss regardless of market movement
- **SELL**: Take a short position (-1.0) - profit if market goes down

The LLM receives technical indicators, can maintain a strategic journal, and may express emotional states. Performance is rigorously compared against traditional quantitative strategies.

### Research Capabilities
- **🧪 Compare LLMs vs Traditional Strategies**: Test any LLM against momentum, mean-reversion, volatility timing, and other proven approaches
- **🧠 Memory & Adaptation**: Study how LLMs learn from experience with strategic journals, feeling logs, and optional complete trading history
- **🎯 Calibration Analysis**: Assess if LLM confidence matches actual performance
- **🧮 Behavioral Biases**: Detect human-like trading biases in AI decisions
- **📊 Statistical Rigor**: Bootstrap testing, out-of-sample validation, dual-criteria HOLD evaluation

---

## 🔬 Research Objectives

### Memory and Learning Dynamics
**Investigation of temporal learning in sequential financial decisions**
- Analysis of historical context integration in LLM decision processes
- Assessment of adaptive behavior based on performance feedback
- Evaluation of emotional state impacts on decision quality

### Probabilistic Calibration
**Assessment of confidence-outcome alignment in AI predictions**
- Measurement of overconfidence/underconfidence patterns
- Decision-specific calibration analysis (BUY/SELL/HOLD)
- Longitudinal calibration stability assessment

### Behavioral Pattern Analysis
**Detection of systematic biases in AI trading behavior**
- Loss aversion quantification in position management
- Disposition effect identification in profit-taking behavior
- Risk management appropriateness in uncertain conditions

---

## 📊 At a Glance

| Component | Status | Description |
|-----------|--------|-------------|
| **LLM Integration** | ✅ OpenRouter API | BERT, GPT, Claude, Gemini models |
| **Baseline Strategies** | ✅ 7 Strategies | Momentum, Mean Reversion, Volatility Timing |
| **Statistical Validation** | ✅ Bootstrap Testing | Out-of-sample, significance testing |
| **Decision Analysis** | ✅ Dual-Criteria | Calibration + behavioral pattern analysis |
| **Reporting** | ✅ Comprehensive | Automated research reports |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scipy matplotlib seaborn requests
```

### Quick Test Run
```bash
# Run with dummy model (no API key needed)
python -m src.main
```

### Real LLM Experiment
```bash
# 1. Get OpenRouter API key from https://openrouter.ai/
#    See documentation: https://openrouter.ai/docs
export OPENROUTER_API_KEY="your_key_here"

# 2. Configure your experiment in src/config.py
TEST_MODE = False  # Set to False for full dataset (default: ~2700 days)
# Or customize: DAYS_TO_RUN = 100  # If keeping TEST_MODE = True

# 3. Choose LLM model
ACTIVE_MODEL = "bert"  # or other models

# 4. Run experiment
python -m src.main
```

**⚠️ Important**: Set `TEST_MODE = False` in `src/config.py` for meaningful results, or adjust `DAYS_TO_RUN` if keeping test mode.

## 📊 Sample Results (Dummy Model)

*These are test results using a dummy model - run with real LLMs for actual research!*

### Strategy Performance Comparison
![Baseline Comparison](example_results/plots/dummy_model_memory_feeling_baseline_comparison.png)
*LLM strategies vs traditional approaches (dummy data for illustration)*

### Decision Calibration Analysis
![Calibration by Decision](example_results/plots/dummy_model_memory_feeling_calibration_by_decision.png)
*Confidence vs actual performance across BUY/HOLD/SELL decisions*

### Behavioral Pattern Analysis
![Decision Patterns](example_results/plots/dummy_model_memory_feeling_decision_patterns.png)
*Decision changes after wins vs losses - evidence of learning/adaptation*

## 📈 Data Source

**Stock data provided by Stooq**: Access financial data at [stooq.com](https://stooq.com/)

The framework is designed to work with any daily stock/index data. Modify `src/data_prep.py` to integrate your preferred data source.

---

---

## 🏗️ Architecture

```
📁 src/
├── 🎯 main.py                 # Pipeline orchestrator
├── 🤖 openrouter_model.py     # LLM API integration (BERT primary)
├── 📊 backtest.py            # Strategy evaluation engine
├── 🧮 baselines.py           # 7 traditional quantitative strategies
├── 📈 statistical_validation.py # Bootstrap testing + out-of-sample
├── 📋 report_generator.py    # Automated research reports
└── ⚙️ config.py              # 6 experimental configurations

📁 results/
├── 📊 analysis/              # Statistical validation JSON/CSV
├── 📈 plots/                 # Calibration, patterns, risk analysis
├── 📋 reports/               # Comprehensive experiment reports
└── 📄 parsed/                # Processed trading decisions
```

---

## 📈 Trading Strategies

### 🤖 LLM Strategy Configurations

| Configuration | Memory | Feelings | Dates | Purpose |
|---------------|--------|----------|-------|---------|
| **baseline** | ❌ | ❌ | ❌ | Pure LLM without context |
| **memory_only** | ✅ | ❌ | ❌ | Strategic journal learning |
| **memory_feeling** | ✅ | ✅ | ❌ | Emotional state tracking |
| **dates_memory** | ✅ | ❌ | ✅ | Temporal awareness ⚠️ |
| **dates_full** | ✅ | ✅ | ✅ | Complete context ⚠️ |

**⚠️ Research Note**: Date-enabled configurations shouldn't be used if you wan to prevent data leakage from historical knowledge contamination.

### 🎯 Baseline Strategies
| Strategy | Logic | Parameters | 2015-2017 Return |
|----------|-------|------------|------------------|
| **Buy & Hold** | Passive index | - | +14.2% |
| **Momentum** | Trend following | 20-day MA | +5.1% |
| **Mean Reversion** | Buy dips | ±2σ thresholds | +5.5% |
| **Volatility Timing** | Risk management | 20% vol threshold | +11.1% |
| **Contrarian** | Fade extremes | Inverse signals | +9.1% |
| **Random** | Statistical baseline | Equal weights | -0.03% |

---

## 📊 Statistical Validation Framework

### Bootstrap Significance Testing
```python
# Rigorous statistical comparison
bootstrap_results = bootstrap_sharpe_comparison(
    llm_returns, benchmark_returns, n_bootstrap=5000
)

print(f"p-value: {bootstrap_results['p_value_two_sided']:.4f}")
print(f"95% CI: [{bootstrap_results['ci_95_bootstrap'][0]:+.3f}, {bootstrap_results['ci_95_bootstrap'][1]:+.3f}]")
```

**Key Features:**
- ✅ 5,000 bootstrap resamples for robust significance testing
- ✅ Confidence intervals account for return distribution non-normality
- ✅ Effect sizes (Cohen's d) for practical significance
- ✅ Multiple testing corrections

### Out-of-Sample Validation
```
OUT-OF-SAMPLE VALIDATION:
  Train Period: -0.109 Sharpe (350 periods: 2015-2016)
  Test Period:  1.145 Sharpe (150 periods: 2016-2017)
  Sharpe Decay: +1150.5% 🚨 OVERFITTING DETECTED
```

### HOLD Decision Analysis (Dual-Criteria)
```
HOLD DECISION ANALYSIS:
Overall Success Rate: 40.4% (POOR)

Quiet Market Success (<0.2% moves): 0.6% ✅ Appropriate caution
Contextual Correctness: 100.0% ✅ Right conditions
Combined Score: 40.4% ⚠️ Too conservative overall
```

---

## 📋 Sample Results

### Performance Comparison
```
STRATEGY COMPARISON - dummy_model_memory_only
================================================================================
Strategy                 Return    Sharpe   MaxDD   Win%
buy_and_hold             132.0%    0.686   -33.9%   54.2
momentum                 96.6%     0.857   -13.2%   37.4
LLM_STRATEGY             -25.0%    -0.165  -45.0%   48.0 ◄
random_mean (n=30)       -48.2%     N/A     N/A     N/A
================================================================================
```

### Statistical Significance
```
BOOTSTRAP TEST VS INDEX:
  Strategy Sharpe: -0.275
  Index Sharpe: 0.518
  Difference: -0.793
  p-value: 0.023 (SIGNIFICANT)
  95% CI: [-2.806, +1.089]
```

### HOLD Analysis
```
HOLD DECISION ANALYSIS
Overall HOLD Success Rate: 40.4% (POOR)

Quiet Market Success (<0.2% Daily Moves): 0.6%
Contextual Decision Correctness: 100.0%
```

---

## 🎨 Generated Visualizations

- **📊 Baseline Comparisons**: Bar charts, equity curves
- **🎯 Calibration Plots**: Prediction accuracy, confidence analysis
- **📈 Risk Analysis**: VaR curves, drawdown timelines, stress tests
- **📋 Decision Patterns**: Win rates by scenario, position duration
- **📉 Rolling Performance**: Sharpe evolution, volatility tracking

---

## 🔬 Methodology Details

### Experimental Design
- **Data**: Daily stock/index prices (customizable period and source)
- **LLM Models**: BERT, GPT, Claude, Gemini via OpenRouter API
- **Evaluation**: Bootstrap significance testing, out-of-sample validation
- **Backtesting**: Configurable assumptions and transaction costs

### Statistical Framework
- **Primary Metric**: Risk-adjusted returns (Sharpe ratio)
- **Significance Level**: p < 0.05 (bootstrap tests)
- **Validation**: Chronological train/test splits
- **HOLD Evaluation**: Dual-criteria assessment (quiet markets + context)

### Decision Analysis Pipeline
1. **LLM Response Parsing** → Decision + Probability + Explanation + Journal
2. **Calibration Assessment** → Confidence vs Reality analysis
3. **Pattern Recognition** → Behavioral pattern detection
4. **HOLD Evaluation** → Dual-criteria success assessment
5. **Statistical Validation** → Bootstrap significance testing

---

## 📖 Usage Examples

### Run Complete Experiment
```bash
# Configure in src/config.py
ACTIVE_EXPERIMENT = "memory_only"  # or "baseline", "dates_memory", etc.

# Run pipeline
python -m src.main
```

### Analyze Specific Model
```python
from src.report_generator import generate_comprehensive_report

# Generate full research report
report_path = generate_comprehensive_report("bert_memory_only")
print(f"Report saved: {report_path}")
```

### Statistical Validation
```python
from src.statistical_validation import comprehensive_statistical_validation

# Run full validation suite
validation = comprehensive_statistical_validation(parsed_df, model_tag)
print(f"LLM vs Index p-value: {validation['bootstrap_vs_index']['p_value']}")
```

---



## 🔬 Future Research Directions

### Immediate Extensions
- **HOLD Decision Investigation**: Investigate 100% context success rate in HOLD decisions, even with random models
- **Technical Indicators**: Implement RSI, MACD, and other momentum oscillators for richer market signals
- **Enhanced Feature Engineering**: Improve prompt engineering with new technical indicators and market regime detection
- **Cross-Model Comparison**: BERT vs GPT vs Claude calibration differences
- **Market Regime Analysis**: Calibration stability across bull/bear cycles


### Long-Term Questions
- **AI Behavioral Training**: Can we train LLMs to avoid human cognitive biases?
- **Hybrid Systems**: Combining LLM intuition with quantitative risk management
- **Real-Time Adaptation**: Online learning from live trading feedback
- **Multi-Asset Extension**: Beyond single-stock to portfolio optimization

---

## 📖 Documentation

- **[Configuration Guide](docs/configuration.md)**: Complete setup and configuration options
- **[Research Methodology](docs/methodology.md)**: Statistical framework and experimental design
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute to the project
- **[Strategic Journal Feature](docs/STRATEGIC_JOURNAL_FEATURE.md)**: Memory system implementation details

## 🤝 Contributing to Research

We welcome contributions from researchers, developers, and financial practitioners. See our detailed [Contributing Guide](CONTRIBUTING.md) for comprehensive information.

### Quick Start for Contributors
- **📖 Read the [Contributing Guide](CONTRIBUTING.md)** for detailed instructions
- **🔬 Research**: Test new LLMs, develop strategies, enhance statistical methods
- **💻 Code**: Bug fixes, features, performance improvements
- **🧪 Testing**: Add test coverage, validate results
- **📚 Docs**: Improve documentation and tutorials

### Research Opportunities
- **Model Benchmarking**: Compare BERT, Claude, Gemini, and emerging LLMs
- **Strategy Development**: New baseline strategies and prompting techniques
- **Market Analysis**: Test across different markets and time periods
- **Behavioral Research**: Develop new AI bias detection methods

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenRouter** for LLM API access
- **Stooq** for providing historical financial data

---

This framework provides methodological tools for systematic investigation of artificial intelligence in financial decision-making, integrating computational methods with behavioral economics and quantitative finance.
