# Research Methodology Guide

This document details the statistical and methodological framework used to evaluate LLM trading strategies.

## 📊 Statistical Validation Framework

### Bootstrap Significance Testing

#### Purpose
Bootstrap testing provides robust statistical inference for comparing LLM strategy performance against benchmarks, accounting for the non-normal distributions typical in financial returns.

#### Methodology
```python
def bootstrap_sharpe_comparison(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 10000
) -> Dict
```

**Algorithm:**
1. Calculate observed Sharpe ratios for strategy and benchmark
2. Generate `n_bootstrap` resamples (default: 10,000)
3. For each resample:
   - Sample with replacement from original data (paired bootstrap)
   - Recalculate Sharpe ratios
   - Compute difference: strategy_sharpe - benchmark_sharpe
4. Construct confidence intervals from bootstrap distribution
5. Perform two-sided and one-sided hypothesis tests

#### Statistical Outputs
- **p-value (two-sided)**: Probability of observing difference by chance
- **p-value (strategy better)**: One-sided test for outperformance
- **p-value (strategy worse)**: One-sided test for underperformance
- **95% confidence interval**: Bootstrap CI for Sharpe difference
- **Effect size**: Cohen's d normalized by bootstrap standard deviation

#### Interpretation Guidelines
- **p < 0.05**: Statistically significant difference
- **Effect size > 0.5**: Moderate practical significance
- **Effect size > 0.8**: Large practical significance

### Out-of-Sample Validation

#### Purpose
Detect overfitting by evaluating performance on unseen future data.

#### Methodology
**Temporal Split:**
- **Training period**: First 70% of data (chronological)
- **Test period**: Remaining 30% (future unseen data)

**Validation Metrics:**
- **Sharpe ratio decay**: (train_sharpe - test_sharpe) / train_sharpe
- **Return decay**: (train_return - test_return) / |train_return|
- **Overfitting indicators**:
  - Sharpe decay > 50%
  - Return decay > 70%
  - Test Sharpe < -0.5 (very poor performance)

#### Overfitting Detection
```python
overfitting_indicators = {
    "sharpe_decay_severe": sharpe_decay > 0.5,
    "return_decay_severe": return_decay > 0.7,
    "test_performance_poor": test_sharpe < -0.5,
    "high_train_overfitting": train_sharpe > 1.0 and test_sharpe < 0.0
}
```

### HOLD Decision Evaluation

#### Dual-Criteria Assessment
HOLD positions are evaluated using two complementary criteria:

#### 1. Quiet Market Success
**Definition**: Market movement within ±0.2% threshold
**Rationale**: In quiet markets, cash performs similarly to any directional bet
**Scoring**: Binary (1 if quiet, 0 if volatile)

#### 2. Risk Avoidance Success
**Definition**: HOLD prevents significant losses (>2%) that directional bets would incur
**Logic**: Simulates BUY vs SELL performance and awards credit when HOLD avoids meaningful losses
**Threshold**: Requires avoiding losses >2% (increased from 0.5% for genuine risk management)
**Scoring**:
- 1.0: Avoided significant loss (>2%) in volatile conditions
- 0.5: Reasonable performance despite volatility
- 0.0: No risk avoidance benefit

#### Combined Evaluation
```python
overall_score = (0.6 × quiet_success) + (0.4 × risk_avoidance_score)
success_threshold = 0.4  # HOLD successful if score >= 0.4
```

**Win Rate Calculation**: HOLD decisions are considered successful when the combined score meets the success threshold, providing meaningful calibration analysis instead of always returning 0% win rates.

## 🤖 Behavioral Analysis Framework

The framework includes systematic behavioral pattern detection to assess whether LLMs manifest human-like trading biases:

### Decision Pattern Analysis
Examines how LLMs respond to prior outcomes and tests for systematic behavioral tendencies:

**Win/Loss Response Patterns:**
```python
df["previous_outcome"] = df["previous_return"].apply(
    lambda x: "win" if x > 0 else ("loss" if x < 0 else "neutral")
)
```

**Behavioral Metrics:**
- Decision distribution after wins vs. losses (BUY/HOLD/SELL percentages)
- Confidence modulation following different outcomes
- Statistical significance of behavioral differences

### Confidence Calibration Assessment
Evaluates whether LLM probability estimates align with actual outcome frequencies:

**Calibration Metrics:**
- **Expected Calibration Error (ECE)**: Mean absolute difference between predicted and actual win rates
- **Maximum Calibration Error**: Largest calibration deviation
- **Overconfidence Assessment**: Systematic over/under-estimation patterns

### Market Regime Adaptation
Assesses LLM performance consistency across different market conditions:

**Regime Classification:**
- Low volatility: Below median 20-day volatility
- Moderate volatility: Median to 75th percentile
- High volatility: Above 75th percentile

**Performance by Regime:**
- Risk-adjusted returns within each volatility regime
- Win rates across market conditions
- Adaptation patterns during market stress

## 📈 Performance Metrics

### Primary Metrics

#### Sharpe Ratio
```python
sharpe = (mean_return / volatility) × √252  # Annualized
```
**Interpretation**: Risk-adjusted returns (higher = better)

#### Win Rate
```python
win_rate = (returns > 0).mean()
```
**Limitation**: Always 0% for HOLD positions by construction

#### Maximum Drawdown
```python
max_drawdown = min(cumulative_returns - running_maximum)
```
**Interpretation**: Worst peak-to-trough decline

### Decision-Specific Analysis

#### BUY Decisions
- **Win Rate**: (strategy_return > 0).mean()
- **Expected Return**: strategy_return.mean()
- **Calibration**: predicted_probability vs actual_win_rate

#### HOLD Decisions
- **Quiet Market Success**: Within ±0.2% threshold
- **Context Appropriateness**: High volatility/regime change conditions
- **Risk Management Score**: Combined dual-criteria assessment

#### SELL Decisions
- **Win Rate**: (strategy_return > 0).mean()
- **Expected Return**: strategy_return.mean()
- **Calibration**: predicted_probability vs actual_win_rate

## 🎯 Experimental Design

### Research Questions Addressed

#### Memory and Adaptation
- **RQ1**: Do LLMs improve decisions with historical context?
- **RQ2**: Can LLMs learn from performance feedback?
- **RQ3**: Does emotional self-reflection enhance decision quality?

#### Calibration and Overconfidence
- **RQ4**: Do LLM confidence levels match actual performance?
- **RQ5**: How does calibration vary across decision types?
- **RQ6**: Does calibration improve with experience?

#### Behavioral Patterns
- **RQ7**: Do LLMs exhibit human-like trading biases?
- **RQ8**: How do LLMs respond to loss sequences?
- **RQ9**: Are HOLD decisions contextually appropriate?

### Controlled Experiment Design

#### Independent Variables
- **Memory Context**: None, Journal, Journal + Feelings, Optional Full Trading History
- **Temporal Awareness**: Anonymized vs Real Dates
- **Technical Indicators**: RSI, MACD, Stochastic, Bollinger Bands (configurable inclusion)
- **LLM Architecture**: Different model families and sizes

#### Data Leakage Controls
**Critical Methodological Safeguard**: Date anonymization prevents LLMs from leveraging pre-trained historical knowledge.

**Contamination Sources:**
- **Event Recognition**: Models might identify major crises (2008 GFC, 2020 COVID)
- **Policy Awareness**: Knowledge of Federal Reserve actions, interest rate changes
- **Market Memory**: Recognition of bull/bear market periods from training data
- **Calendar Effects**: Holiday, earnings season, or monthly patterns

**Validity Protection:**
- **Anonymized experiments first**: Establish genuine skill before testing with dates
- **Contamination control**: Ensures results reflect algorithmic trading logic
- **Research integrity**: Prevents false positives from historical pattern matching

#### Dependent Variables
- **Risk-adjusted returns** (Sharpe ratio)
- **Win rates** by decision type
- **Calibration accuracy**
- **HOLD appropriateness**
- **Behavioral pattern consistency**

#### Control Variables
- **Data period**: Fixed historical window
- **Technical features**: Consistent feature engineering
- **Position sizing**: Fixed rules
- **Transaction costs**: Excluded for baseline comparison
- **Date anonymization**: Prevents historical knowledge contamination (critical for validity)

## 🧪 Data and Preprocessing

### Data Source
- **Provider**: Stooq Financial Data
- **Asset**: S&P 500 Index (^GSPC)
- **Period**: Configurable date range (default: 2015-2023)
- **Frequency**: Daily trading data

### Feature Engineering
The implementation includes comprehensive technical indicators alongside basic price-based and momentum features:

**Core Technical Indicators:**
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100 scale)
- **MACD (Moving Average Convergence Divergence)**: Trend momentum with signal line and histogram
- **Stochastic Oscillator**: Momentum timing indicator (0-100 scale)
- **Bollinger Bands**: Volatility-based support/resistance levels

**Technical Memory Integration:**
- **Historical Series**: 20-day lagged values for pattern recognition and trend analysis
- **Aggregated Memory**: Weekly/monthly summaries with technical statistics (averages, percentages, ranges)
- **Multi-Timeframe Analysis**: Enables correlation of technical signals across daily, weekly, and monthly periods

```python
# Daily returns
df["return_1d"] = df["close"].pct_change() * 100.0

# Lagged returns (past daily performance)
for k in range(1, PAST_RET_LAGS + 1):  # PAST_RET_LAGS = 20
    df[f"ret_lag_{k}"] = df["return_1d"].shift(k)

# Momentum indicators
df["ma20_pct"] = df["return_1d"].rolling(MA20_WINDOW).sum()  # 20-day cum. return
df["ret_5d"] = df["return_1d"].rolling(RET_5D_WINDOW).sum()  # 5-day momentum

# Risk metrics
daily_vol_20 = df["return_1d"].rolling(VOL20_WINDOW).std()
df["vol20_annualized"] = daily_vol_20 * np.sqrt(252)  # Annualized volatility

# Advanced Technical Indicators (always calculated)
df["rsi_14"] = compute_rsi(df["close"], RSI_WINDOW)
macd_line, macd_signal, macd_hist = compute_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
df["macd_line"] = macd_line
df["macd_signal"] = macd_signal
df["macd_histogram"] = macd_hist
stoch_k, stoch_d = compute_stochastic(df["high"], df["low"], df["close"], STOCH_K, STOCH_D, STOCH_SMOOTH_K)
df["stoch_k"] = stoch_k
df["stoch_d"] = stoch_d
bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df["close"], BB_WINDOW, BB_STD)
df["bb_upper"] = bb_upper
df["bb_middle"] = bb_middle
df["bb_lower"] = bb_lower
df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

# Target variable
df["next_return_1d"] = df["return_1d"].shift(-1)  # Next day's return
```

### Data Quality Checks
- **Missing data**: Remove rows with NaN values (df.dropna())
- **Data validation**: Ensure close prices are not all NaN
- **Chronological sorting**: Sort data by date
- **Feature completeness**: Drop incomplete feature rows

## 📋 Implementation Details

### Software Architecture
The framework implements a modular Python architecture with specialized modules for each stage of the evaluation pipeline:

- **`main.py`**: Orchestrates the complete evaluation pipeline from data to results
- **`data_prep.py`**: Loads Stooq CSV data, computes comprehensive technical indicators (RSI, MACD, Stochastic, Bollinger Bands)
- **`prompts.py`**: Generates context-rich prompts with hierarchical memory integration
- **`trading_engine.py`**: Manages OpenRouter API calls and temporal memory systems
- **`backtest.py`**: Calculates Sharpe ratios, drawdowns, and equity curves
- **`statistical_validation.py`**: Implements bootstrap testing with 10,000 resamples
- **`decision_analysis.py`**: Detects behavioral biases, confidence calibration, and market regime analysis
- **`reporting.py`**: Generates matplotlib visualizations and markdown reports

### Position Management
- **BUY**: +1.0 (long position)
- **HOLD**: 0.0 (cash position)
- **SELL**: -1.0 (short position)

### Return Calculation
```python
strategy_return = position × next_day_return
```

### Risk Management
- **No leverage**: Positions limited to ±1.0
- **No margin calls**: Simplified assumption
- **No transaction costs**: Baseline comparison
- **Daily rebalancing**: End-of-day position changes

## 🎯 Research Rigor Standards

### Reproducibility
- **Version control**: Complete codebase versioning
- **Random seeds**: Fixed seeds available for key experiments
- **Environment**: Python dependencies specified in requirements.txt
- **Documentation**: Methodological framework documented

### Statistical Validation
- **Bootstrap testing**: 5,000-10,000 resamples for significance testing
- **Out-of-sample validation**: Temporal split testing for overfitting
- **Confidence intervals**: 95% confidence levels
- **HOLD evaluation**: Dual-criteria assessment framework

### Experimental Controls
- **Multiple baselines**: 7 quantitative strategies for comparison
- **Controlled configurations**: 6 standardized experiment setups
- **Data integrity**: Consistent preprocessing across experiments
- **Position management**: Standardized trading rules

This methodology ensures rigorous, reproducible evaluation of LLM financial decision-making capabilities.
