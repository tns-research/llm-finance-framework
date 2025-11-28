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

#### 2. Contextual Decision Correctness
**High Volatility Context**: >75th percentile of 20-day volatility
**Regime Change**: Significant 10-day trend direction shifts (>0.1% change)
**Decision Uncertainty**: Mixed recent decisions (>60% entropy)
**Extreme Conditions**: Large recent moves (>3% in 3 days)

**Scoring**: Weighted combination of contextual factors (0-2.0 scale)

#### Combined Evaluation
```python
overall_score = (0.6 × quiet_success) + (0.4 × context_correctness)
```

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
The current implementation includes basic price-based and momentum features:

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

# Target variable
df["next_return_1d"] = df["return_1d"].shift(-1)  # Next day's return
```

**Note**: Advanced technical indicators (RSI, MACD, Bollinger Bands) are not currently implemented.

### Data Quality Checks
- **Missing data**: Remove rows with NaN values (df.dropna())
- **Data validation**: Ensure close prices are not all NaN
- **Chronological sorting**: Sort data by date
- **Feature completeness**: Drop incomplete feature rows

## 📋 Implementation Details

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
