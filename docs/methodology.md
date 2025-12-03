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

## 🎯 LLM Prompt Architecture & Hierarchical Memory System

### Prompt Construction Process

The framework builds prompts through a systematic 4-stage process that combines static context with dynamic, hierarchical memory layers.

#### Stage 1: System Prompt Assembly
**Location**: `src/config.py` - `SYSTEM_PROMPT` variable
**Purpose**: Establishes LLM role and provides invariant context

```python
SYSTEM_PROMPT = """
You are a cautious but rational equity index hedge fund trader. Your role is to beat the S&P500.

Your task is to decide a trading action for the S and P 500 index for the next trading day based only on the information provided in the user message.

Technical indicators available include:
- 20-day moving average momentum (trend strength)
- 20-day annualized volatility (risk measure)
- 5-day recent momentum (short-term trend)
- 14-day Relative Strength Index (RSI) - momentum oscillator ranging from 0-100
- MACD(12,26,9) - Moving Average Convergence Divergence with histogram
- Stochastic Oscillator(14,3) - momentum indicator ranging from 0-100
- Bollinger Bands(20,2) - volatility bands showing price extremes

Rules for decision making:
1) Use only the information in the input. Do not use any knowledge about what happens after the input date.
2) RSI measures momentum from 0-100, with >70 overbought and <30 oversold - look for divergences and reversals.
[Additional rules for MACD, Stochastic, Bollinger Bands, decision format...]
"""
```

#### Stage 2: User Message Construction
**Location**: `src/trading_engine.py` - `run_single_model()` function
**Process**: Combines multiple data sources into coherent prompt structure

```python
# Stage 2.1: Base Market Data
base_prompt = row["prompt_text"]  # Contains technical indicators and market summary

# Stage 2.2: Strategic Journal (Memory Layer 1)
journal_text = journal_manager.get_journal_block(current_date, SHOW_DATE_TO_LLM, ENABLE_TECHNICAL_INDICATORS)

# Stage 2.3: Performance Summary (Memory Layer 2)
performance_summary = performance_tracker.get_performance_summary()

# Stage 2.4: Memory Blocks (Memory Layer 3)
memory_blocks = memory_manager.get_all_memory_blocks()

# Stage 2.5: Trading History (Memory Layer 4)
trading_history_block = trade_history_manager.get_history_block(SHOW_DATE_TO_LLM, ENABLE_FULL_TRADING_HISTORY)

# Stage 2.6: Final User Message Assembly
user_prompt = build_final_user_prompt(
    base_prompt, journal_text, performance_summary,
    memory_blocks, trading_history_block, ENABLE_STRATEGIC_JOURNAL
)
```

### Hierarchical Memory Layer Details

##### Stage 2.1: Raw Technical Data Layer 📊
**Source**: `src/data_prep.py` - Processed DataFrame columns
**Construction**: Static prompt template with dynamic data insertion

```
Past 20 daily returns in percent, most recent last
{returns_list}  # df["return_1d"].tail(20).round(2).tolist()

Past 20 days RSI(14) values, most recent last
{rsi_list}  # df["rsi_14"].tail(20).round(1).tolist()

Past 20 days MACD histogram values, most recent last
{macd_list}  # df["macd_histogram"].tail(20).round(3).tolist()

Past 20 days Stochastic %K values, most recent last
{stoch_list}  # df["stoch_k"].tail(20).round(1).tolist()

Past 20 days Bollinger Band positions (0=lower, 1=upper), most recent last
{bb_list}  # df["bb_position"].tail(20).round(2).tolist()

20 day total return  {total_return_20d:.2f} percent
20 day realized volatility  {vol_20d:.2f} percent annualized
5 day total return  {total_return_5d:.2f} percent

RSI(14)  {current_rsi:.2f}
MACD(12,26,9) Line: {macd_line:.3f}, Signal: {macd_signal:.3f}, Histogram: {macd_hist:.3f}
Stochastic(14,3) %K: {stoch_k:.2f}, %D: {stoch_d:.2f}
Bollinger Bands(20,2) Upper: {bb_upper:.2f}, Middle: {bb_middle:.2f}, Lower: {bb_lower:.2f}

Summary
The index has a {trend_desc} 20 day total return and {vol_desc} volatility.
Recent 5 day return is {return_5d:.2f} percent.
RSI(14) is {rsi_desc} (current_rsi).
MACD histogram is {macd_desc} (macd_hist).
Stochastic %K is {stoch_desc} (stoch_k).
Price is at {bb_desc} of Bollinger Bands.
```

**Technical Implementation**:
- **Data Extraction**: `df[col].tail(20).round(precision).tolist()` for time series
- **Current Values**: `df[col].iloc[-1]` for latest indicators
- **Text Generation**: Conditional logic for qualitative descriptions ("neutral", "bearish", etc.)
- **Memory Efficiency**: Raw numerical data minimizes token usage while preserving information

**Purpose**: Provides quantitative foundation for LLM pattern recognition and technical analysis

##### Stage 2.2: Strategic Journal Layer 🧠 (Short-term Memory)
**Source**: `src/journal_manager.py` - JournalManager.get_journal_block()
**Construction**: Rolling window of LLM's recent trading decisions and reasoning

**Data Structure**:
```python
# Each journal entry contains:
{
    "date": datetime,           # Trade date
    "decision": str,           # BUY/HOLD/SELL
    "prob": float,             # Confidence probability
    "next_return_1d": float,   # Next day's market return
    "strategy_return": float,  # Strategy's return for this trade
    "cumulative_return": float,# Running cumulative strategy return
    "index_cumulative_return": float,  # Running cumulative index return
    "explanation": str,        # LLM's decision explanation
    "strategic_journal": str,  # LLM's forward-looking strategy
    "feeling_log": str         # LLM's emotional state
}
```

**Formatting Logic**:
```python
def format_single_entry(self, trade_data, current_date, show_dates, enable_technical_indicators):
    # Date prefix with relative time or absolute date
    if show_dates:
        prefix = f"Date {trade_data['date'].strftime('%Y-%m-%d')}: "
    else:
        relative_time = self.get_relative_time_label(trade_data["date"], current_date)
        prefix = f"{relative_time}: "

    # Core trade information
    base_info = (
        f"action {trade_data['decision']} (prob {trade_data['prob']:.2f}), "
        f"next day index return {trade_data['next_return_1d']:.2f}%, "
        f"strategy return {trade_data['strategy_return']:.2f}%, "
        f"cumulative strategy return {trade_data['cumulative_return']:.2f}%, "
        f"cumulative index return {trade_data['index_cumulative_return']:.2f}%."
    )

    # Technical indicators (optional)
    if enable_technical_indicators and trade_data.get("rsi_14"):
        tech_indicators = self._format_technical_indicators(trade_data)
        base_info += f" Technical indicators: {tech_indicators}."

    # LLM's explanations
    base_info += (
        f" Explanation: {trade_data['explanation']} "
        f"Strategic journal: {trade_data['strategic_journal']} "
        f"Feeling: {trade_data['feeling_log']}"
    )

    return prefix + base_info
```

**Memory Mechanism**:
- **Rolling Window**: Maintains exactly 10 most recent entries
- **Self-Reflection**: LLM reads its own previous reasoning
- **Context Preservation**: Maintains decision consistency across time
- **Emotional Tracking**: Includes LLM's reported emotional states

**Example Output**:
```
Past trades and results so far:
2 weeks ago: action BUY (prob 0.45), next day index return 0.21%, strategy return 0.21%, cumulative strategy return -0.39%, cumulative index return 1.52%. Technical indicators: RSI(14): 46.3 | MACD: 5.57/6.74/-1.175 | Stochastic: 65.7/69.7 | BB Position: 0.42. Explanation: Market indicators suggest upward momentum. Technical analysis shows bullish patterns. Strategic journal: Reviewing recent performance and adjusting strategy accordingly. Current market regime suggests buy positioning. Feeling: Feeling cautiously optimistic about current market conditions.
```

**Purpose**: Enables immediate self-reflection, decision consistency, and adaptive behavior based on recent outcomes

##### Stage 2.3: Performance Summary Layer 📈 (Quantitative Feedback)
**Source**: `src/performance_tracker.py` - PerformanceTracker.get_performance_summary()
**Construction**: Real-time calculation of trading performance metrics

**Tracked Metrics**:
```python
class PerformanceTracker:
    def __init__(self):
        self.cumulative_return = 0.0        # Running strategy return
        self.index_cumulative_return = 0.0  # Running benchmark return
        self.decision_count = 0              # Total decisions made
        self.buy_count = 0                   # BUY decisions
        self.hold_count = 0                  # HOLD decisions
        self.sell_count = 0                  # SELL decisions
        self.win_count = 0                   # Profitable trades (strategy_return > 0)
        self.current_decision = None         # Current position
        self.current_position_duration = 0   # Days in current position
```

**Calculation Logic**:
```python
def update_daily_performance(self, decision: str, daily_return: float, index_return: float):
    # Update position duration tracking
    self._update_position_duration(decision)

    # Accumulate returns
    self.cumulative_return += daily_return
    self.index_cumulative_return += index_return

    # Update decision statistics
    self.decision_count += 1
    self._increment_decision_counter(decision)

    # Track wins (profitable trades)
    if daily_return > 0:
        self.win_count += 1
```

**Formatting Logic**:
```python
def get_performance_summary(self) -> str:
    if self.decision_count == 0:
        return (
            "No trades executed yet.\n"
            "Strategy cumulative return so far  0.00 percent.\n"
            "S and P 500 cumulative return so far  0.00 percent.\n"
            "BUY 0, HOLD 0, SELL 0.\n"
            "Win rate undefined."
        )

    edge = self.cumulative_return - self.index_cumulative_return
    outperform_word = "outperforming" if edge > 0 else "underperforming"
    win_rate_pct = (self.win_count / self.decision_count) * 100.0

    return (
        f"Total strategy return so far  {self.cumulative_return:.2f} percent.\n"
        f"Total S&P 500 return so far  {self.index_cumulative_return:.2f} percent.\n"
        f"You are {outperform_word} the index by {edge:.2f} percent.\n"
        f"Number of decisions so far  {self.decision_count} "
        f"(BUY {self.buy_count}, HOLD {self.hold_count}, SELL {self.sell_count}).\n"
        f"Win rate so far  {win_rate_pct:.1f} percent."
    )
```

**Technical Implementation**:
- **Real-time Updates**: Metrics updated after each trading decision
- **Benchmark Comparison**: Continuous tracking vs S&P 500 performance
- **Decision Analytics**: Breakdown by decision type (BUY/HOLD/SELL)
- **Win Rate Calculation**: Percentage of profitable trades
- **Position Tracking**: Duration of current market position

**Example Output**:
```
Total strategy return so far  0.79 percent.
Total S&P 500 return so far  0.69 percent.
You are outperforming the index by 0.10 percent.
Number of decisions so far  39 (BUY 13, HOLD 11, SELL 15).
Win rate so far  35.9 percent.
```

**Purpose**: Provides quantitative feedback loop for LLM learning and adaptation

##### Stage 2.4: Hierarchical Memory Layer 🏗️ (Long-term Learning)
**Source**: `src/memory_manager.py` & `src/period_manager.py` - LLM-generated period summaries
**Construction**: Separate LLM calls generate reflective summaries for different timeframes

**Generation Process**:
```python
# Weekly Summary Generation (every Friday)
if is_week_boundary:
    weekly_summary = llm_call(
        prompt=f"""
        Analyze the past week's trading performance and generate a strategic summary:

        Week Data:
        - Start Date: {week_start}
        - End Date: {week_end}
        - Market Return: {market_return:.2f}%
        - Strategy Return: {strategy_return:.2f}%
        - Decisions: BUY {buy_count}, HOLD {hold_count}, SELL {sell_count}
        - Win Rate: {win_rate:.1f}%

        Technical Summary:
        - RSI Average: {rsi_avg:.1f}
        - MACD Bullish Periods: {macd_bullish_pct:.0f}%
        - Stochastic Overbought Days: {stoch_overbought:.0f}%
        - Bollinger Band Touches: {bb_touches}

        Generate a 2-3 sentence strategic analysis covering:
        1. Performance assessment
        2. Technical indicator insights
        3. Strategic implications for future trading
        4. Emotional/reflective commentary
        """,
        model=router_model
    )
    memory_manager.store_weekly_summary(week_end, weekly_summary)
```

**Data Structure**:
```python
# MemoryManager stores summaries by period type
class MemoryManager:
    def __init__(self):
        self.weekly_summaries = []    # List of (date, summary) tuples
        self.monthly_summaries = []   # List of (date, summary) tuples
        self.quarterly_summaries = [] # List of (date, summary) tuples
        self.yearly_summaries = []    # List of (date, summary) tuples

    def store_weekly_summary(self, date, summary):
        self.weekly_summaries.append((date, summary))
        # Maintain last 5 weekly summaries
        if len(self.weekly_summaries) > 5:
            self.weekly_summaries.pop(0)
```

**Retrieval Logic**:
```python
def get_all_memory_blocks(self):
    """Return formatted memory blocks for LLM prompt"""
    return {
        'weekly': self._format_weekly_memories(),
        'monthly': self._format_monthly_memories(),
        'quarterly': self._format_quarterly_memories(),
        'yearly': self._format_yearly_memories()
    }

def _format_weekly_memories(self):
    """Format weekly summaries for prompt inclusion"""
    if not self.weekly_summaries:
        return "No weekly summaries yet."

    formatted = ["Weekly memory (most recent first)"]
    for i, (date, summary) in enumerate(reversed(self.weekly_summaries[-5:])):
        weeks_ago = len(self.weekly_summaries) - i - 1
        if weeks_ago == 0:
            label = "1 week ago:"
        else:
            label = f"{weeks_ago + 1} weeks ago:"

        formatted.append(f"{label}")
        formatted.append(f"{summary}")
        formatted.append("")  # Empty line between entries

    return "\n".join(formatted).rstrip()
```

**Technical Implementation**:
- **Separate LLM Calls**: Each period summary generated by dedicated prompt
- **Rolling Windows**: Maintain limited history (5 weekly, 2 monthly, etc.)
- **Date-Based Triggers**: Summaries generated at period boundaries
- **Structured Prompts**: Consistent format for LLM analysis
- **Memory Efficiency**: Summarized insights reduce token usage

**Example Weekly Summary Generation Prompt**:
```
Analyze the past week's trading performance:

Market total return  -1.54 percent.
Strategy total return  0.62 percent.
RSI averaged 47.8 with 0.0% overbought days.
MACD was bullish 0.0% of the time.

Generate strategic analysis covering performance, technical insights,
strategic implications, and reflective commentary.
```

**Example Output**:
```
Weekly memory (most recent first)
1 week ago:
Weekly summary. Market total return -1.54 percent. Strategy total return 0.62 percent.
The strategy outperformed the index by 2.16 percent over 5 days.
RSI bullish signals, MACD showing mostly bearish momentum.
Reflect on whether your positioning matched the prevailing trend and volatility,
and whether your risk management was consistent.
Feeling: Feeling cautiously reflective about this period.
```

**Purpose**: Enables multi-timeframe learning, strategic adaptation, and long-term pattern recognition

##### Stage 2.5: Complete Trading History Layer 📋 (Full Context)
**Source**: `src/trade_history_manager.py` - TradeHistoryManager.get_history_block()
**Construction**: CSV-formatted complete chronological trading record

**Data Structure**:
```python
# Each entry contains complete trade information
trade_entry = {
    "date": "2023-01-01",        # Trade date (YYYY-MM-DD)
    "trade_id": 1,               # Sequential trade identifier
    "decision": "BUY",           # LLM decision (BUY/HOLD/SELL)
    "position": 1.0,             # Position value (-1.0, 0.0, 1.0)
    "result": 0.923515           # Strategy return for this trade
}
```

**Formatting Logic**:
```python
def get_history_block(self, show_dates: bool, enabled: bool = True) -> str:
    if not enabled or self.is_empty():
        return "TRADING_HISTORY:\nNo trading history yet."

    # Choose column headers based on date visibility
    if show_dates:
        header = "date,decision,position,result"
        rows = [f"{entry['date']},{entry['decision']},{entry['position']},{entry['result']}"
               for entry in self.entries]
    else:
        header = "trade_id,decision,position,result"
        rows = [f"{entry['trade_id']},{entry['decision']},{entry['position']},{entry['result']}"
               for entry in self.entries]

    return f"TRADING_HISTORY:\n{header}\n" + "\n".join(rows)
```

**Technical Implementation**:
- **Flexible Headers**: `date` field vs `trade_id` based on configuration
- **Complete Chronology**: Every trade preserved in order
- **CSV Format**: Machine-readable for pattern analysis
- **Memory Efficiency**: Numeric data only, minimal token usage
- **Configuration Control**: `ENABLE_FULL_TRADING_HISTORY` toggle

**Example with Dates**:
```
TRADING_HISTORY:
date,decision,position,result
2023-01-01,BUY,1.0,0.923515
2023-01-02,SELL,-1.0,0.148067
2023-01-03,HOLD,0.0,0.0
```

**Example without Dates**:
```
TRADING_HISTORY:
trade_id,decision,position,result
1,BUY,1.0,0.923515
2,SELL,-1.0,0.148067
3,HOLD,0.0,0.0
```

**Privacy Mechanism**:
- **Date Anonymization**: When `SHOW_DATE_TO_LLM=False`, removes temporal patterns
- **Research Protection**: Prevents LLMs from leveraging pre-trained historical knowledge
- **Contamination Control**: Ensures results reflect algorithmic skill, not data leakage

**Purpose**: Provides complete historical context for pattern recognition while maintaining research integrity

---

## Complete Prompt Assembly Process

### Final User Message Construction
**Location**: `src/trading_engine.py` - Lines 150-171
**Integration**: Combines all memory layers into final LLM prompt

```python
def build_final_user_prompt(base_prompt, journal_text, performance_summary,
                          memory_blocks, trading_history_block, enable_strategic_journal):
    """Assemble complete user message for LLM"""

    user_message = base_prompt  # Raw technical data

    if enable_strategic_journal:
        user_message += "\n\nStrategic journal\n" + journal_text

    user_message += "\n\nPerformance summary so far\n" + performance_summary

    # Add memory blocks
    if memory_blocks['weekly']:
        user_message += "\n\n" + memory_blocks['weekly']
    if memory_blocks['monthly']:
        user_message += "\n\n" + memory_blocks['monthly']
    if memory_blocks['quarterly']:
        user_message += "\n\n" + memory_blocks['quarterly']
    if memory_blocks['yearly']:
        user_message += "\n\n" + memory_blocks['yearly']

    # Add trading history
    user_message += "\n\n" + trading_history_block

    return user_message
```

### Memory Layer Integration Order
```
1. 📊 Raw Technical Data     (Foundation - quantitative input)
2. 🧠 Strategic Journal      (Immediate - recent decisions & reasoning)
3. 📈 Performance Summary    (Current - metrics & benchmark comparison)
4. 🏗️ Memory Blocks         (Long-term - period-based reflections)
5. 📋 Trading History       (Complete - full chronological context)
```

### Token Efficiency Considerations
- **Raw Data**: Numerical arrays minimize token usage
- **Structured Summaries**: LLM-generated insights replace raw data
- **Rolling Windows**: Limited history prevents token overflow
- **Conditional Inclusion**: Features can be toggled based on requirements

### Quality Assurance
- **Data Integrity**: Each layer verified independently
- **Prompt Validation**: Complete prompts tested for formatting
- **Memory Consistency**: All layers reference same time periods
- **Research Ethics**: Date anonymization prevents contamination

This hierarchical architecture enables LLMs to maintain sophisticated context across multiple temporal horizons while maintaining research validity and computational efficiency.

### Memory Hierarchy Benefits

This hierarchical architecture provides cognitive scaffolding:

1. **Immediate Context** (Strategic Journal): "What did I just do and why?"
2. **Short-term Learning** (Weekly Memory): "What patterns emerged this week?"
3. **Long-term Adaptation** (Monthly Memory): "How am I performing over months?"
4. **Quantitative Feedback** (Performance Summary): "Am I beating the market?"
5. **Complete Reference** (Trading History): "Full historical context for analysis"

### Technical Implementation

- **Strategic Journal**: Managed by TradeHistoryManager (`journal_entries`)
- **Memory Blocks**: Managed by MemoryManager (LLM-generated period summaries)
- **Performance Summary**: Managed by PerformanceTracker (real-time calculations)
- **Trading History**: Managed by TradeHistoryManager (`trading_history`)

This architecture enables LLMs to maintain consistency, learn from feedback, and adapt strategies across multiple temporal horizons.

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
