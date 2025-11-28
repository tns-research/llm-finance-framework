# LLM Trading Strategy Experiment Report
## Model: dummy_model_memory_feeling | Generated: 2025-11-28 03:30

---

## Data Sources Summary

- Total data sources collected: 5
- statistical_validation: 7 items
- baseline_comparison: 8 items
- calibration_analysis: available
- pattern_analysis: available
- plots: 5 plot files

---

## 📈 Executive Summary

- **Total Return**: 3.53% (vs Index: 14.15%)
- **Period**: 500 trading days
- **Statistical Significance**: ❌ Not significant vs index (p=0.7352)
- **Effect Size**: -0.351 (Cohen's d)
- **Overfitting**: 🚨 Detected

---

## 🎯 Performance Overview

![Baseline Comparison](plots\dummy_model_memory_feeling_baseline_comparison.png)
*Figure 1: Strategy performance vs baseline strategies*

![Equity Curves](plots\dummy_model_memory_feeling_equity_curves.png)
*Figure 2: Equity curves over time*

---

## 📊 Statistical Validation


================================================================================
STATISTICAL VALIDATION REPORT - dummy_model_memory_feeling
================================================================================

Dataset: 500 periods (2015-03-18 00:00:00 to 2017-03-10 00:00:00)
Strategy Return: 3.53% | Index Return: 14.15%

OUT-OF-SAMPLE VALIDATION:
  Train Period: 0.191 Sharpe (350 periods)
  Test Period:  0.011 Sharpe (150 periods)
  Sharpe Decay: +94.2%
  Overfitting Detected: YES
  Generalizes Well: NO

BOOTSTRAP TEST VS INDEX:
  Strategy Sharpe: 0.153
  Index Sharpe: 0.518
  Difference: -0.365
  p-value (two-sided): 0.7352
  95% CI: [-2.394, +1.664]
  RESULT: No significant difference detected 🤔

SUMMARY ASSESSMENT:
  Overall: CONCERNING
  Confidence: LOW
  Key Findings:
    ⚠️  Strategy shows signs of overfitting or poor generalization
    🚨 High overfitting detected
    🤔 Strategy performance vs index is not statistically significant
  Recommendations:
    • Consider model regularization or simpler strategy
================================================================================



---

## 🎪 Decision Analysis

![Calibration Plot](plots\dummy_model_memory_feeling_calibration.png)
*Figure 4: Prediction confidence vs actual performance*

![Calibration by Decision](plots\dummy_model_memory_feeling_calibration_by_decision.png)
*Figure 5: Calibration analysis by decision type (BUY/HOLD/SELL)*

### Detailed Calibration Analysis

```markdown
# Calibration Analysis Report - dummy_model_memory_feeling

## Overview

This report analyzes the calibration quality of the **dummy_model_memory_feeling** model, 
measuring how well predicted probabilities match actual outcomes.

---

## Overall Calibration Metrics

**Total Trading Days:** 500
**Overall Win Rate:** 34.8%
**Mean Predicted Probability:** 65.1%

### Calibration Quality Indicators

- **Expected Calibration Error (ECE):** 30.4%
- **Maximum Calibration Error:** 50.1%
- **Overconfidence Score:** 43.6%

- **Calibration Quality:** **POOR** - Model shows significant calibration problems

### Confidence Assessment

- **Assessment:** **OVERCONFIDENT** - Model tends to be too optimistic about success probability

---

## Calibration by Decision Type

This analysis shows if the model has different calibration characteristics for BUY, HOLD, and SELL decisions.

### BUY Decisions

- **Count:** 169 decisions
- **Actual Win Rate:** 52.1%
- **Mean Predicted Probability:** 66.2%
- **Overconfidence:** +14.1% (overconfident)

### HOLD Decisions

- **Count:** 155 decisions
- **Actual Win Rate:** 0.0%
- **Mean Predicted Probability:** 65.0%
- **Overconfidence:** +65.0% (overconfident)

### SELL Decisions

- **Count:** 176 decisions
- **Actual Win Rate:** 48.9%
- **Mean Predicted Probability:** 64.1%
- **Overconfidence:** +15.2% (overconfident)

---

## Recommendations

- **Calibration training needed:** Consider recalibrating the model using techniques like isotonic regression or Platt scaling
- **Overconfidence detected:** Model predictions are too optimistic. Consider adjusting confidence thresholds or using ensemble methods
- **Overconfidence in BUY, HOLD, SELL decisions:** Consider more conservative thresholds for these actions

---

## Visualizations

![Calibration Plot](../plots/dummy_model_memory_feeling_calibration.png)

![Calibration by Decision](../plots/dummy_model_memory_feeling_calibration_by_decision.png)
```

![Decision Patterns](plots\dummy_model_memory_feeling_decision_patterns.png)
*Figure 6: Decision patterns after wins vs losses*

---

## 📉 Risk Analysis

---

## 🛡️ HOLD Decision Analysis

**Overall HOLD Success Rate:** 40.4% (POOR)

### Quiet Market Success (<0.2% Daily Moves)

- **Success Rate:** 0.6%
- **Successful HOLDs:** 1/155
- **HOLD succeeded in 0.6% of very quiet markets (<0.2% daily moves)**

### Contextual Decision Correctness

- **Average Context Score:** 1.57
- **Context Success Rate:** 100.0%
- **HOLD was contextually appropriate in 100.0% of cases**

**Top Context Reasons:**

- Extreme Conditions: 155 decisions
- Regime Change: 148 decisions
- High Volatility: 33 decisions
- Decision Uncertainty: 1 decisions

### Combined Assessment

- **Quiet Market Weight:** 60%
- **Context Weight:** 40%
- **Overall Score:** 40.4%

### HOLD Usage Statistics

- **Total HOLD Decisions:** 155
- **HOLD Usage Rate:** 31.0%
- **Avg Market Move During HOLD:** 52.40%
- **HOLD During Quiet Markets:** 0.6%

**Key Insight:** HOLD decisions were most successful during quiet markets (0.6%) and appropriate contexts (100.0%)

---

## 📈 Additional Performance Analysis

---

## 💡 Key Insights & Recommendations

### Key Findings
- ⚠️  Strategy shows signs of overfitting or poor generalization
- 🚨 High overfitting detected
- 🤔 Strategy performance vs index is not statistically significant

### Recommendations
- Consider model regularization or simpler strategy

### Overall Assessment
- **Assessment**: CONCERNING
- **Confidence Level**: LOW

---

## 📁 Technical Details

- **Model Tag**: dummy_model_memory_feeling
- **Generated**: 2025-11-28 03:30:38

### Data Sources
- **Statistical Validation**: Bootstrap analysis and out-of-sample testing
- **Baseline Comparison**: Performance vs multiple strategies
- **Calibration Analysis**: Available
- **Pattern Analysis**: Available
- **Plots**: 5 chart files

### Analysis Components
- Statistical significance testing (bootstrap)
- Out-of-sample validation
- Risk-adjusted performance metrics
- Decision calibration analysis
- Baseline strategy comparisons
- Rolling performance analysis

---

*This report was automatically generated by the LLM Finance Experiment framework.*
*For questions about methodology or results, refer to the technical documentation.*