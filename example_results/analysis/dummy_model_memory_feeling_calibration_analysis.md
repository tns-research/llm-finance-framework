# Calibration Analysis Report - dummy_model_memory_feeling

## Overview

This report analyzes the calibration quality of the **dummy_model_memory_feeling** model, 
measuring how well predicted probabilities match actual outcomes.

---

## Overall Calibration Metrics

**Total Trading Days:** 1000
**Overall Win Rate:** 35.0%
**Mean Predicted Probability:** 65.2%

### Calibration Quality Indicators

- **Expected Calibration Error (ECE):** 30.2%
- **Maximum Calibration Error:** 50.7%
- **Overconfidence Score:** 43.4%

- **Calibration Quality:** **POOR** - Model shows significant calibration problems

### Confidence Assessment

- **Assessment:** **OVERCONFIDENT** - Model tends to be too optimistic about success probability

---

## Calibration by Decision Type

This analysis shows if the model has different calibration characteristics for BUY, HOLD, and SELL decisions.

### BUY Decisions

- **Count:** 339 decisions
- **Actual Win Rate:** 55.5%
- **Mean Predicted Probability:** 65.6%
- **Overconfidence:** +10.1% (overconfident)

### HOLD Decisions

- **Count:** 333 decisions
- **Actual Win Rate:** 0.0%
- **Mean Predicted Probability:** 63.9%
- **Overconfidence:** +63.9% (overconfident)

### SELL Decisions

- **Count:** 328 decisions
- **Actual Win Rate:** 49.4%
- **Mean Predicted Probability:** 66.0%
- **Overconfidence:** +16.6% (overconfident)

---

## Recommendations

- **Calibration training needed:** Consider recalibrating the model using techniques like isotonic regression or Platt scaling
- **Overconfidence detected:** Model predictions are too optimistic. Consider adjusting confidence thresholds or using ensemble methods
- **Overconfidence in BUY, HOLD, SELL decisions:** Consider more conservative thresholds for these actions

---

## Visualizations

![Calibration Plot](../plots/dummy_model_memory_feeling_calibration.png)

![Calibration by Decision](../plots/dummy_model_memory_feeling_calibration_by_decision.png)