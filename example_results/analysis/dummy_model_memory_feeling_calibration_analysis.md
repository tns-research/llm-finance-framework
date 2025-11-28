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