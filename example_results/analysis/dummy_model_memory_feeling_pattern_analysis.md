# Decision Pattern Analysis Report - dummy_model_memory_feeling

## Overview

This report analyzes the decision-making patterns of the **dummy_model_memory_feeling** model, 
focusing on how the model behaves after wins versus losses, and how long it holds positions.

---

## Decision Patterns After Wins vs Losses

**Total Decisions Analyzed:** 499
- After Wins: 161
- After Losses: 164
- After Neutral: 174

### Decisions After Wins

- **BUY:** 32.3%
- **HOLD:** 37.9%
- **SELL:** 29.8%
- **Mean Confidence:** 0.660 (±0.148)

### Decisions After Losses

- **BUY:** 34.1%
- **HOLD:** 33.5%
- **SELL:** 32.3%
- **Mean Confidence:** 0.649 (±0.139)

### Statistical Independence Test

Chi-square test for independence between previous outcome and next decision:
- **χ² statistic:** 0.6784
- **p-value:** 0.7123
- **Degrees of freedom:** 2
- **Result:** The relationship is not statistically significant (α=0.05)

> [!NOTE]
> The model's decisions appear **independent** of previous outcomes. 
> This suggests consistent strategy regardless of recent wins/losses.

---

## Position Duration Analysis

**Total Position Changes:** 325
**Average Position Duration:** 1.54 days
**Median Position Duration:** 1 days
**Maximum Position Duration:** 8 days

### BUY Positions

- **Count:** 102
- **Mean Duration:** 1.61 days
- **Median Duration:** 1 days
- **Max Duration:** 6 days

### HOLD Positions

- **Count:** 116
- **Mean Duration:** 1.49 days
- **Median Duration:** 1 days
- **Max Duration:** 5 days

### SELL Positions

- **Count:** 108
- **Mean Duration:** 1.51 days
- **Median Duration:** 1 days
- **Max Duration:** 8 days

### Longest Position Streak

The longest consecutive position was **SELL** held for **8 days**.

---

## Visualizations

![Decision Pattern Visualizations](..\plots\dummy_model_memory_feeling_decision_patterns.png)
