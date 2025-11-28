# Decision Pattern Analysis Report - dummy_model_memory_feeling

## Overview

This report analyzes the decision-making patterns of the **dummy_model_memory_feeling** model, 
focusing on how the model behaves after wins versus losses, and how long it holds positions.

---

## Decision Patterns After Wins vs Losses

**Total Decisions Analyzed:** 499
- After Wins: 174
- After Losses: 170
- After Neutral: 155

### Decisions After Wins

- **BUY:** 35.1%
- **HOLD:** 32.2%
- **SELL:** 32.8%
- **Mean Confidence:** 0.649 (±0.146)

### Decisions After Losses

- **BUY:** 35.3%
- **HOLD:** 27.1%
- **SELL:** 37.6%
- **Mean Confidence:** 0.653 (±0.148)

### Statistical Independence Test

Chi-square test for independence between previous outcome and next decision:
- **χ² statistic:** 1.3473
- **p-value:** 0.5098
- **Degrees of freedom:** 2
- **Result:** The relationship is not statistically significant (α=0.05)

> [!NOTE]
> The model's decisions appear **independent** of previous outcomes. 
> This suggests consistent strategy regardless of recent wins/losses.

---

## Position Duration Analysis

**Total Position Changes:** 324
**Average Position Duration:** 1.56 days
**Median Position Duration:** 1 days
**Maximum Position Duration:** 7 days

### BUY Positions

- **Count:** 108
- **Mean Duration:** 1.56 days
- **Median Duration:** 1 days
- **Max Duration:** 5 days

### HOLD Positions

- **Count:** 103
- **Mean Duration:** 1.50 days
- **Median Duration:** 1 days
- **Max Duration:** 7 days

### SELL Positions

- **Count:** 114
- **Mean Duration:** 1.54 days
- **Median Duration:** 1 days
- **Max Duration:** 6 days

### Longest Position Streak

The longest consecutive position was **HOLD** held for **7 days**.

---

## Visualizations

![Decision Pattern Visualizations](..\plots\dummy_model_memory_feeling_decision_patterns.png)
