# Decision Pattern Analysis Report - dummy_model_memory_feeling

## Overview

This report analyzes the decision-making patterns of the **dummy_model_memory_feeling** model, 
focusing on how the model behaves after wins versus losses, and how long it holds positions.

---

## Decision Patterns After Wins vs Losses

**Total Decisions Analyzed:** 999
- After Wins: 350
- After Losses: 316
- After Neutral: 333

### Decisions After Wins

- **BUY:** 34.6%
- **HOLD:** 30.6%
- **SELL:** 34.9%
- **Mean Confidence:** 0.651 (±0.148)

### Decisions After Losses

- **BUY:** 32.6%
- **HOLD:** 34.8%
- **SELL:** 32.6%
- **Mean Confidence:** 0.653 (±0.149)

### Statistical Independence Test

Chi-square test for independence between previous outcome and next decision:
- **χ² statistic:** 1.3602
- **p-value:** 0.5066
- **Degrees of freedom:** 2
- **Result:** The relationship is not statistically significant (α=0.05)

> [!NOTE]
> The model's decisions appear **independent** of previous outcomes. 
> This suggests consistent strategy regardless of recent wins/losses.

---

## Position Duration Analysis

**Total Position Changes:** 651
**Average Position Duration:** 1.57 days
**Median Position Duration:** 1 days
**Maximum Position Duration:** 7 days

### BUY Positions

- **Count:** 224
- **Mean Duration:** 1.51 days
- **Median Duration:** 1 days
- **Max Duration:** 7 days

### HOLD Positions

- **Count:** 217
- **Mean Duration:** 1.53 days
- **Median Duration:** 1 days
- **Max Duration:** 7 days

### SELL Positions

- **Count:** 211
- **Mean Duration:** 1.55 days
- **Median Duration:** 1 days
- **Max Duration:** 7 days

### Longest Position Streak

The longest consecutive position was **HOLD** held for **7 days**.

---

## Visualizations

![Decision Pattern Visualizations](..\plots\dummy_model_memory_feeling_decision_patterns.png)
