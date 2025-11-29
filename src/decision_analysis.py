# src/decision_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict


def analyze_decisions_after_outcomes(parsed_df: pd.DataFrame) -> dict:
    """
    Analyze decision patterns following wins vs losses.

    Returns statistics on:
    - Decision distribution after wins vs losses
    - Average confidence (probability) after wins vs losses
    - Behavioral differences

    Args:
        parsed_df: DataFrame with columns including 'previous_return', 'decision', 'prob'

    Returns:
        Dictionary with analysis results
    """
    # Filter out first row (has no previous return)
    df = parsed_df[parsed_df["previous_return"].notna()].copy()

    if len(df) == 0:
        return {"error": "No valid data for analysis (no rows with previous_return)"}

    # Classify previous outcomes
    df["previous_outcome"] = df["previous_return"].apply(
        lambda x: "win" if x > 0 else ("loss" if x < 0 else "neutral")
    )

    # Decision distribution after wins
    after_wins = df[df["previous_outcome"] == "win"]
    after_losses = df[df["previous_outcome"] == "loss"]
    after_neutral = df[df["previous_outcome"] == "neutral"]

    results = {
        "total_decisions": len(df),
        "total_wins": len(after_wins),
        "total_losses": len(after_losses),
        "total_neutral": len(after_neutral),
    }

    # Decision distributions
    if len(after_wins) > 0:
        win_dist = after_wins["decision"].value_counts(normalize=True) * 100
        results["decisions_after_wins"] = {
            "BUY": win_dist.get("BUY", 0),
            "HOLD": win_dist.get("HOLD", 0),
            "SELL": win_dist.get("SELL", 0),
            "mean_prob": after_wins["prob"].mean(),
            "std_prob": after_wins["prob"].std(),
        }
    else:
        results["decisions_after_wins"] = None

    if len(after_losses) > 0:
        loss_dist = after_losses["decision"].value_counts(normalize=True) * 100
        results["decisions_after_losses"] = {
            "BUY": loss_dist.get("BUY", 0),
            "HOLD": loss_dist.get("HOLD", 0),
            "SELL": loss_dist.get("SELL", 0),
            "mean_prob": after_losses["prob"].mean(),
            "std_prob": after_losses["prob"].std(),
        }
    else:
        results["decisions_after_losses"] = None

    # Chi-square test for independence
    if len(after_wins) > 0 and len(after_losses) > 0:
        # Create contingency table
        contingency = pd.crosstab(
            df[df["previous_outcome"].isin(["win", "loss"])]["previous_outcome"],
            df[df["previous_outcome"].isin(["win", "loss"])]["decision"],
        )

        if contingency.size > 0:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results["chi_square_test"] = {
                "chi2": chi2,
                "p_value": p_value,
                "dof": dof,
                "significant": p_value < 0.05,
            }
        else:
            results["chi_square_test"] = None
    else:
        results["chi_square_test"] = None

    return results


def analyze_position_duration_stats(parsed_df: pd.DataFrame) -> dict:
    """
    Calculate statistics on position holding durations.

    Returns:
    - Average duration by position type (BUY, HOLD, SELL)
    - Duration distribution analysis
    - Longest holding streaks

    Args:
        parsed_df: DataFrame with columns including 'decision', 'position_duration'

    Returns:
        Dictionary with duration statistics
    """
    df = parsed_df.copy()

    # Find position changes to identify complete position runs
    position_runs = []

    for i, row in df.iterrows():
        if row["position_changed"] or i == 0:
            # Start of a new position
            position_runs.append(
                {
                    "decision": row["decision"],
                    "start_idx": i,
                    "duration": 1,
                }
            )
        else:
            # Continuation of current position
            if position_runs:
                position_runs[-1]["duration"] = row["position_duration"]

    # Convert to DataFrame for easier analysis
    runs_df = pd.DataFrame(position_runs)

    results = {
        "total_position_changes": df["position_changed"].sum(),
        "average_position_duration": df["position_duration"].mean(),
        "median_position_duration": df["position_duration"].median(),
        "max_position_duration": df["position_duration"].max(),
    }

    # Duration by decision type
    for decision in ["BUY", "HOLD", "SELL"]:
        decision_runs = runs_df[runs_df["decision"] == decision]
        if len(decision_runs) > 0:
            results[f"{decision}_stats"] = {
                "count": len(decision_runs),
                "mean_duration": decision_runs["duration"].mean(),
                "median_duration": decision_runs["duration"].median(),
                "max_duration": decision_runs["duration"].max(),
            }
        else:
            results[f"{decision}_stats"] = None

    # Find longest streaks
    if len(runs_df) > 0:
        longest_streak = runs_df.loc[runs_df["duration"].idxmax()]
        results["longest_streak"] = {
            "decision": longest_streak["decision"],
            "duration": longest_streak["duration"],
        }
    else:
        results["longest_streak"] = None

    return results


def create_decision_pattern_plots(
    parsed_df: pd.DataFrame, model_tag: str, output_dir: str
):
    """
    Create visualizations comparing decision distributions after wins vs losses.

    Creates:
    - Bar charts showing BUY/HOLD/SELL percentages after wins vs losses
    - Box plots showing probability confidence after wins vs losses

    Args:
        parsed_df: DataFrame with decision and outcome data
        model_tag: Name of the model for plot titles
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter out first row
    df = parsed_df[parsed_df["previous_return"].notna()].copy()

    if len(df) == 0:
        print("[WARN] No data for decision pattern plots")
        return

    # Classify previous outcomes
    df["previous_outcome"] = df["previous_return"].apply(
        lambda x: "Win" if x > 0 else ("Loss" if x < 0 else "Neutral")
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Decision Pattern Analysis - {model_tag}", fontsize=16, fontweight="bold"
    )

    # 1. Decision distribution after wins vs losses
    ax1 = axes[0, 0]
    after_wins = df[df["previous_outcome"] == "Win"]
    after_losses = df[df["previous_outcome"] == "Loss"]

    decisions = ["BUY", "HOLD", "SELL"]
    x = np.arange(len(decisions))
    width = 0.35

    if len(after_wins) > 0:
        win_counts = [
            (after_wins["decision"] == d).sum() / len(after_wins) * 100
            for d in decisions
        ]
    else:
        win_counts = [0, 0, 0]

    if len(after_losses) > 0:
        loss_counts = [
            (after_losses["decision"] == d).sum() / len(after_losses) * 100
            for d in decisions
        ]
    else:
        loss_counts = [0, 0, 0]

    ax1.bar(
        x - width / 2, win_counts, width, label="After Wins", color="green", alpha=0.7
    )
    ax1.bar(
        x + width / 2, loss_counts, width, label="After Losses", color="red", alpha=0.7
    )
    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    ax1.set_title("Decision Distribution After Wins vs Losses", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(decisions)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (w, l) in enumerate(zip(win_counts, loss_counts)):
        ax1.text(
            i - width / 2, w + 1, f"{w:.1f}%", ha="center", va="bottom", fontsize=9
        )
        ax1.text(
            i + width / 2, l + 1, f"{l:.1f}%", ha="center", va="bottom", fontsize=9
        )

    # 2. Probability confidence after wins vs losses
    ax2 = axes[0, 1]
    outcome_data = []
    outcome_labels = []

    if len(after_wins) > 0:
        outcome_data.append(after_wins["prob"].values)
        outcome_labels.append(f"After Wins\n(n={len(after_wins)})")

    if len(after_losses) > 0:
        outcome_data.append(after_losses["prob"].values)
        outcome_labels.append(f"After Losses\n(n={len(after_losses)})")

    if outcome_data:
        bp = ax2.boxplot(outcome_data, labels=outcome_labels, patch_artist=True)
        colors = ["lightgreen", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors[: len(outcome_data)]):
            patch.set_facecolor(color)

    ax2.set_ylabel("Probability Confidence", fontweight="bold")
    ax2.set_title("Confidence Levels After Wins vs Losses", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 1)

    # 3. Position duration distribution
    ax3 = axes[1, 0]
    duration_data = df["position_duration"].value_counts().sort_index()
    ax3.bar(duration_data.index, duration_data.values, color="steelblue", alpha=0.7)
    ax3.set_xlabel("Position Duration (days)", fontweight="bold")
    ax3.set_ylabel("Frequency", fontweight="bold")
    ax3.set_title("Position Duration Distribution", fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Decision sequence after consecutive wins/losses
    ax4 = axes[1, 1]

    # Calculate win/loss streaks
    df["is_win"] = df["previous_return"] > 0
    df["streak"] = 0

    current_streak = 0
    for i in range(len(df)):
        if i == 0:
            current_streak = 1 if df.iloc[i]["is_win"] else -1
        else:
            if df.iloc[i]["is_win"] == df.iloc[i - 1]["is_win"]:
                current_streak = current_streak + (1 if df.iloc[i]["is_win"] else -1)
            else:
                current_streak = 1 if df.iloc[i]["is_win"] else -1
        df.loc[df.index[i], "streak"] = current_streak

    # Group by streak length and decision
    streak_decisions = df.groupby(["streak", "decision"]).size().unstack(fill_value=0)

    # Plot for streaks -3 to 3
    streak_range = range(-3, 4)
    buy_pcts = []
    hold_pcts = []
    sell_pcts = []

    for s in streak_range:
        if s in streak_decisions.index:
            total = streak_decisions.loc[s].sum()
            buy_pcts.append(
                streak_decisions.loc[s].get("BUY", 0) / total * 100 if total > 0 else 0
            )
            hold_pcts.append(
                streak_decisions.loc[s].get("HOLD", 0) / total * 100 if total > 0 else 0
            )
            sell_pcts.append(
                streak_decisions.loc[s].get("SELL", 0) / total * 100 if total > 0 else 0
            )
        else:
            buy_pcts.append(0)
            hold_pcts.append(0)
            sell_pcts.append(0)

    x_pos = np.arange(len(streak_range))
    ax4.bar(x_pos, buy_pcts, label="BUY", color="green", alpha=0.7)
    ax4.bar(x_pos, hold_pcts, bottom=buy_pcts, label="HOLD", color="gray", alpha=0.7)
    ax4.bar(
        x_pos,
        sell_pcts,
        bottom=np.array(buy_pcts) + np.array(hold_pcts),
        label="SELL",
        color="red",
        alpha=0.7,
    )

    ax4.set_xlabel("Win/Loss Streak (negative=loss, positive=win)", fontweight="bold")
    ax4.set_ylabel("Decision Distribution (%)", fontweight="bold")
    ax4.set_title("Decisions by Win/Loss Streak", fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([str(s) for s in streak_range])
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, f"{model_tag}_decision_patterns.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Decision pattern plots saved to: {output_path}")


def analyze_market_regimes_for_decisions(parsed_df: pd.DataFrame) -> Dict:
    """
    Analyze how decision-making performs across different market regimes.
    This complements the statistical risk analysis with decision-specific insights.

    Args:
        parsed_df: Parsed trading data with decisions and returns

    Returns:
        Dict with regime-specific decision analysis
    """
    if "decision" not in parsed_df.columns or "next_return_1d" not in parsed_df.columns:
        return {}

    # First get the basic regime classification
    from .statistical_validation import analyze_market_regimes

    regime_analysis = analyze_market_regimes(parsed_df)

    if not regime_analysis:
        return {}

    # Now analyze decisions within each regime
    decisions = parsed_df["decision"].values
    regime_decisions = {}

    # Classify market regimes (duplicate logic for now - could be refactored)
    market_returns = parsed_df["next_return_1d"].values
    rolling_vol = pd.Series(market_returns).rolling(20).std() * np.sqrt(252)
    vol_median = rolling_vol.median()
    vol_high = rolling_vol.quantile(0.75)

    regimes = []
    for vol in rolling_vol:
        if pd.isna(vol):
            regimes.append("unknown")
        elif vol > vol_high:
            regimes.append("high_volatility")
        elif vol > vol_median:
            regimes.append("moderate_volatility")
        else:
            regimes.append("low_volatility")

    # Decision analysis by regime
    for regime in ["low_volatility", "moderate_volatility", "high_volatility"]:
        regime_mask = np.array(regimes) == regime
        if np.any(regime_mask):
            regime_decisions_list = decisions[regime_mask]

            buy_count = np.sum(regime_decisions_list == "BUY")
            hold_count = np.sum(regime_decisions_list == "HOLD")
            sell_count = np.sum(regime_decisions_list == "SELL")
            total_decisions = len(regime_decisions_list)

            regime_decisions[regime] = {
                "decision_distribution": {
                    "BUY": int(buy_count),
                    "HOLD": int(hold_count),
                    "SELL": int(sell_count),
                },
                "decision_percentages": {
                    "BUY": float(buy_count / total_decisions * 100),
                    "HOLD": float(hold_count / total_decisions * 100),
                    "SELL": float(sell_count / total_decisions * 100),
                },
            }

    return {"regime_performance": regime_analysis, "regime_decisions": regime_decisions}


def generate_pattern_analysis_report(
    parsed_df: pd.DataFrame, model_tag: str, output_path: str
):
    """
    Generate comprehensive markdown report combining all analyses.

    Args:
        parsed_df: DataFrame with all decision and outcome data
        model_tag: Name of the model
        output_path: Path to save the markdown report
    """
    # Run analyses
    decision_analysis = analyze_decisions_after_outcomes(parsed_df)
    duration_analysis = analyze_position_duration_stats(parsed_df)

    # Build report content
    report_lines = [
        f"# Decision Pattern Analysis Report - {model_tag}",
        "",
        "## Overview",
        "",
        f"This report analyzes the decision-making patterns of the **{model_tag}** model, ",
        "focusing on how the model behaves after wins versus losses, and how long it holds positions.",
        "",
        "---",
        "",
        "## Decision Patterns After Wins vs Losses",
        "",
    ]

    # Decision analysis results
    if "error" in decision_analysis:
        report_lines.append(f"**Error:** {decision_analysis['error']}")
    else:
        report_lines.extend(
            [
                f"**Total Decisions Analyzed:** {decision_analysis['total_decisions']}",
                f"- After Wins: {decision_analysis['total_wins']}",
                f"- After Losses: {decision_analysis['total_losses']}",
                f"- After Neutral: {decision_analysis['total_neutral']}",
                "",
            ]
        )

        # After wins
        if decision_analysis["decisions_after_wins"]:
            wins = decision_analysis["decisions_after_wins"]
            report_lines.extend(
                [
                    "### Decisions After Wins",
                    "",
                    f"- **BUY:** {wins['BUY']:.1f}%",
                    f"- **HOLD:** {wins['HOLD']:.1f}%",
                    f"- **SELL:** {wins['SELL']:.1f}%",
                    f"- **Mean Confidence:** {wins['mean_prob']:.3f} (±{wins['std_prob']:.3f})",
                    "",
                ]
            )

        # After losses
        if decision_analysis["decisions_after_losses"]:
            losses = decision_analysis["decisions_after_losses"]
            report_lines.extend(
                [
                    "### Decisions After Losses",
                    "",
                    f"- **BUY:** {losses['BUY']:.1f}%",
                    f"- **HOLD:** {losses['HOLD']:.1f}%",
                    f"- **SELL:** {losses['SELL']:.1f}%",
                    f"- **Mean Confidence:** {losses['mean_prob']:.3f} (±{losses['std_prob']:.3f})",
                    "",
                ]
            )

        # Statistical test
        if decision_analysis["chi_square_test"]:
            chi_test = decision_analysis["chi_square_test"]
            significance_text = (
                "**statistically significant**"
                if chi_test["significant"]
                else "not statistically significant"
            )
            report_lines.extend(
                [
                    "### Statistical Independence Test",
                    "",
                    f"Chi-square test for independence between previous outcome and next decision:",
                    f"- **χ² statistic:** {chi_test['chi2']:.4f}",
                    f"- **p-value:** {chi_test['p_value']:.4f}",
                    f"- **Degrees of freedom:** {chi_test['dof']}",
                    f"- **Result:** The relationship is {significance_text} (α=0.05)",
                    "",
                ]
            )

            if chi_test["significant"]:
                report_lines.append("> [!IMPORTANT]")
                report_lines.append(
                    "> The model's decision-making **is significantly influenced** by previous outcomes. "
                )
                report_lines.append(
                    "> This suggests the model adapts its strategy based on recent performance."
                )
            else:
                report_lines.append("> [!NOTE]")
                report_lines.append(
                    "> The model's decisions appear **independent** of previous outcomes. "
                )
                report_lines.append(
                    "> This suggests consistent strategy regardless of recent wins/losses."
                )

            report_lines.append("")

    report_lines.extend(
        [
            "---",
            "",
            "## Position Duration Analysis",
            "",
        ]
    )

    # Duration analysis results
    report_lines.extend(
        [
            f"**Total Position Changes:** {duration_analysis['total_position_changes']}",
            f"**Average Position Duration:** {duration_analysis['average_position_duration']:.2f} days",
            f"**Median Position Duration:** {duration_analysis['median_position_duration']:.0f} days",
            f"**Maximum Position Duration:** {duration_analysis['max_position_duration']} days",
            "",
        ]
    )

    # By decision type
    for decision in ["BUY", "HOLD", "SELL"]:
        stats_key = f"{decision}_stats"
        if duration_analysis[stats_key]:
            stats = duration_analysis[stats_key]
            report_lines.extend(
                [
                    f"### {decision} Positions",
                    "",
                    f"- **Count:** {stats['count']}",
                    f"- **Mean Duration:** {stats['mean_duration']:.2f} days",
                    f"- **Median Duration:** {stats['median_duration']:.0f} days",
                    f"- **Max Duration:** {stats['max_duration']} days",
                    "",
                ]
            )

    # Longest streak
    if duration_analysis["longest_streak"]:
        streak = duration_analysis["longest_streak"]
        report_lines.extend(
            [
                "### Longest Position Streak",
                "",
                f"The longest consecutive position was **{streak['decision']}** held for **{streak['duration']} days**.",
                "",
            ]
        )

    # Add visualization
    report_lines.extend(
        [
            "---",
            "",
            "## Visualizations",
            "",
        ]
    )

    # Determine relative path to plot
    report_dir = os.path.dirname(output_path)
    plots_filename = f"{model_tag}_decision_patterns.png"

    # Try to find plots directory relative to report
    # Assuming report is in results/analysis/ and plots in results/plots/
    relative_plot_path = os.path.join("..", "plots", plots_filename)

    report_lines.extend(
        [
            f"![Decision Pattern Visualizations]({relative_plot_path})",
            "",
        ]
    )

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[INFO] Pattern analysis report saved to: {output_path}")
