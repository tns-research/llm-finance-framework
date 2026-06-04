# src/reporting/charts.py
"""Matplotlib chart generation and calibration reporting.

Moved verbatim out of the old monolithic src/reporting.py (pure structural
split, no logic change). Re-exported via the reporting package __init__.

Note: this module sits one package level deeper than the original
src/reporting.py, so the in-body ``from .statistical_validation import ...``
imports become ``from ..statistical_validation import ...``.
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import TRADING_DAYS_PER_YEAR


def calculate_decision_success(parsed_df: pd.DataFrame) -> pd.Series:
    """
    Calculate success indicators for each decision type using appropriate criteria.

    - BUY/SELL: strategy_return > 0 (positive returns)
    - HOLD: Use specialized HOLD success criteria (quiet markets, risk avoidance)

    Returns:
        pd.Series: Boolean success indicators for each decision
    """
    from ..statistical_validation import evaluate_hold_decisions_dual_criteria

    df = parsed_df.copy()

    # Default: strategy_return > 0 for BUY/SELL
    success = (df["strategy_return"] > 0).astype(int)

    # Override HOLD decisions with specialized criteria
    hold_mask = df["decision"] == "HOLD"
    if hold_mask.any():
        hold_analysis = evaluate_hold_decisions_dual_criteria(df)
        if "hold_success_indicators" in hold_analysis:
            hold_success_df = hold_analysis["hold_success_indicators"]
            # Merge success indicators back to main dataframe
            hold_success_map = dict(
                zip(hold_success_df["index"], hold_success_df["hold_success"])
            )
            success.loc[hold_mask] = df.loc[hold_mask].index.map(
                lambda idx: hold_success_map.get(idx, 0)
            )

    return success


def create_calibration_plot(parsed_df, model_tag: str, output_path: str):
    """
    Create a calibration plot showing predicted probability vs actual win rate.

    The plot bins predictions by probability and shows:
    - Diagonal line (perfect calibration)
    - Actual frequency of wins per bin
    - Count of predictions in each bin

    parsed_df: DataFrame with columns 'prob' and 'strategy_return'
    model_tag: name of the model for plot title
    output_path: where to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = parsed_df.copy()

    # Define if a prediction was a "win" (positive strategy return)
    df["win"] = (df["strategy_return"] > 0).astype(int)

    # Create probability bins
    bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1, 0.1-0.2, ..., 0.9-1.0]
    df["prob_bin"] = pd.cut(df["prob"], bins=bins, include_lowest=True)

    # Calculate actual win rate per bin
    calibration_data = (
        df.groupby("prob_bin", observed=False)
        .agg(
            mean_predicted_prob=("prob", "mean"),
            actual_win_rate=("win", "mean"),
            count=("win", "count"),
        )
        .reset_index()
    )

    # Filter out bins with no data
    calibration_data = calibration_data[calibration_data["count"] > 0]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration", alpha=0.7)

    # Plot actual calibration
    ax.scatter(
        calibration_data["mean_predicted_prob"],
        calibration_data["actual_win_rate"],
        s=calibration_data["count"] * 10,  # Size proportional to count
        alpha=0.6,
        color="steelblue",
        edgecolors="black",
        linewidth=1.5,
        label="Actual Win Rate",
    )

    # Add count labels
    for _, row in calibration_data.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (row["mean_predicted_prob"], row["actual_win_rate"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    # Formatting
    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Win Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Calibration Plot - {model_tag}\nPredicted Probability vs Actual Win Rate",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=10)

    # Add statistics text box
    total_trades = len(df)
    overall_win_rate = df["win"].mean()
    mean_predicted = df["prob"].mean()

    stats_text = (
        f"Total Trades: {total_trades}\n"
        f"Overall Win Rate: {overall_win_rate:.2%}\n"
        f"Mean Predicted Prob: {mean_predicted:.2%}"
    )

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Calibration plot saved to: {output_path}")

    return calibration_data


def create_calibration_by_decision_plot(parsed_df, model_tag: str, output_path: str):
    """
    Separate calibration curves for BUY/HOLD/SELL decisions.
    This reveals if the model is overconfident on specific action types.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, decision in enumerate(["BUY", "HOLD", "SELL"]):
        ax = axes[idx]
        subset = parsed_df[parsed_df["decision"] == decision]

        if len(subset) < 5:
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\n(n={len(subset)})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{decision} Decisions")
            continue

        subset = subset.copy()
        # Use decision-type-specific success criteria
        subset["win"] = calculate_decision_success(subset)

        # Bin and plot
        bins = np.linspace(0, 1, 6)
        subset["prob_bin"] = pd.cut(subset["prob"], bins=bins, include_lowest=True)

        calibration = (
            subset.groupby("prob_bin", observed=False)
            .agg(
                mean_prob=("prob", "mean"),
                win_rate=("win", "mean"),
                count=("win", "count"),
            )
            .dropna()
        )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.scatter(
            calibration["mean_prob"],
            calibration["win_rate"],
            s=calibration["count"] * 5,
            alpha=0.7,
        )
        ax.set_title(f"{decision} Decisions (n={len(subset)})")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Calibration by Decision Type - {model_tag}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n[INFO] Calibration by decision plot saved to: {output_path}")


def create_risk_analysis_chart(
    parsed_df: pd.DataFrame, model_tag: str, output_path: str
):
    """
    Create comprehensive risk analysis chart with VaR and stress tests.

    Args:
        parsed_df: Parsed trading data
        model_tag: Model identifier for chart title
        output_path: Path to save the chart
    """
    from ..statistical_validation import calculate_var_and_stress_tests

    if "strategy_return" not in parsed_df.columns:
        print(f"Warning: No strategy returns found for risk analysis chart")
        return

    returns = parsed_df["strategy_return"].values
    dates = parsed_df["date"] if "date" in parsed_df.columns else None

    # Get risk analysis data
    risk_data = calculate_var_and_stress_tests(returns, dates)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Risk Analysis - {model_tag}", fontsize=16, fontweight="bold")

    # 1. Returns Distribution
    ax1 = axes[0, 0]
    ax1.hist(returns, bins=50, alpha=0.7, density=True, label="Strategy Returns")
    ax1.axvline(
        np.mean(returns),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(returns):.3f}%",
    )
    ax1.set_title("Returns Distribution", fontweight="bold")
    ax1.set_xlabel("Daily Return (%)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Value at Risk (VaR) Over Time
    ax2 = axes[0, 1]
    if "var_95" in risk_data and len(risk_data["var_95"]) > 0:
        var_dates = risk_data.get("var_dates", list(range(len(risk_data["var_95"]))))
        ax2.plot(
            var_dates, risk_data["var_95"], label="VaR 95%", color="orange", alpha=0.8
        )
        ax2.plot(
            var_dates, risk_data["var_99"], label="VaR 99%", color="red", alpha=0.8
        )
        ax2.fill_between(
            var_dates,
            risk_data["var_99"],
            risk_data["var_95"],
            alpha=0.2,
            color="red",
            label="Tail Risk Zone",
        )

    ax2.set_title("Rolling Value at Risk", fontweight="bold")
    ax2.set_ylabel("VaR (%)")
    if ax2.get_legend_handles_labels()[
        1
    ]:  # Only show legend if there are labeled artists
        ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Drawdown Analysis
    ax3 = axes[1, 0]
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max

    ax3.fill_between(range(len(drawdowns)), 0, drawdowns, color="red", alpha=0.3)
    ax3.plot(drawdowns, color="red", linewidth=1)
    ax3.set_title("Drawdown Analysis", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(alpha=0.3)

    # 4. Stress Test Scenarios
    ax4 = axes[1, 1]
    if "stress_tests" in risk_data:
        for scenario_name, scenario_data in risk_data["stress_tests"].items():
            if scenario_name != "base_case":  # Skip base case to avoid clutter
                ax4.plot(
                    scenario_data["cumulative"],
                    label=scenario_name.replace("_", " ").title(),
                    alpha=0.8,
                )

    ax4.set_title("Stress Test Scenarios", fontweight="bold")
    ax4.set_ylabel("Cumulative Return (%)")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Risk analysis chart saved: {output_path}")


def create_rolling_performance_chart(
    parsed_df: pd.DataFrame, model_tag: str, output_path: str
):
    """
    Create rolling performance analysis chart.

    Args:
        parsed_df: Parsed trading data
        model_tag: Model identifier for chart title
        output_path: Path to save the chart
    """
    if "strategy_return" not in parsed_df.columns or "date" not in parsed_df.columns:
        print(f"Warning: Missing required columns for rolling performance chart")
        return

    df = parsed_df.sort_values("date").copy()
    window_sizes = [63, 126, 252]  # ~3, 6, 12 months

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Rolling Performance Analysis - {model_tag}", fontsize=16, fontweight="bold"
    )

    # 1. Rolling Sharpe Ratio
    ax1 = axes[0, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_sharpe = (
                df["strategy_return"]
                .rolling(window)
                .apply(
                    lambda x: (
                        x.mean() / x.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                        if x.std() > 0
                        else 0
                    )
                )
            )
            ax1.plot(
                df["date"], rolling_sharpe, label=f"{window//21}M Window", alpha=0.8
            )

    ax1.set_title("Rolling Sharpe Ratio", fontweight="bold")
    ax1.set_ylabel("Annualized Sharpe Ratio")
    if ax1.get_legend_handles_labels()[
        1
    ]:  # Only show legend if there are labeled artists
        ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Rolling Returns
    ax2 = axes[0, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_returns = df["strategy_return"].rolling(window).sum()
            ax2.plot(
                df["date"], rolling_returns, label=f"{window//21}M Window", alpha=0.8
            )

    ax2.set_title("Rolling Total Returns", fontweight="bold")
    ax2.set_ylabel("Total Return (%)")
    if ax2.get_legend_handles_labels()[
        1
    ]:  # Only show legend if there are labeled artists
        ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Rolling Maximum Drawdown
    ax3 = axes[1, 0]
    for window in window_sizes:
        if len(df) > window:
            rolling_cumulative = df["strategy_return"].rolling(window).sum()
            rolling_max = rolling_cumulative.rolling(window, min_periods=1).max()
            rolling_dd = rolling_cumulative - rolling_max
            ax3.plot(df["date"], rolling_dd, label=f"{window//21}M Window", alpha=0.8)

    ax3.set_title("Rolling Maximum Drawdown", fontweight="bold")
    ax3.set_ylabel("Drawdown (%)")
    if ax3.get_legend_handles_labels()[
        1
    ]:  # Only show legend if there are labeled artists
        ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Rolling Win Rate
    ax4 = axes[1, 1]
    for window in window_sizes:
        if len(df) > window:
            rolling_wins = (df["strategy_return"] > 0).rolling(window).sum()
            rolling_win_rate = rolling_wins / window * 100
            ax4.plot(
                df["date"], rolling_win_rate, label=f"{window//21}M Window", alpha=0.8
            )

    ax4.set_title("Rolling Win Rate", fontweight="bold")
    ax4.set_ylabel("Win Rate (%)")
    ax4.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50% Line")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Rolling performance chart saved: {output_path}")


def generate_calibration_analysis_report(
    calibration_data: pd.DataFrame,
    parsed_df: pd.DataFrame,
    model_tag: str,
    output_path: str,
):
    """
    Generate a comprehensive markdown report analyzing model calibration.

    Args:
        calibration_data: DataFrame from create_calibration_plot() with calibration statistics
        parsed_df: Original parsed DataFrame with all decision data
        model_tag: Name of the model
        output_path: Path to save the markdown report
    """
    import numpy as np

    # Calculate overall calibration metrics
    total_trades = len(parsed_df)
    overall_win_rate = (parsed_df["strategy_return"] > 0).mean()
    mean_predicted = parsed_df["prob"].mean()

    # Calculate calibration quality metrics
    if len(calibration_data) > 0:
        # Expected calibration error (ECE)
        ece = (
            np.sum(
                np.abs(
                    calibration_data["mean_predicted_prob"]
                    - calibration_data["actual_win_rate"]
                )
                * calibration_data["count"]
            )
            / total_trades
        )

        # Maximum calibration error
        max_ce = np.max(
            np.abs(
                calibration_data["mean_predicted_prob"]
                - calibration_data["actual_win_rate"]
            )
        )

        # Overconfidence score (predicted > actual for high confidence)
        high_conf_data = calibration_data[calibration_data["mean_predicted_prob"] > 0.7]
        if len(high_conf_data) > 0:
            overconfidence = np.mean(
                high_conf_data["mean_predicted_prob"]
                - high_conf_data["actual_win_rate"]
            )
        else:
            overconfidence = 0.0
    else:
        ece = max_ce = overconfidence = 0.0

    # Decision-specific calibration
    decision_calibration = {}
    for decision in ["BUY", "HOLD", "SELL"]:
        subset = parsed_df[parsed_df["decision"] == decision].copy()
        if len(subset) >= 5:
            # Use decision-type-specific success criteria
            subset["win"] = calculate_decision_success(subset)
            decision_calibration[decision] = {
                "count": len(subset),
                "win_rate": subset["win"].mean(),
                "mean_prob": subset["prob"].mean(),
                "overconfidence": subset["prob"].mean() - subset["win"].mean(),
            }
        else:
            decision_calibration[decision] = None

    # Build report
    report_lines = [
        f"# Calibration Analysis Report - {model_tag}",
        "",
        "## Overview",
        "",
        f"This report analyzes the calibration quality of the **{model_tag}** model, ",
        "measuring how well predicted probabilities match actual outcomes.",
        "",
        "---",
        "",
        "## Overall Calibration Metrics",
        "",
        f"**Total Trading Days:** {total_trades}",
        f"**Overall Win Rate:** {overall_win_rate:.1%}",
        f"**Mean Predicted Probability:** {mean_predicted:.1%}",
        "",
        "### Calibration Quality Indicators",
        "",
        f"- **Expected Calibration Error (ECE):** {ece:.1%}",
        f"- **Maximum Calibration Error:** {max_ce:.1%}",
        f"- **Overconfidence Score:** {overconfidence:.1%}",
        "",
    ]

    # ECE interpretation
    if ece < 0.05:
        ece_quality = "**EXCELLENT** - Model is very well calibrated"
    elif ece < 0.10:
        ece_quality = "**GOOD** - Model shows reasonable calibration"
    elif ece < 0.20:
        ece_quality = "**FAIR** - Model has moderate calibration issues"
    else:
        ece_quality = "**POOR** - Model shows significant calibration problems"

    report_lines.extend(
        [
            f"- **Calibration Quality:** {ece_quality}",
            "",
        ]
    )

    # Overconfidence interpretation
    if overconfidence > 0.10:
        conf_assessment = "**OVERCONFIDENT** - Model tends to be too optimistic about success probability"
    elif overconfidence < -0.10:
        conf_assessment = "**UNDERCONFIDENT** - Model tends to be too pessimistic about success probability"
    else:
        conf_assessment = "**WELL-CALIBRATED** - Model confidence matches reality"

    report_lines.extend(
        [
            "### Confidence Assessment",
            "",
            f"- **Assessment:** {conf_assessment}",
            "",
        ]
    )

    # Decision-specific calibration
    report_lines.extend(
        [
            "---",
            "",
            "## Calibration by Decision Type",
            "",
            "This analysis shows if the model has different calibration characteristics for BUY, HOLD, and SELL decisions.",
            "",
        ]
    )

    for decision in ["BUY", "HOLD", "SELL"]:
        calib = decision_calibration[decision]
        if calib:
            overconf = calib["overconfidence"]
            if overconf > 0.05:
                conf_desc = "overconfident"
            elif overconf < -0.05:
                conf_desc = "underconfident"
            else:
                conf_desc = "well-calibrated"

            report_lines.extend(
                [
                    f"### {decision} Decisions",
                    "",
                    f"- **Count:** {calib['count']} decisions",
                    f"- **Actual Win Rate:** {calib['win_rate']:.1%}",
                    f"- **Mean Predicted Probability:** {calib['mean_prob']:.1%}",
                    f"- **Overconfidence:** {overconf:+.1%} ({conf_desc})",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    f"### {decision} Decisions",
                    "",
                    f"- **Insufficient data** (n < 5)",
                    "",
                ]
            )

    # Recommendations
    report_lines.extend(
        [
            "---",
            "",
            "## Recommendations",
            "",
        ]
    )

    recommendations = []

    if ece > 0.15:
        recommendations.append(
            "- **Calibration training needed:** Consider recalibrating the model using techniques like isotonic regression or Platt scaling"
        )

    if overconfidence > 0.10:
        recommendations.append(
            "- **Overconfidence detected:** Model predictions are too optimistic. Consider adjusting confidence thresholds or using ensemble methods"
        )

    if overconfidence < -0.10:
        recommendations.append(
            "- **Underconfidence detected:** Model predictions are too conservative. Consider boosting confidence for high-probability predictions"
        )

    # Check for decision-specific issues
    overconfident_decisions = [
        d for d, c in decision_calibration.items() if c and c["overconfidence"] > 0.10
    ]
    underconfident_decisions = [
        d for d, c in decision_calibration.items() if c and c["overconfidence"] < -0.10
    ]

    if overconfident_decisions:
        recommendations.append(
            f"- **Overconfidence in {', '.join(overconfident_decisions)} decisions:** Consider more conservative thresholds for these actions"
        )

    if underconfident_decisions:
        recommendations.append(
            f"- **Underconfidence in {', '.join(underconfident_decisions)} decisions:** Consider being more aggressive with these actions"
        )

    if not recommendations:
        recommendations.append(
            "- **Calibration looks good:** No major issues detected. Continue monitoring calibration quality."
        )

    for rec in recommendations:
        report_lines.append(rec)

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## Visualizations",
            "",
            f"![Calibration Plot](../plots/{model_tag}_calibration.png)",
            "",
            f"![Calibration by Decision](../plots/{model_tag}_calibration_by_decision.png)",
        ]
    )

    # Write report
    report_content = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"\n[INFO] Calibration analysis report saved to: {output_path}")

    return {
        "ece": ece,
        "max_ce": max_ce,
        "overconfidence": overconfidence,
        "total_trades": total_trades,
        "overall_win_rate": overall_win_rate,
        "mean_predicted": mean_predicted,
        "decision_calibration": decision_calibration,
    }


def create_category_performance_plot(
    category_stats: pd.DataFrame, llm_metrics: Dict, output_path: str
) -> None:
    """
    Create comprehensive category performance comparison plot.

    Args:
        category_stats: DataFrame from calculate_category_performance()
        llm_metrics: LLM performance metrics
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Set style (fallback if seaborn not available)
    try:
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        # Fallback to matplotlib default styling
        plt.style.use("default")

    plt.rcParams["figure.figsize"] = (16, 10)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Strategy Category Performance Analysis", fontsize=16, fontweight="bold"
    )

    # Prepare LLM data
    llm_return = llm_metrics.get("total_return", 0)
    llm_sharpe = llm_metrics.get("sharpe_like", 0) * (
        TRADING_DAYS_PER_YEAR**0.5
    )  # Annualized
    llm_win_rate = llm_metrics.get("hit_rate", 0) * 100

    # Plot 1: Returns comparison
    categories = category_stats["category"].values
    category_returns = category_stats["avg_return"].values

    bars1 = axes[0, 0].bar(
        categories,
        category_returns,
        alpha=0.7,
        color="skyblue",
        label="Category Average",
    )
    axes[0, 0].axhline(
        y=llm_return,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"LLM Strategy ({llm_return:+.1f}%)",
    )

    # Add value labels on bars
    for bar, value in zip(bars1, category_returns):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 0].set_title("Returns by Strategy Category", fontweight="bold")
    axes[0, 0].set_ylabel("Total Return (%)")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sharpe ratios
    category_sharpes = category_stats["avg_sharpe"].values

    bars2 = axes[0, 1].bar(
        categories,
        category_sharpes,
        alpha=0.7,
        color="lightgreen",
        label="Category Average",
    )
    axes[0, 1].axhline(
        y=llm_sharpe,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"LLM Strategy ({llm_sharpe:+.2f})",
    )

    for bar, value in zip(bars2, category_sharpes):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[0, 1].set_title("Risk-Adjusted Returns (Sharpe Ratio)", fontweight="bold")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Win rates
    category_win_rates = category_stats["avg_win_rate"].values

    bars3 = axes[1, 0].bar(
        categories,
        category_win_rates,
        alpha=0.7,
        color="orange",
        label="Category Average",
    )
    axes[1, 0].axhline(
        y=llm_win_rate,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"LLM Strategy ({llm_win_rate:.1f}%)",
    )

    for bar, value in zip(bars3, category_win_rates):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1, 0].set_title("Win Rates by Strategy Category", fontweight="bold")
    axes[1, 0].set_ylabel("Win Rate (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Strategy counts and performance ranges
    strategy_counts = category_stats["strategy_count"].values
    performance_ranges = category_stats["best_return"] - category_stats["worst_return"]

    bars4 = axes[1, 1].bar(
        categories, strategy_counts, alpha=0.7, color="purple", label="Strategy Count"
    )

    # Add performance range as line
    ax2 = axes[1, 1].twinx()
    ax2.plot(
        categories,
        performance_ranges,
        "o-",
        color="red",
        linewidth=2,
        label="Performance Range",
    )
    ax2.set_ylabel("Performance Range (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    for bar, count in zip(bars4, strategy_counts):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    axes[1, 1].set_title("Strategy Counts & Performance Ranges", fontweight="bold")
    axes[1, 1].set_ylabel("Number of Strategies")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)

    # Create combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Strategy category performance plot saved: {output_path}")


def create_indicator_performance_plot(
    indicator_performance: Dict, output_path: str
) -> None:
    """
    Create comprehensive indicator-specific performance analysis plot.

    Args:
        indicator_performance: Dict from analyze_indicator_specific_performance()
        output_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter out errors
    valid_indicators = {
        k: v for k, v in indicator_performance.items() if "error" not in v
    }

    if not valid_indicators:
        print(f"Warning: No valid indicator performance data for plot")
        return

    # Set style (fallback if seaborn not available)
    try:
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        # Fallback to matplotlib default styling
        plt.style.use("default")

    plt.rcParams["figure.figsize"] = (16, 10)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Indicator-Specific LLM Performance Analysis", fontsize=16, fontweight="bold"
    )

    indicators = list(valid_indicators.keys())

    # Plot 1: Returns comparison (signal vs no signal)
    signal_returns = [
        v.get("llm_return_when_signal", 0) * 100 for v in valid_indicators.values()
    ]
    no_signal_returns = [
        v.get("llm_return_when_no_signal", 0) * 100 for v in valid_indicators.values()
    ]

    x = np.arange(len(indicators))
    width = 0.35

    bars1 = axes[0, 0].bar(
        x - width / 2,
        signal_returns,
        width,
        alpha=0.8,
        color="darkblue",
        label="When Indicator Signals",
    )
    bars2 = axes[0, 0].bar(
        x + width / 2,
        no_signal_returns,
        width,
        alpha=0.8,
        color="lightblue",
        label="When No Signal",
    )

    # Add value labels
    for bar, value in zip(bars1, signal_returns):
        if abs(value) > 1:  # Only label significant values
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + np.sign(value) * 2,
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

    for bar, value in zip(bars2, no_signal_returns):
        if abs(value) > 1:
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + np.sign(value) * 2,
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

    axes[0, 0].set_title("Returns by Indicator Signal State", fontweight="bold")
    axes[0, 0].set_ylabel("Total Return (%)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(indicators, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Plot 2: Win rates comparison
    signal_win_rates = [
        v.get("signal_win_rate", 0) * 100 for v in valid_indicators.values()
    ]
    no_signal_win_rates = [
        v.get("no_signal_win_rate", 0) * 100 for v in valid_indicators.values()
    ]

    bars3 = axes[0, 1].bar(
        x - width / 2,
        signal_win_rates,
        width,
        alpha=0.8,
        color="darkgreen",
        label="When Indicator Signals",
    )
    bars4 = axes[0, 1].bar(
        x + width / 2,
        no_signal_win_rates,
        width,
        alpha=0.8,
        color="lightgreen",
        label="When No Signal",
    )

    for bar, value in zip(bars3, signal_win_rates):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    for bar, value in zip(bars4, no_signal_win_rates):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    axes[0, 1].set_title("Win Rates by Indicator Signal State", fontweight="bold")
    axes[0, 1].set_ylabel("Win Rate (%)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(indicators, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(
        y=50, color="black", linestyle="--", alpha=0.5, label="Random (50%)"
    )

    # Plot 3: Signal frequency and performance differential
    signal_percentages = [
        v.get("signal_percentage", 0) for v in valid_indicators.values()
    ]
    performance_diffs = [
        v.get("performance_differential", 0) * 100 for v in valid_indicators.values()
    ]

    bars5 = axes[1, 0].bar(
        indicators,
        signal_percentages,
        alpha=0.7,
        color="orange",
        label="Signal Frequency (%)",
    )

    for bar, value in zip(bars5, signal_percentages):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    axes[1, 0].set_title("Indicator Signal Frequency", fontweight="bold")
    axes[1, 0].set_ylabel("Percentage of Days with Signals (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Performance differential (signal vs no signal)
    colors = ["green" if x > 0 else "red" for x in performance_diffs]
    bars6 = axes[1, 1].bar(indicators, performance_diffs, alpha=0.8, color=colors)

    for bar, value in zip(bars6, performance_diffs):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + np.sign(value) * 0.5,
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    axes[1, 1].set_title(
        "Performance Differential (Signal - No Signal)", fontweight="bold"
    )
    axes[1, 1].set_ylabel("Return Difference (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Indicator performance analysis plot saved: {output_path}")


def create_technical_indicators_plot(
    features_df: pd.DataFrame,
    decisions_df: pd.DataFrame = None,
    model_tag: str = "indicators",
    output_path: str = None,
):
    """
    Create comprehensive technical indicators plot.

    Shows available indicators: Price, RSI (if available), and optionally trading decisions.
    Gracefully handles missing indicator data.
    """
    # Check what indicators are available
    has_rsi = "rsi_14" in features_df.columns
    has_macd = "macd_line" in features_df.columns
    has_stoch = "stoch_k" in features_df.columns
    has_bb = "bb_upper" in features_df.columns

    # Determine subplot layout based on available indicators
    n_indicators = sum([has_rsi, has_macd, has_stoch, has_bb])
    if n_indicators == 0:
        # No indicators available - just show price
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax1.plot(
            features_df["date"],
            features_df["close"],
            label="Close Price",
            color="black",
            alpha=0.8,
        )
        ax1.set_title(f"Price Chart - {model_tag} (No Technical Indicators)")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        axes = [ax1]
    else:
        # Create subplots for price + indicators
        fig, axes = plt.subplots(
            n_indicators + 1, 1, figsize=(15, 8 + 3 * n_indicators)
        )

        # Price subplot (always first)
        axes[0].plot(
            features_df["date"],
            features_df["close"],
            label="Close Price",
            color="black",
            alpha=0.8,
        )
        axes[0].set_title(f"Technical Indicators - {model_tag}")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI subplot (if available)
        subplot_idx = 1
        if has_rsi:
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["rsi_14"],
                label="RSI(14)",
                color="purple",
                linewidth=1.5,
            )
            axes[subplot_idx].axhline(
                y=70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)"
            )
            axes[subplot_idx].axhline(
                y=30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)"
            )
            axes[subplot_idx].axhline(
                y=50, color="gray", linestyle="-", alpha=0.5, label="Neutral (50)"
            )
            axes[subplot_idx].fill_between(
                features_df["date"], 30, 70, alpha=0.1, color="yellow"
            )
            axes[subplot_idx].set_title("RSI(14)")
            axes[subplot_idx].set_ylabel("RSI Value")
            axes[subplot_idx].legend()
            axes[subplot_idx].grid(True, alpha=0.3)
            subplot_idx += 1

        # MACD subplot (if available)
        if has_macd:
            # Plot MACD line and signal
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["macd_line"],
                label="MACD Line",
                color="blue",
                linewidth=1.5,
            )
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["macd_signal"],
                label="Signal Line",
                color="red",
                linewidth=1.5,
            )
            # Plot histogram
            colors = [
                "green" if x >= 0 else "red" for x in features_df["macd_histogram"]
            ]
            axes[subplot_idx].bar(
                features_df["date"],
                features_df["macd_histogram"],
                color=colors,
                alpha=0.7,
                label="Histogram",
            )
            axes[subplot_idx].set_title("MACD")
            axes[subplot_idx].set_ylabel("MACD Value")
            axes[subplot_idx].legend()
            axes[subplot_idx].grid(True, alpha=0.3)
            subplot_idx += 1

        # Stochastic subplot (if available)
        if has_stoch:
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["stoch_k"],
                label="%K",
                color="orange",
                linewidth=1.5,
            )
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["stoch_d"],
                label="%D",
                color="purple",
                linewidth=1.5,
            )
            axes[subplot_idx].axhline(
                y=80, color="red", linestyle="--", alpha=0.7, label="Overbought (80)"
            )
            axes[subplot_idx].axhline(
                y=20, color="green", linestyle="--", alpha=0.7, label="Oversold (20)"
            )
            axes[subplot_idx].fill_between(
                features_df["date"], 20, 80, alpha=0.1, color="yellow"
            )
            axes[subplot_idx].set_title("Stochastic Oscillator")
            axes[subplot_idx].set_ylabel("Stochastic Value")
            axes[subplot_idx].legend()
            axes[subplot_idx].grid(True, alpha=0.3)
            subplot_idx += 1

        # Bollinger Bands subplot (if available)
        if has_bb:
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["close"],
                label="Close Price",
                color="black",
                alpha=0.8,
            )
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["bb_upper"],
                label="Upper Band",
                color="red",
                linestyle="--",
                alpha=0.7,
            )
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["bb_middle"],
                label="Middle Band",
                color="blue",
                linestyle="-",
                alpha=0.7,
            )
            axes[subplot_idx].plot(
                features_df["date"],
                features_df["bb_lower"],
                label="Lower Band",
                color="green",
                linestyle="--",
                alpha=0.7,
            )
            axes[subplot_idx].fill_between(
                features_df["date"],
                features_df["bb_lower"],
                features_df["bb_upper"],
                alpha=0.1,
                color="yellow",
            )
            axes[subplot_idx].set_title("Bollinger Bands")
            axes[subplot_idx].set_ylabel("Price")
            axes[subplot_idx].legend()
            axes[subplot_idx].grid(True, alpha=0.3)

    # Add trading signals if provided (only to price chart)
    if decisions_df is not None:
        price_ax = axes[0]  # Always plot signals on price chart

        # For signals, we need to merge with whatever indicator data is available
        # Use RSI if available, otherwise just use basic signals
        if has_rsi:
            decisions_with_indicators = decisions_df.merge(
                features_df[["date", "rsi_14"]], on="date", how="left"
            )
        else:
            decisions_with_indicators = decisions_df.copy()

        buy_signals = decisions_with_indicators[
            decisions_with_indicators["decision"] == "BUY"
        ]
        sell_signals = decisions_with_indicators[
            decisions_with_indicators["decision"] == "SELL"
        ]

        # Add buy/sell signals to the most relevant subplot
        if has_rsi:
            # Find RSI subplot (should be axes[1] if it exists)
            rsi_ax_idx = 1 if has_rsi else None
            if rsi_ax_idx is not None:
                axes[rsi_ax_idx].scatter(
                    buy_signals["date"],
                    buy_signals["rsi_14"],
                    marker="^",
                    color="green",
                    s=80,
                    label="BUY Signal",
                    zorder=5,
                )
                axes[rsi_ax_idx].scatter(
                    sell_signals["date"],
                    sell_signals["rsi_14"],
                    marker="v",
                    color="red",
                    s=80,
                    label="SELL Signal",
                    zorder=5,
                )
                # Update RSI subplot title to include signals
                axes[rsi_ax_idx].set_title("RSI(14) with Trading Signals")
        else:
            # No RSI - add signals to price chart instead
            price_ax.scatter(
                buy_signals["date"],
                buy_signals["close"],
                marker="^",
                color="green",
                s=100,
                label="BUY Signal",
                zorder=5,
            )
            price_ax.scatter(
                sell_signals["date"],
                sell_signals["close"],
                marker="v",
                color="red",
                s=100,
                label="SELL Signal",
                zorder=5,
            )

    # Set date formatting for x-axis
    import matplotlib.dates as mdates

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Technical indicators plot saved to: {output_path}")
    plt.close()


def create_technical_indicators_timeline(
    features_df: pd.DataFrame,
    decisions_df: pd.DataFrame = None,
    model_tag: str = "indicators",
    output_path: str = None,
):
    """
    Create comprehensive technical indicators timeline with decision overlays.

    Shows the evolution of RSI, MACD, Stochastic, and Bollinger Bands over time
    with trading decision overlays for better visualization of signal timing.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Check what indicators are available
    has_rsi = "rsi_14" in features_df.columns
    has_macd = "macd_line" in features_df.columns
    has_stoch = "stoch_k" in features_df.columns
    has_bb = "bb_upper" in features_df.columns

    if not any([has_rsi, has_macd, has_stoch, has_bb]):
        print(f"[WARNING] No technical indicators available for timeline chart")
        return

    # Create subplots based on available indicators
    n_indicators = sum([has_rsi, has_macd, has_stoch, has_bb])
    fig, axes = plt.subplots(n_indicators, 1, figsize=(16, 4 * n_indicators))

    if n_indicators == 1:
        axes = [axes]  # Make it iterable

    subplot_idx = 0

    # RSI subplot (if available)
    if has_rsi:
        ax = axes[subplot_idx]
        ax.plot(
            features_df["date"],
            features_df["rsi_14"],
            label="RSI(14)",
            color="purple",
            linewidth=1.5,
        )
        ax.axhline(
            y=70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)"
        )
        ax.axhline(
            y=30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)"
        )
        ax.axhline(y=50, color="gray", linestyle="-", alpha=0.5, label="Neutral (50)")
        ax.fill_between(features_df["date"], 30, 70, alpha=0.1, color="yellow")
        ax.set_title(f"RSI(14) Timeline - {model_tag}", fontweight="bold")
        ax.set_ylabel("RSI Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add decision overlays if available
        if decisions_df is not None:
            decisions_with_rsi = decisions_df.merge(
                features_df[["date", "rsi_14"]], on="date", how="left"
            ).dropna()

            buy_signals = decisions_with_rsi[decisions_with_rsi["decision"] == "BUY"]
            sell_signals = decisions_with_rsi[decisions_with_rsi["decision"] == "SELL"]

            ax.scatter(
                buy_signals["date"],
                buy_signals["rsi_14"],
                marker="^",
                color="green",
                s=80,
                label="BUY Signal",
                zorder=5,
            )
            ax.scatter(
                sell_signals["date"],
                sell_signals["rsi_14"],
                marker="v",
                color="red",
                s=80,
                label="SELL Signal",
                zorder=5,
            )

        subplot_idx += 1

    # MACD subplot (if available)
    if has_macd:
        ax = axes[subplot_idx]
        ax.plot(
            features_df["date"],
            features_df["macd_line"],
            label="MACD Line",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            features_df["date"],
            features_df["macd_signal"],
            label="Signal Line",
            color="red",
            linewidth=1.5,
        )
        ax.bar(
            features_df["date"],
            features_df["macd_histogram"],
            color=["green" if x >= 0 else "red" for x in features_df["macd_histogram"]],
            alpha=0.7,
            label="Histogram",
            width=1,
        )
        ax.set_title(f"MACD Timeline - {model_tag}", fontweight="bold")
        ax.set_ylabel("MACD Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        subplot_idx += 1

    # Stochastic subplot (if available)
    if has_stoch:
        ax = axes[subplot_idx]
        ax.plot(
            features_df["date"],
            features_df["stoch_k"],
            label="%K",
            color="orange",
            linewidth=1.5,
        )
        ax.plot(
            features_df["date"],
            features_df["stoch_d"],
            label="%D",
            color="purple",
            linewidth=1.5,
        )
        ax.axhline(
            y=80, color="red", linestyle="--", alpha=0.7, label="Overbought (80)"
        )
        ax.axhline(
            y=20, color="green", linestyle="--", alpha=0.7, label="Oversold (20)"
        )
        ax.fill_between(features_df["date"], 20, 80, alpha=0.1, color="yellow")
        ax.set_title(f"Stochastic Oscillator Timeline - {model_tag}", fontweight="bold")
        ax.set_ylabel("Stochastic Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        subplot_idx += 1

    # Bollinger Bands subplot (if available)
    if has_bb:
        ax = axes[subplot_idx]
        ax.plot(
            features_df["date"],
            features_df["close"],
            label="Close Price",
            color="black",
            alpha=0.8,
        )
        ax.plot(
            features_df["date"],
            features_df["bb_upper"],
            label="Upper Band",
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            features_df["date"],
            features_df["bb_middle"],
            label="Middle Band",
            color="blue",
            linestyle="-",
            alpha=0.7,
        )
        ax.plot(
            features_df["date"],
            features_df["bb_lower"],
            label="Lower Band",
            color="green",
            linestyle="--",
            alpha=0.7,
        )
        ax.fill_between(
            features_df["date"],
            features_df["bb_lower"],
            features_df["bb_upper"],
            alpha=0.1,
            color="yellow",
        )
        ax.set_title(f"Bollinger Bands Timeline - {model_tag}", fontweight="bold")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    # Format dates for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle(
        f"Technical Indicators Timeline - {model_tag}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Technical indicators timeline saved to: {output_path}")
    plt.close()


def create_rsi_performance_analysis(
    parsed_df: pd.DataFrame, features_df: pd.DataFrame, model_tag: str, output_path: str
):
    """
    Analyze RSI distribution by decision type and performance correlation.
    Only generates analysis if RSI data is available.
    """
    # Check if RSI data is available
    if "rsi_14" not in features_df.columns:
        print(
            f"[WARNING] RSI data not available for {model_tag} - skipping RSI performance analysis"
        )
        return

    # Merge data
    analysis_df = parsed_df.merge(
        features_df[["date", "rsi_14"]], on="date", how="left"
    ).dropna()

    if len(analysis_df) == 0:
        print(f"[WARNING] No valid RSI data for analysis in {model_tag}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"RSI Performance Analysis - {model_tag}", fontsize=14)

    # 1. RSI distribution by decision
    decisions = ["BUY", "HOLD", "SELL"]
    colors = ["green", "blue", "red"]

    for decision, color in zip(decisions, colors):
        rsi_values = analysis_df[analysis_df["decision"] == decision]["rsi_14"]
        if len(rsi_values) > 0:
            axes[0, 0].hist(
                rsi_values,
                alpha=0.7,
                label=f"{decision} (n={len(rsi_values)})",
                color=color,
                bins=15,
                density=True,
            )

    axes[0, 0].set_title("RSI Distribution by Decision Type")
    axes[0, 0].set_xlabel("RSI Value")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 0].axvline(x=70, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].axvline(x=30, color="green", linestyle="--", alpha=0.7)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Win rate by RSI range
    rsi_bins = pd.cut(analysis_df["rsi_14"], bins=10)
    win_rates = analysis_df.groupby(rsi_bins, observed=True)["strategy_return"].apply(
        lambda x: (x > 0).mean() * 100
    )

    bin_labels = [
        f"{interval.left:.0f}-{interval.right:.0f}" for interval in win_rates.index
    ]
    axes[0, 1].bar(
        range(len(win_rates)),
        win_rates.values,
        tick_label=bin_labels,
        color="skyblue",
        alpha=0.8,
    )
    axes[0, 1].set_title("Win Rate by RSI Range")
    axes[0, 1].set_xlabel("RSI Range")
    axes[0, 1].set_ylabel("Win Rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. RSI levels for winning vs losing trades
    winning_trades = analysis_df[analysis_df["strategy_return"] > 0]
    losing_trades = analysis_df[analysis_df["strategy_return"] < 0]

    axes[1, 0].hist(
        winning_trades["rsi_14"],
        alpha=0.7,
        label=f"Winning (n={len(winning_trades)})",
        color="green",
        bins=15,
        density=True,
    )
    axes[1, 0].hist(
        losing_trades["rsi_14"],
        alpha=0.7,
        label=f"Losing (n={len(losing_trades)})",
        color="red",
        bins=15,
        density=True,
    )
    axes[1, 0].set_title("RSI Distribution: Winning vs Losing Trades")
    axes[1, 0].set_xlabel("RSI Value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    axes[1, 0].axvline(x=70, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(x=30, color="green", linestyle="--", alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. RSI momentum analysis
    analysis_df["rsi_change"] = analysis_df["rsi_14"].diff()
    rsi_momentum_bins = pd.cut(analysis_df["rsi_change"], bins=5)
    momentum_returns = analysis_df.groupby(rsi_momentum_bins, observed=True)[
        "strategy_return"
    ].mean()

    momentum_labels = [
        f"{interval.left:.2f}-{interval.right:.2f}"
        for interval in momentum_returns.index
    ]
    axes[1, 1].bar(
        range(len(momentum_returns)),
        momentum_returns.values * 100,
        tick_label=momentum_labels,
        color="orange",
        alpha=0.8,
    )
    axes[1, 1].set_title("Average Return by RSI Momentum")
    axes[1, 1].set_xlabel("RSI Change (Daily)")
    axes[1, 1].set_ylabel("Average Return (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] RSI performance analysis plot saved to: {output_path}")
    plt.close()
