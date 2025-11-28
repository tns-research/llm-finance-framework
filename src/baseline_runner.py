# src/baseline_runner.py
"""
Standalone script to run baseline comparisons.

Can be run independently to analyze baselines without running LLM models,
or called from trading_engine to add baseline context to LLM results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .baselines import (
    run_all_baselines,
    compare_llm_to_baselines,
    print_baseline_comparison,
    enhanced_compare_llm_to_baselines,
    print_enhanced_baseline_comparison,
    calculate_llm_vs_random_stats,
    BASELINE_REGISTRY,
)


def save_baseline_results(
    baseline_df: pd.DataFrame,
    output_dir: str,
    filename: str = "baseline_comparison.csv"
) -> str:
    """Save baseline comparison results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    baseline_df.to_csv(output_path, index=False)
    print(f"[INFO] Baseline results saved to: {output_path}")
    return output_path


def create_baseline_comparison_plot(
    comparison_df: pd.DataFrame,
    model_tag: str,
    output_path: str
):
    """
    Create a bar chart comparing all strategies by total return.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Baseline Comparison - {model_tag}', fontsize=16, fontweight='bold')
    
    # Sort by total return for plotting
    df = comparison_df.sort_values("total_return", ascending=True)
    
    # Color based on whether it's the LLM or a baseline
    colors = ['steelblue' if name != 'LLM_STRATEGY' else 'darkorange' 
              for name in df["baseline"]]
    
    # 1. Total Return comparison
    ax1 = axes[0, 0]
    bars = ax1.barh(df["baseline"], df["total_return"], color=colors, alpha=0.8)
    ax1.set_xlabel("Total Return (%)", fontweight='bold')
    ax1.set_title("Total Return Comparison", fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df["total_return"]):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{val:.1f}%', va='center', fontsize=9)
    
    # 2. Sharpe Ratio comparison
    ax2 = axes[0, 1]
    sharpe_df = df[df["sharpe_annualized"].notna()]
    colors2 = ['steelblue' if name != 'LLM_STRATEGY' else 'darkorange' 
               for name in sharpe_df["baseline"]]
    bars2 = ax2.barh(sharpe_df["baseline"], sharpe_df["sharpe_annualized"], 
                     color=colors2, alpha=0.8)
    ax2.set_xlabel("Annualized Sharpe Ratio", fontweight='bold')
    ax2.set_title("Risk-Adjusted Return (Sharpe)", fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Max Drawdown comparison (lower is better, so we show negative)
    ax3 = axes[1, 0]
    dd_df = df[df["max_drawdown"].notna()]
    colors3 = ['steelblue' if name != 'LLM_STRATEGY' else 'darkorange' 
               for name in dd_df["baseline"]]
    bars3 = ax3.barh(dd_df["baseline"], dd_df["max_drawdown"], color=colors3, alpha=0.8)
    ax3.set_xlabel("Max Drawdown (%)", fontweight='bold')
    ax3.set_title("Maximum Drawdown (closer to 0 is better)", fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Win Rate comparison
    ax4 = axes[1, 1]
    wr_df = df[df["win_rate"].notna()]
    colors4 = ['steelblue' if name != 'LLM_STRATEGY' else 'darkorange' 
               for name in wr_df["baseline"]]
    bars4 = ax4.barh(wr_df["baseline"], wr_df["win_rate"], color=colors4, alpha=0.8)
    ax4.set_xlabel("Win Rate (%)", fontweight='bold')
    ax4.set_title("Win Rate (% of profitable days)", fontweight='bold')
    ax4.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% line')
    ax4.grid(axis='x', alpha=0.3)
    ax4.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Baseline comparison plot saved to: {output_path}")


def create_equity_curves_plot(
    features_df: pd.DataFrame,
    model_tag: str,
    output_path: str,
    llm_returns: pd.Series = None,
    llm_dates: pd.Series = None
):
    """
    Plot equity curves for all baselines (and optionally LLM).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for each baseline
    colors = {
        "random": "gray",
        "random_mean (n=30)": "darkgray",  # Statistical random mean
        "random_representative": "lightgray",  # Individual random run for equity curve
        "buy_and_hold": "black",
        "momentum": "blue",
        "contrarian": "purple",
        "mean_reversion": "green",
        "volatility_timing": "cyan",
        "momentum_vol_combined": "teal",
        "LLM_STRATEGY": "darkorange",
    }
    
    linestyles = {
        "buy_and_hold": "-",
        "momentum": "--",
        "contrarian": ":",
        "mean_reversion": "-.",
        "volatility_timing": "--",
        "momentum_vol_combined": "-",
        "random": ":",
        "random_mean (n=30)": "-",  # Solid line for statistical mean
        "random_representative": "--",  # Dashed for individual run
        "LLM_STRATEGY": "-",
    }
    
    linewidths = {
        "buy_and_hold": 2.5,
        "LLM_STRATEGY": 2.5,
    }
    
    # Use integer index for x-axis (trading day number)
    x_axis = range(len(features_df))
    
    # Plot each baseline
    for name, baseline_fn in BASELINE_REGISTRY.items():
        baseline_df = baseline_fn(features_df)
        equity = (1 + baseline_df["strategy_return"] / 100).cumprod()

        ax.plot(
            x_axis,
            equity,
            label=name,
            color=colors.get(name, "gray"),
            linestyle=linestyles.get(name, "-"),
            linewidth=linewidths.get(name, 1.5),
            alpha=0.8
        )

    # Add representative random baseline for equity curve visualization
    # This shows an individual random run alongside the statistical summary
    from .baselines import random_baseline
    random_representative_df = random_baseline(features_df, seed=42)  # Deterministic seed
    random_equity = (1 + random_representative_df["strategy_return"] / 100).cumprod()

    ax.plot(
        x_axis,
        random_equity,
        label="random_representative",
        color=colors.get("random_representative", "lightgray"),
        linestyle=linestyles.get("random_representative", "--"),
        linewidth=1.5,
        alpha=0.7
    )
    
    # Plot LLM if provided
    if llm_returns is not None:
        llm_equity = (1 + llm_returns / 100).cumprod()
        # LLM might have different length, use its own range
        llm_x = range(len(llm_equity))
        ax.plot(
            llm_x,
            llm_equity,
            label="LLM_STRATEGY",
            color=colors["LLM_STRATEGY"],
            linestyle="-",
            linewidth=2.5,
            alpha=1.0
        )
    
    ax.set_xlabel("Trading Day", fontweight='bold', fontsize=12)
    ax.set_ylabel("Equity (starting at 1.0)", fontweight='bold', fontsize=12)
    ax.set_title(f'Equity Curves Comparison - {model_tag}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Equity curves plot saved to: {output_path}")


def run_baseline_analysis(
    features_path: str,
    output_dir: str,
    llm_metrics: dict = None,
    llm_parsed_df: pd.DataFrame = None,
    model_tag: str = "analysis"
) -> pd.DataFrame:
    """
    Main function to run complete baseline analysis.
    
    Args:
        features_path: Path to features.csv
        output_dir: Directory to save results
        llm_metrics: Optional dict of LLM performance metrics
        llm_parsed_df: Optional DataFrame of LLM parsed results (for equity curve)
        model_tag: Tag for labeling outputs
    
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 60)
    print("RUNNING BASELINE ANALYSIS")
    print("=" * 60)
    
    # Load features
    features_df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"Loaded {len(features_df)} rows from {features_path}")
    
    # Handle START_ROW if configured
    from .config import START_ROW
    if START_ROW is not None:
        features_df = features_df.iloc[START_ROW:].reset_index(drop=True)
        print(f"After START_ROW={START_ROW}: {len(features_df)} rows")
    
    # Handle TEST_MODE limit
    from .config import TEST_MODE, TEST_LIMIT
    if TEST_MODE:
        features_df = features_df.head(TEST_LIMIT)
        print(f"TEST_MODE active: limited to {TEST_LIMIT} rows")
    
    # Run all baselines
    print("\nRunning baselines...")
    baseline_results = run_all_baselines(features_df)
    
    # Compare with LLM if provided
    if llm_metrics is not None:
        # Use enhanced statistical comparison
        comparison_df, random_stats = enhanced_compare_llm_to_baselines(
            llm_metrics, baseline_results, features_df, n_random_runs=30
        )

        # Calculate statistical significance
        llm_return = llm_metrics.get("total_return", 0)
        llm_stats = calculate_llm_vs_random_stats(llm_return, random_stats)

        # Print enhanced comparison
        print_enhanced_baseline_comparison(comparison_df, random_stats, llm_stats, model_tag)
    else:
        comparison_df = baseline_results.sort_values("total_return", ascending=False)
        # Print regular comparison (no LLM to compare)
        print_baseline_comparison(comparison_df, model_tag)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{model_tag}_baseline_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to: {csv_path}")
    
    # Create plots
    plots_dir = os.path.join(output_dir, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    bar_plot_path = os.path.join(plots_dir, f"{model_tag}_baseline_comparison.png")
    create_baseline_comparison_plot(comparison_df, model_tag, bar_plot_path)
    
    equity_plot_path = os.path.join(plots_dir, f"{model_tag}_equity_curves.png")
    llm_returns = llm_parsed_df["strategy_return"] if llm_parsed_df is not None else None
    llm_dates = llm_parsed_df["date"] if llm_parsed_df is not None else None
    create_equity_curves_plot(features_df, model_tag, equity_plot_path, llm_returns, llm_dates)
    
    return comparison_df


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Can be run standalone to analyze baselines without LLM
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    features_path = os.path.join(base_dir, "data", "processed", "features.csv")
    output_dir = os.path.join(base_dir, "results", "analysis")
    
    if not os.path.exists(features_path):
        print(f"ERROR: Features file not found at {features_path}")
        print("Run the main pipeline first to generate features.")
        sys.exit(1)
    
    run_baseline_analysis(
        features_path=features_path,
        output_dir=output_dir,
        model_tag="baselines_only"
    )

