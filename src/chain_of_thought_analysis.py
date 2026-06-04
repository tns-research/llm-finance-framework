# src/chain_of_thought_analysis.py
"""
Chain of Thought Reasoning Quality Analysis

This module provides specialized analysis functions for evaluating the quality
and effectiveness of chain of thought reasoning in LLM trading decisions.
"""

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_chain_of_thought_quality(parsed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze quality and characteristics of chain of thought reasoning.

    Returns metrics on:
    - Content length and vocabulary richness
    - Analytical framework completeness (5-step structure)
    - Reasoning consistency and depth
    - Quality correlation with decision confidence

    Args:
        parsed_df: DataFrame with chain_of_thought column

    Returns:
        Dictionary with quality analysis metrics
    """
    if "chain_of_thought" not in parsed_df.columns:
        return {"error": "chain_of_thought column not found in data"}

    cot_texts = parsed_df["chain_of_thought"].dropna()

    if len(cot_texts) == 0:
        return {"error": "No chain of thought data available"}

    results = {
        "total_reasoning_entries": len(cot_texts),
        "average_reasoning_length": cot_texts.str.len().mean(),
        "reasoning_length_std": cot_texts.str.len().std(),
        "reasoning_length_distribution": {
            "short": (cot_texts.str.len() < 100).sum(),
            "medium": (
                (cot_texts.str.len() >= 100) & (cot_texts.str.len() < 300)
            ).sum(),
            "long": (cot_texts.str.len() >= 300).sum(),
        },
    }

    # Framework completeness scoring (5-step analytical process)
    framework_indicators = [
        "market regime",
        "market conditions",
        "volatility",
        "trend",
        "technical indicator",
        "rsi",
        "macd",
        "stochastic",
        "bollinger",
        "risk assessment",
        "volatility",
        "drawdown",
        "position sizing",
        "strategic consideration",
        "performance",
        "market timing",
        "capital preservation",
        "decision synthesis",
        "final recommendation",
        "rationale",
    ]

    completeness_scores = []
    for text in cot_texts:
        text_lower = text.lower()
        matches = sum(
            1 for indicator in framework_indicators if indicator in text_lower
        )
        completeness_scores.append(min(matches / len(framework_indicators), 1.0))

    results["framework_completeness"] = {
        "mean_score": np.mean(completeness_scores),
        "std_score": np.std(completeness_scores),
        "high_quality": (np.array(completeness_scores) > 0.7).sum(),
        "low_quality": (np.array(completeness_scores) < 0.3).sum(),
    }

    # Correlation with decision confidence
    if "prob" in parsed_df.columns:
        confidence_scores = parsed_df.loc[cot_texts.index, "prob"]
        correlation = np.corrcoef(completeness_scores, confidence_scores)[0, 1]
        results["confidence_correlation"] = correlation

    return results


def correlate_reasoning_with_performance(parsed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Correlate reasoning quality with trading performance.

    Analyzes whether higher quality reasoning leads to better returns,
    more consistent decisions, or improved risk-adjusted performance.

    Args:
        parsed_df: DataFrame with chain_of_thought, strategy_return, and other columns

    Returns:
        Dictionary with performance correlation analysis
    """
    if (
        "chain_of_thought" not in parsed_df.columns
        or "strategy_return" not in parsed_df.columns
    ):
        return {
            "error": "Required columns (chain_of_thought, strategy_return) not found"
        }

    # Filter to entries with reasoning data
    df_with_cot = parsed_df.dropna(subset=["chain_of_thought"])
    if len(df_with_cot) == 0:
        return {"error": "No data with chain of thought available"}

    # Calculate reasoning quality scores
    quality_scores = []
    for cot_text in df_with_cot["chain_of_thought"]:
        # Simple quality score based on length and analytical terms
        length_score = min(len(cot_text) / 300, 1.0)  # Normalize to 300 chars

        analytical_terms = [
            "market",
            "technical",
            "risk",
            "strategic",
            "analysis",
            "assessment",
        ]
        term_count = sum(1 for term in analytical_terms if term in cot_text.lower())
        term_score = min(term_count / len(analytical_terms), 1.0)

        quality_scores.append((length_score + term_score) / 2)

    # Performance correlation analysis
    returns = df_with_cot["strategy_return"].values
    quality_scores = np.array(quality_scores)

    results = {
        "sample_size": len(df_with_cot),
        "quality_score_distribution": {
            "mean": np.mean(quality_scores),
            "std": np.std(quality_scores),
            "high_quality": (quality_scores > 0.7).sum(),
            "low_quality": (quality_scores < 0.3).sum(),
        },
        "return_distribution": {
            "mean_return": np.mean(returns),
            "win_rate": (returns > 0).mean(),
            "avg_win": returns[returns > 0].mean() if (returns > 0).any() else 0,
            "avg_loss": returns[returns < 0].mean() if (returns < 0).any() else 0,
        },
    }

    # Correlation analysis
    if len(quality_scores) > 1:
        # Quality vs Returns correlation
        quality_return_corr = np.corrcoef(quality_scores, returns)[0, 1]
        results["quality_return_correlation"] = quality_return_corr

        # Quality vs Win Rate by quality terciles
        tercile_size = len(quality_scores) // 3
        sorted_indices = np.argsort(quality_scores)
        low_quality_returns = returns[sorted_indices[:tercile_size]]
        high_quality_returns = returns[sorted_indices[-tercile_size:]]

        results["quality_tercile_analysis"] = {
            "low_quality_win_rate": (low_quality_returns > 0).mean(),
            "high_quality_win_rate": (high_quality_returns > 0).mean(),
            "win_rate_difference": (high_quality_returns > 0).mean()
            - (low_quality_returns > 0).mean(),
        }

    return results


def create_reasoning_quality_visualizations(
    parsed_df: pd.DataFrame, model_tag: str, output_dir: str
):
    """
    Create visualizations for chain of thought reasoning quality analysis.

    Generates plots showing:
    - Reasoning quality distribution
    - Quality vs performance correlation
    - Reasoning patterns by market conditions

    Args:
        parsed_df: DataFrame with chain_of_thought and performance data
        model_tag: Model identifier for plot titles
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    if "chain_of_thought" not in parsed_df.columns:
        print("[WARN] No chain_of_thought data available for visualizations")
        return

    # Calculate quality scores
    quality_scores = []
    cot_texts = parsed_df["chain_of_thought"].dropna()

    for cot_text in cot_texts:
        length_score = min(len(cot_text) / 300, 1.0)
        analytical_terms = [
            "market",
            "technical",
            "risk",
            "strategic",
            "analysis",
            "assessment",
        ]
        term_count = sum(1 for term in analytical_terms if term in cot_text.lower())
        term_score = min(term_count / len(analytical_terms), 1.0)
        quality_scores.append((length_score + term_score) / 2)

    if len(quality_scores) == 0:
        print("[WARN] No valid quality scores calculated")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Chain of Thought Reasoning Quality Analysis - {model_tag}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Quality score distribution
    ax1 = axes[0, 0]
    ax1.hist(quality_scores, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Reasoning Quality Score", fontweight="bold")
    ax1.set_ylabel("Frequency", fontweight="bold")
    ax1.set_title("Distribution of Reasoning Quality Scores", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Quality vs Performance correlation
    ax2 = axes[0, 1]
    if "strategy_return" in parsed_df.columns:
        returns = parsed_df.loc[cot_texts.index, "strategy_return"]
        ax2.scatter(quality_scores, returns, alpha=0.6, color="darkgreen")
        ax2.set_xlabel("Reasoning Quality Score", fontweight="bold")
        ax2.set_ylabel("Strategy Return", fontweight="bold")
        ax2.set_title("Reasoning Quality vs Performance", fontweight="bold")
        ax2.grid(alpha=0.3)

        # Add correlation line
        if len(quality_scores) > 1:
            corr = np.corrcoef(quality_scores, returns)[0, 1]
            ax2.text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=ax2.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    # 3. Reasoning quality by decision type
    ax3 = axes[1, 0]
    if "decision" in parsed_df.columns:
        decisions = parsed_df.loc[cot_texts.index, "decision"]
        decision_quality = {}
        for decision in ["BUY", "HOLD", "SELL"]:
            mask = decisions == decision
            if mask.any():
                decision_quality[decision] = np.mean(
                    [q for q, m in zip(quality_scores, mask) if m]
                )

        if decision_quality:
            decisions_list = list(decision_quality.keys())
            quality_list = list(decision_quality.values())
            bars = ax3.bar(
                decisions_list, quality_list, color=["green", "gray", "red"], alpha=0.7
            )
            ax3.set_xlabel("Decision Type", fontweight="bold")
            ax3.set_ylabel("Average Reasoning Quality", fontweight="bold")
            ax3.set_title("Reasoning Quality by Decision Type", fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, quality_list):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # 4. Reasoning length vs quality correlation
    ax4 = axes[1, 1]
    reasoning_lengths = [len(text) for text in cot_texts]
    ax4.scatter(reasoning_lengths, quality_scores, alpha=0.6, color="purple")
    ax4.set_xlabel("Reasoning Length (characters)", fontweight="bold")
    ax4.set_ylabel("Quality Score", fontweight="bold")
    ax4.set_title("Reasoning Length vs Quality Correlation", fontweight="bold")
    ax4.grid(alpha=0.3)

    # Add correlation coefficient
    if len(reasoning_lengths) > 1:
        length_quality_corr = np.corrcoef(reasoning_lengths, quality_scores)[0, 1]
        ax4.text(
            0.05,
            0.95,
            f"Correlation: {length_quality_corr:.3f}",
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.8),
        )

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, f"{model_tag}_chain_of_thought_quality.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[INFO] Chain of thought quality visualizations saved to: {output_path}")
