# src/report/charts.py
"""Local statistical-visualization chart generator.

Moved verbatim out of report_generator.py (pure structural split, no logic change).
Produces matplotlib PNGs on disk; consumed by generate_additional_charts.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def create_statistical_visualizations(
    validation_results: Dict, model_tag: str, plots_dir: Path
) -> Dict[str, str]:
    """
    Create visualizations of statistical validation results.
    """
    charts = {}

    # Bootstrap distribution plot
    if "bootstrap_vs_index" in validation_results:
        bootstrap_data = validation_results["bootstrap_vs_index"]

        if "all_results" in bootstrap_data:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(
                f"Statistical Validation Visualizations - {model_tag}",
                fontsize=14,
                fontweight="bold",
            )

            # 1. Bootstrap Distribution
            ax1 = axes[0]
            bootstrap_diffs = np.array(bootstrap_data["all_results"])
            ax1.hist(
                bootstrap_diffs,
                bins=50,
                alpha=0.7,
                density=True,
                label="Bootstrap Distribution",
            )
            ax1.axvline(
                bootstrap_data["sharpe_difference"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}',
            )
            ax1.axvline(
                np.mean(bootstrap_diffs),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(bootstrap_diffs):.3f}",
            )
            ax1.set_title("Bootstrap Sharpe Difference Distribution", fontweight="bold")
            ax1.set_xlabel("Sharpe Ratio Difference")
            ax1.set_ylabel("Density")
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. Confidence Interval
            ax2 = axes[1]
            ax2.hist(bootstrap_diffs, bins=30, alpha=0.7, density=True)
            ci_lower, ci_upper = bootstrap_data["ci_95_bootstrap"]
            ax2.axvline(
                ci_lower,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"95% CI Lower: {ci_lower:.3f}",
            )
            ax2.axvline(
                ci_upper,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"95% CI Upper: {ci_upper:.3f}",
            )
            ax2.axvline(
                bootstrap_data["sharpe_difference"],
                color="red",
                linestyle="-",
                linewidth=2,
                label=f'Observed: {bootstrap_data["sharpe_difference"]:.3f}',
            )
            ax2.set_title("Confidence Interval Analysis", fontweight="bold")
            ax2.set_xlabel("Sharpe Ratio Difference")
            ax2.set_ylabel("Density")
            ax2.legend()
            ax2.grid(alpha=0.3)

            plt.tight_layout()

            chart_path = plots_dir / f"{model_tag}_statistical_validation.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            charts["statistical_validation"] = f"../plots/{chart_path.name}"
            print(f"✓ Statistical validation visualization saved: {chart_path}")

    return charts
