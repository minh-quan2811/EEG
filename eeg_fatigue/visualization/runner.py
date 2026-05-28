import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .ratio_plots import plot_ratio_analysis
from .stats_plots import plot_kruskal_wallis_boxplot, plot_chi_square_distribution


def visualize_ratio_analysis(results: dict, fatigue_levels: dict,
                            subjects: list[str], results_dir: Path) -> None:
    """Generate before/after ratio analysis visualization."""
    plot_ratio_analysis(results, fatigue_levels, subjects, results_dir)
    plt.close("all")
    print("[OK] Ratio analysis visualization saved.")


def visualize_statistics(level_ratios: dict, contingency_table: np.ndarray,
                        chi2_result: tuple, results_dir: Path) -> None:
    """Generate statistical visualizations (boxplots, chi-square plots)."""
    plot_kruskal_wallis_boxplot(level_ratios, results_dir)
    plot_chi_square_distribution(contingency_table, chi2_result, results_dir)
    plt.close("all")
    print("[OK] Main visualizations saved.")
