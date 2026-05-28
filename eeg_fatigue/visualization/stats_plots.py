import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..config import LEVEL_LABELS, LEVEL_COLORS


def plot_kruskal_wallis_boxplot(level_ratios: dict, results_dir: Path = Path("results")) -> plt.Figure:
    """Box plot of ratio per fatigue level with Kruskal-Wallis annotation."""
    all_levels  = sorted(level_ratios.keys())
    plot_data   = [level_ratios[lvl] for lvl in all_levels]
    plot_colors = [LEVEL_COLORS.get(lvl, "#aaaaaa") for lvl in all_levels]

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(all_levels)), 5))
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    tick_labels = [
        f"Level {lvl}\n{LEVEL_LABELS.get(lvl, '')}"
        for lvl in all_levels
    ]
    ax.set_xticks(range(1, len(all_levels) + 1))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("(theta+alpha)/beta Ratio")
    ax.set_title("Kruskal-Wallis: Ratio Distribution per Fatigue Level")
    ax.grid(axis="y", alpha=0.3)

    y_min = ax.get_ylim()[0]
    y_range = ax.get_ylim()[1] - y_min
    for i, lvl in enumerate(all_levels):
        ax.text(i + 1, y_min - 0.05 * y_range,
                f"n={len(level_ratios[lvl])}", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "kruskal_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_chi_square_distribution(contingency_table: np.ndarray, chi2_result: tuple,
                                 results_dir: Path = Path("results")) -> plt.Figure:
    """Stacked % bar and grouped count bar for Before vs After fatigue levels."""
    chi2, p_chi, _, cramers_v = chi2_result
    strength = ("Large"    if cramers_v > 0.5 else
                "Medium"   if cramers_v > 0.3 else
                "Small"    if cramers_v > 0.1 else "Negligible")

    all_levels = sorted(LEVEL_LABELS.keys())
    n_levels   = len(all_levels)

    # percentage per condition
    level_pcts = []
    for row in contingency_table:
        total = sum(row)
        level_pcts.append([100 * v / total if total > 0 else 0 for v in row])

    fig, axes = plt.subplots(1, 2, figsize=(max(12, 3 * n_levels), 5))
    fig.suptitle("Chi-Square: Fatigue Level Distribution — Before vs After Work",
                 fontsize=13, fontweight="bold")

    cond_labels = ["Before Work", "After Work"]

    # stacked bar
    bottom = np.zeros(2)
    for i, lvl in enumerate(all_levels):
        vals  = [level_pcts[c][i] for c in range(2)]
        color = LEVEL_COLORS.get(lvl, "#aaaaaa")
        axes[0].bar(cond_labels, vals, bottom=bottom,
                    color=color, alpha=0.85, label=LEVEL_LABELS.get(lvl, str(lvl)))
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                axes[0].text(j, b + v / 2, f"{v:.0f}%",
                             ha="center", va="center",
                             fontsize=10, fontweight="bold", color="white")
        bottom += np.array(vals)

    axes[0].set_ylabel("Percentage of Windows (%)")
    axes[0].set_title("Window Distribution by Fatigue Level")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)

    # grouped bar
    x = np.arange(n_levels)
    w = 0.35
    b_counts = [contingency_table[0, i] for i in range(n_levels)]
    a_counts = [contingency_table[1, i] for i in range(n_levels)]
    axes[1].bar(x - w / 2, b_counts, width=w, color="#3498db", alpha=0.85, label="Before work")
    axes[1].bar(x + w / 2, a_counts, width=w, color="#e74c3c", alpha=0.85, label="After work")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{LEVEL_LABELS.get(lvl, str(lvl))}\n(Level {lvl})" for lvl in all_levels],
        fontsize=8
    )
    axes[1].set_ylabel("Number of Windows")
    axes[1].set_title("Window Counts per Level")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.text(0.5, -0.02,
             f"Chi-square chi2={chi2:.3f}, p={p_chi:.4f}, "
             f"Cramer's V={cramers_v:.3f} ({strength} effect)",
             ha="center", fontsize=10, style="italic", color="#555555")

    plt.tight_layout()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "chisquare_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig