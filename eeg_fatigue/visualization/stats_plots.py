import numpy as np
import matplotlib.pyplot as plt
from ..config import LEVEL_LABELS, LEVEL_COLORS

def plot_kruskal_wallis_boxplot(level_ratios):
    """Box plot of ratio per fatigue level with Kruskal-Wallis annotation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = [level_ratios[lvl] for lvl in [0, 1, 2, 3]]
    plot_colors = [LEVEL_COLORS[lvl] for lvl in [0, 1, 2, 3]]
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([
        "Level 0\nNo Fatigue\n(Z < −0.674)",
        "Level 1\nLow Fatigue\n(−0.674 ≤ Z < 0)",
        "Level 2\nMild Fatigue\n(0 ≤ Z < 0.674)",
        "Level 3\nHigh Fatigue\n(Z ≥ 0.674)"
    ], fontsize=8)
    
    ax.set_ylabel("(θ+α)/β Ratio")
    ax.set_title("Kruskal-Wallis: Ratio Distribution per Fatigue Level")
    ax.grid(axis='y', alpha=0.3)

    for i, lvl in enumerate([0, 1, 2, 3]):
        ax.text(i+1, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={len(level_ratios[lvl])}", ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig("results/method1_kruskal_boxplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_chi_square_distribution(contingency_table, chi2_result):
    """Stacked % bar and grouped count bar for Before vs After fatigue levels."""
    chi2, p_chi, dof, cramers_v = chi2_result
    strength = "Large" if cramers_v > 0.5 else "Medium" if cramers_v > 0.3 else "Small" if cramers_v > 0.1 else "Negligible"

    totals = contingency_table
    level_pcts = []
    for row in totals:
        total = sum(row)
        level_pcts.append([100 * v / total for v in row])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Chi-Square: Fatigue Level Distribution — Before vs After Work",
                 fontsize=13, fontweight='bold')

    cond_labels = ["Before Work", "After Work"]
    bottom = np.zeros(2)
    for lvl in [0, 1, 2, 3]:
        vals = [level_pcts[c][lvl] for c in range(2)]
        axes[0].bar(cond_labels, vals, bottom=bottom,
                    color=LEVEL_COLORS[lvl], alpha=0.85, label=LEVEL_LABELS[lvl])
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                axes[0].text(i, b + v/2, f"{v:.0f}%", ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white')
        bottom += np.array(vals)

    axes[0].set_ylabel("Percentage of Windows (%)")
    axes[0].set_title("Window Distribution by Fatigue Level")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)

    x = np.arange(4)
    w = 0.35
    b_counts = [totals[0, lvl] for lvl in [0, 1, 2, 3]]
    a_counts = [totals[1, lvl] for lvl in [0, 1, 2, 3]]
    axes[1].bar(x - w/2, b_counts, width=w, color="#3498db", alpha=0.85, label="Before work")
    axes[1].bar(x + w/2, a_counts, width=w, color="#e74c3c", alpha=0.85, label="After work")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["No Fatigue\n(Level 0)", "Low Fatigue\n(Level 1)",
                             "Mild Fatigue\n(Level 2)", "High Fatigue\n(Level 3)"], fontsize=8)
    axes[1].set_ylabel("Number of Windows")
    axes[1].set_title("Window Counts per Level")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.02,
             f"Chi-square χ²={chi2:.3f}, p={p_chi:.4f}, Cramér's V={cramers_v:.3f} ({strength} effect)",
             ha='center', fontsize=10, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig("results/method3_chisquare_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig
