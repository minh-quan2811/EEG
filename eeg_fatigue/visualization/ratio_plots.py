import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ..config import LEVEL_LABELS, LEVEL_COLORS

def plot_ratio_analysis(results, fatigue_levels, valid_subjects):
    """4-panel figure: Before vs After, Session trend, Fatigue distribution, Z-scores."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("EEG Fatigue Analysis — (θ+α)/β Ratio", fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(valid_subjects))
    w = 0.35
    b_vals = [np.nanmean([r for sess in results[s]["before"].values() for r in sess]) for s in valid_subjects]
    a_vals = [np.nanmean([r for sess in results[s]["after"].values() for r in sess]) for s in valid_subjects]

    ax1.bar(x - w/2, b_vals, width=w, color="#3498db", alpha=0.85, label="Before work")
    ax1.bar(x + w/2, a_vals, width=w, color="#e74c3c", alpha=0.85, label="After work")
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_subjects, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel("Mean (θ+α)/β Ratio")
    ax1.set_title("Mean Ratio: Before vs After (per subject)")
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.colormaps["tab10"].resampled(len(valid_subjects))
    for i, subject in enumerate(valid_subjects):
        after_sess = sorted(results[subject]["after"].keys())
        after_vals = [np.nanmean(results[subject]["after"][s]) for s in after_sess]
        ax2.plot(after_sess, after_vals, marker='o', markersize=5,
                 color=cmap(i), label=subject, linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Session Number")
    ax2.set_ylabel("(θ+α)/β Ratio")
    ax2.set_title("Ratio Trend Across After-Work Sessions")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for cond, color, offset in [("before", "#3498db", -0.2), ("after", "#e74c3c", 0.2)]:
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for subject in valid_subjects:
            for window_levels in fatigue_levels[subject][cond].values():
                for lvl in window_levels:
                    if not np.isnan(lvl):
                        counts[int(lvl)] += 1
        levels = list(counts.keys())
        vals = list(counts.values())
        ax3.bar([l + offset for l in levels], vals, width=0.35,
                color=color, alpha=0.85, label=f"{cond.capitalize()} work")

    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["No Fatigue\n(Z < −0.674)", "Low Fatigue\n(−0.674 ≤ Z < 0)",
                         "Mild Fatigue\n(0 ≤ Z < 0.674)", "High Fatigue\n(Z ≥ 0.674)"],
                        fontsize=8)
    ax3.set_ylabel("Number of Sessions")
    ax3.set_title("Fatigue Level Distribution")
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    for i, subject in enumerate(valid_subjects):
        before_ratios = [r for sess in results[subject]["before"].values()
                         for r in sess if not np.isnan(r)]
        if len(before_ratios) < 2:
            continue
        bm = np.mean(before_ratios)
        bsd = np.std(before_ratios, ddof=1)
        if bsd == 0:
            continue
        after_zs = [(r - bm) / bsd for sess in results[subject]["after"].values()
                    for r in sess if not np.isnan(r)]
        before_zs = [(r - bm) / bsd for sess in results[subject]["before"].values()
                     for r in sess if not np.isnan(r)]

        ax4.scatter([i - 0.15] * len(before_zs), before_zs, color="#3498db",
                    alpha=0.6, s=40, label="Before" if i == 0 else "")
        ax4.scatter([i + 0.15] * len(after_zs), after_zs, color="#e74c3c",
                    alpha=0.6, s=40, label="After" if i == 0 else "")

    ax4.axhline(-0.674, color="#f1c40f", linestyle="--", linewidth=1.2, label="Z=−0.674")
    ax4.axhline(0.0, color="#e67e22", linestyle="--", linewidth=1.2, label="Z=0")
    ax4.axhline(0.674, color="#e74c3c", linestyle="--", linewidth=1.2, label="Z=+0.674")
    ax4.set_xticks(range(len(valid_subjects)))
    ax4.set_xticklabels(valid_subjects, rotation=30, ha='right', fontsize=9)
    ax4.set_ylabel("Z-Score (distance from personal baseline)")
    ax4.set_title("Z-Score Distribution per Subject")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(alpha=0.3)

    plt.savefig("results/eeg_fatigue_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig
