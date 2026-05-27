import numpy as np
from ..config import LEVEL_LABELS, LEVEL_COLORS

def classify_fatigue_level(z_score):
    """Map Z-score to fatigue level (quartile-based, 4 levels)."""
    if np.isnan(z_score):
        return np.nan
    if z_score < -0.674:
        return 0
    elif z_score < 0.0:
        return 1
    elif z_score < 0.674:
        return 2
    else:
        return 3


def build_baseline(results, subject):
    """Extract all before-session window ratios and return (mean, std). Returns (None, None) if insufficient data."""
    before_ratios_all = []
    for session_ratios in results[subject]["before"].values():
        before_ratios_all.extend([r for r in session_ratios if not np.isnan(r)])

    if len(before_ratios_all) < 2:
        return None, None

    baseline_mean = np.mean(before_ratios_all)
    baseline_std = np.std(before_ratios_all, ddof=1)

    if baseline_std == 0:
        return None, None

    return baseline_mean, baseline_std


def compute_all_labels(results, valid_subjects):
    """Compute fatigue levels for all subjects, conditions, sessions. Returns dict."""
    fatigue_levels = {}

    for subject in valid_subjects:
        fatigue_levels[subject] = {"before": {}, "after": {}}

        baseline_mean, baseline_std = build_baseline(results, subject)
        if baseline_mean is None:
            print(f"[SKIP] {subject}: not enough before windows to build baseline.")
            continue

        print(f"\n{subject} → baseline from {sum(len(v) for v in results[subject]['before'].values())} windows: "
              f"mean={baseline_mean:.4f}, std={baseline_std:.4f}")

        for condition in ["before", "after"]:
            for session, window_ratios in results[subject][condition].items():
                window_levels = []
                for w_idx, ratio in enumerate(window_ratios):
                    z = (ratio - baseline_mean) / baseline_std
                    level = classify_fatigue_level(z)
                    window_levels.append(level)
                    print(f"  [{condition}] Session {session} Window {w_idx+1}: "
                          f"ratio={ratio:.4f}, Z={z:.2f} → {LEVEL_LABELS.get(level, 'NaN')}")
                fatigue_levels[subject][condition][session] = window_levels

    return fatigue_levels
