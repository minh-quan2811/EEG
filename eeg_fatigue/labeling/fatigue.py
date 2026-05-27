import numpy as np
from ..config import Z_THRESHOLDS, LEVEL_LABELS

def _validate_config():
    expected_levels = len(Z_THRESHOLDS) + 1
    if len(LEVEL_LABELS) != expected_levels:
        raise ValueError(
            f"Config mismatch: {len(Z_THRESHOLDS)} Z_THRESHOLDS require "
            f"{expected_levels} LEVEL_LABELS, but got {len(LEVEL_LABELS)}."
        )
    if Z_THRESHOLDS != sorted(Z_THRESHOLDS):
        raise ValueError("Z_THRESHOLDS must be sorted in ascending order.")

def classify_fatigue_level(z_score):
    if np.isnan(z_score):
        return np.nan
    for level, threshold in enumerate(Z_THRESHOLDS):
        if z_score < threshold:
            return level
    return len(Z_THRESHOLDS)


def build_baseline(results, subject):
    before_ratios = []
    for session_ratios in results[subject]["before"].values():
        before_ratios.extend([r for r in session_ratios if not np.isnan(r)])
    if len(before_ratios) < 2:
        return None, None
    mean = np.mean(before_ratios)
    std = np.std(before_ratios, ddof=1)
    if std == 0:
        return None, None
    return mean, std


def compute_all_labels(results, valid_subjects):
    _validate_config()
    n_levels = len(Z_THRESHOLDS) + 1
    print(f"Fatigue classification: {n_levels} levels, "
          f"thresholds = {Z_THRESHOLDS}")
    fatigue_levels = {}
    for subject in valid_subjects:
        fatigue_levels[subject] = {"before": {}, "after": {}}
        baseline_mean, baseline_std = build_baseline(results, subject)
        if baseline_mean is None:
            print(f"  [SKIP] {subject}: not enough before windows for baseline.")
            continue
        n_before = sum(len(v) for v in results[subject]["before"].values())
        print(f"\n  {subject} -> baseline from {n_before} windows: "
              f"mean={baseline_mean:.4f}, std={baseline_std:.4f}")
        for condition in ["before", "after"]:
            for session, window_ratios in results[subject][condition].items():
                window_levels = []
                for w_idx, ratio in enumerate(window_ratios):
                    z = (ratio - baseline_mean) / baseline_std
                    level = classify_fatigue_level(z)
                    window_levels.append(level)
                    label = LEVEL_LABELS.get(level, "NaN")
                    print(f"    [{condition}] Session {session} Window {w_idx+1}: "
                          f"ratio={ratio:.4f}, Z={z:.2f} -> {label}")
                fatigue_levels[subject][condition][session] = window_levels
    return fatigue_levels