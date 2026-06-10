import os
import numpy as np
import pandas as pd

from ..features.band_power import compute_band_power_per_window
from ..features.ratios import compute_ratios
from ..labeling.fatigue import classify_fatigue_level


def extract_features_and_labels(subjects, cfg) -> pd.DataFrame:
    agg_mode = getattr(cfg, "AGG_MODE", "global")
    channel_regions = getattr(cfg, "CHANNEL_REGIONS", None)
    print(f"\n  Aggregation mode: {agg_mode}")

    rows = []
    for subject in subjects:
        print(f"\n  Processing {subject}...")

        before_tab = []
        for session in range(1, cfg.NUM_SESSIONS + 1):
            fp = os.path.join(cfg.BEFORE_PATH, subject, f"{session}.set")
            if not os.path.isfile(fp):
                continue
            for pw in compute_band_power_per_window(fp, cfg.BANDS, cfg.WINDOW_SEC, cfg.OVERLAP,
                                                    agg_mode=agg_mode, channel_regions=channel_regions):
                r = compute_ratios(pw)
                if r and not np.isnan(r.get("theta_alpha_beta", np.nan)):
                    before_tab.append(r["theta_alpha_beta"])

        if len(before_tab) < 2:
            print(f"    [SKIP] {subject}: no valid baseline")
            continue
        baseline_mean = np.mean(before_tab)
        baseline_std = np.std(before_tab, ddof=1)
        if baseline_std == 0:
            print(f"    [SKIP] {subject}: baseline std is zero")
            continue

        for condition in ["before", "after"]:
            root = cfg.BEFORE_PATH if condition == "before" else cfg.AFTER_PATH
            for session in range(1, cfg.NUM_SESSIONS + 1):
                fp = os.path.join(root, subject, f"{session}.set")
                if not os.path.isfile(fp):
                    continue
                for w_idx, pw in enumerate(compute_band_power_per_window(
                    fp, cfg.BANDS, cfg.WINDOW_SEC, cfg.OVERLAP,
                    agg_mode=agg_mode, channel_regions=channel_regions
                )):
                    ratios = compute_ratios(pw)
                    if ratios is None:
                        continue
                    tab = ratios.get("theta_alpha_beta", np.nan)
                    if np.isnan(tab):
                        continue
                    z = (tab - baseline_mean) / baseline_std
                    level = classify_fatigue_level(z, cfg)
                    if np.isnan(level):
                        continue

                    row = {
                        "subject": subject,
                        "condition": condition,
                        "session": session,
                        "window": w_idx + 1,
                        "fatigue_level": int(level),
                    }
                    for key, val in pw.items():
                        row[f"{key}_power"] = val
                    for name, val in ratios.items():
                        if name != "theta_alpha_beta" and name not in row:
                            row[name] = val
                    rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  Total windows extracted: {len(df)}")
    return df
