import os
import numpy as np
import pandas as pd


def build_summary_df(results, fatigue_levels, valid_subjects, config):
    rows = []
    for subject in valid_subjects:
        before_ratios = [
            r for sess in results[subject]["before"].values()
            for r in sess if not np.isnan(r)
        ]
        if len(before_ratios) < 2:
            continue
        baseline_mean = np.mean(before_ratios)
        baseline_std = np.std(before_ratios, ddof=1)
        if baseline_std == 0:
            continue

        for condition in ["before", "after"]:
            for session in range(1, config.NUM_SESSIONS + 1):
                window_ratios = results[subject][condition].get(session, [])
                window_levels = fatigue_levels[subject][condition].get(session, [])
                for w_idx, (ratio, level) in enumerate(zip(window_ratios, window_levels)):
                    z = (ratio - baseline_mean) / baseline_std if not np.isnan(ratio) else np.nan
                    rows.append({
                        "subject": subject,
                        "condition": condition,
                        "session": session,
                        "window": w_idx + 1,
                        "ratio": ratio,
                        "z_score": z,
                        "fatigue_level": level,
                        "fatigue_label": config.LEVEL_LABELS.get(level, "NaN"),
                    })
    return pd.DataFrame(rows)


def build_feature_df(valid_subjects, fatigue_levels, config, band_power_func, ratios_func):
    agg_mode = getattr(config, "AGG_MODE", "global")
    channel_regions = getattr(config, "CHANNEL_REGIONS", None)
    all_rows = []

    for subject in valid_subjects:
        before_tab = []
        for session in range(1, config.NUM_SESSIONS + 1):
            filepath = os.path.join(config.BEFORE_PATH, subject, f"{session}.set")
            if not os.path.isfile(filepath):
                continue
            for pw in band_power_func(filepath, config.BANDS, window_sec=config.WINDOW_SEC,
                                      overlap=config.OVERLAP, agg_mode=agg_mode,
                                      channel_regions=channel_regions):
                r = ratios_func(pw)
                if r and not np.isnan(r.get("theta_alpha_beta", np.nan)):
                    before_tab.append(r["theta_alpha_beta"])

        if len(before_tab) < 2:
            print(f"  [SKIP] {subject}: not enough BEFORE windows for baseline.")
            continue
        baseline_mean = np.mean(before_tab)
        baseline_std = np.std(before_tab, ddof=1)
        if baseline_std == 0:
            print(f"  [SKIP] {subject}: baseline std is zero.")
            continue

        for condition in ["before", "after"]:
            root = config.BEFORE_PATH if condition == "before" else config.AFTER_PATH
            for session in range(1, config.NUM_SESSIONS + 1):
                filepath = os.path.join(root, subject, f"{session}.set")
                if not os.path.isfile(filepath):
                    continue

                window_powers = band_power_func(
                    filepath, config.BANDS,
                    window_sec=config.WINDOW_SEC, overlap=config.OVERLAP,
                    agg_mode=agg_mode, channel_regions=channel_regions,
                )
                if not window_powers:
                    continue

                session_levels = fatigue_levels.get(subject, {}).get(condition, {}).get(session, [])

                for w_idx, pw in enumerate(window_powers):
                    ratios = ratios_func(pw)
                    if ratios is None:
                        continue

                    tab = ratios.get("theta_alpha_beta", np.nan)
                    z = (tab - baseline_mean) / baseline_std if not np.isnan(tab) else np.nan
                    level = session_levels[w_idx] if w_idx < len(session_levels) else np.nan

                    row = {
                        "subject": subject,
                        "condition": condition,
                        "session": session,
                        "window": w_idx + 1,
                        "z_score": z,
                        "fatigue_level": level,
                        "fatigue_label": config.LEVEL_LABELS.get(level, "NaN"),
                    }
                    for key, value in pw.items():
                        row[key] = value
                    for name, value in ratios.items():
                        if name not in row:
                            row[name] = value
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if "fatigue_level" in df.columns:
        df = df.dropna(subset=["fatigue_level"])
    return df
