import os
import numpy as np
import pandas as pd


def build_dataframe(results, fatigue_levels, valid_subjects, config):
    rows = []
    for subject in valid_subjects:
        before_ratios_all_windows = []
        for session_ratios in results[subject]["before"].values():
            before_ratios_all_windows.extend(
                [r for r in session_ratios if not np.isnan(r)]
            )
        if len(before_ratios_all_windows) < 2:
            continue
        baseline_mean = np.mean(before_ratios_all_windows)
        baseline_std = np.std(before_ratios_all_windows, ddof=1)
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


def build_feature_dataframe(valid_subjects, fatigue_levels, config,
                             band_power_func, ratios_func):
    all_rows = []
    for subject in valid_subjects:
        before_tab_values = []
        for session in range(1, config.NUM_SESSIONS + 1):
            filepath = os.path.join(config.BEFORE_PATH, subject, f"{session}.set")
            if not os.path.isfile(filepath):
                continue
            window_powers = band_power_func(
                filepath, config.BANDS,
                window_sec=config.WINDOW_SEC,
                overlap=config.OVERLAP,
            )
            for pw in window_powers:
                r = ratios_func(pw)
                if r and not np.isnan(r.get("theta_alpha_beta", np.nan)):
                    before_tab_values.append(r["theta_alpha_beta"])
        if len(before_tab_values) < 2:
            print(f"  [SKIP] {subject}: not enough BEFORE windows for baseline.")
            continue
        baseline_mean = np.mean(before_tab_values)
        baseline_std = np.std(before_tab_values, ddof=1)
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
                    window_sec=config.WINDOW_SEC,
                    overlap=config.OVERLAP,
                )
                if not window_powers:
                    continue
                session_levels = fatigue_levels.get(subject, {}) \
                                                   .get(condition, {}) \
                                                   .get(session, [])
                for w_idx, pw in enumerate(window_powers):
                    ratios = ratios_func(pw)
                    if ratios is None:
                        continue
                    tab = ratios.get("theta_alpha_beta", np.nan)
                    z = (tab - baseline_mean) / baseline_std \
                          if not np.isnan(tab) else np.nan
                    if w_idx < len(session_levels):
                        level = session_levels[w_idx]
                    else:
                        level = np.nan
                    row = {
                        "subject": subject,
                        "condition": condition,
                        "session": session,
                        "window": w_idx + 1,
                        "z_score": z,
                        "fatigue_level": level,
                        "fatigue_label": config.LEVEL_LABELS.get(level, "NaN"),
                    }
                    for band, value in pw.items():
                        row[f"{band}_power"] = value
                    for ratio_name, value in ratios.items():
                        row[ratio_name] = value
                    all_rows.append(row)
    df = pd.DataFrame(all_rows)
    if "fatigue_level" in df.columns:
        df = df.dropna(subset=["fatigue_level"])
    return df


def save_results_csv(df, filename: str = None, results_dir=None) -> str:
    from pathlib import Path
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)
    if filename is None:
        filename = "eeg_fatigue_results.csv"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return str(filepath)