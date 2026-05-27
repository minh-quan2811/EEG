import pandas as pd
import numpy as np

def build_dataframe(results, fatigue_levels, valid_subjects, config):
    """Build summary DataFrame from results and fatigue levels."""
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


def build_feature_dataframe(valid_subjects, config, band_power_func, ratios_func):
    """Build feature DataFrame from all sessions (for correlation analysis)."""
    all_features = []

    for subject in valid_subjects:
        for condition in ["before", "after"]:
            root = config.BEFORE_PATH if condition == "before" else config.AFTER_PATH
            sub_folder = f"{root}/{subject}"

            for session in range(1, config.NUM_SESSIONS + 1):
                filepath = f"{sub_folder}/{session}.set"

                power = band_power_func(filepath, config.BANDS)
                if power is None:
                    continue

                ratios = ratios_func(power)
                if ratios is None:
                    continue

                row = {
                    "subject": subject,
                    "condition": condition,
                    "session": session,
                }
                row.update({f"{band}_power": v for band, v in power.items()})
                row.update(ratios)

                all_features.append(row)

    return pd.DataFrame(all_features)


def save_results_csv(df, filename="results/eeg_fatigue_results.csv"):
    """Save DataFrame to CSV."""
    df.to_csv(filename, index=False)

    print(f"✓ Results saved to {filename}")
    
    return filename
