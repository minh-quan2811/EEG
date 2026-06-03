import os
import numpy as np

from . import band_power
from . import ratios as ratios_module


def extract_all_subjects(valid_subjects, config):
    agg_mode        = getattr(config, "AGG_MODE", "global")
    channel_regions = getattr(config, "CHANNEL_REGIONS", None)

    print(f"\n  Aggregation mode: {agg_mode}")

    results = {}
    for subject in valid_subjects:
        print(f"\n  Subject: {subject}")
        results[subject] = {"before": {}, "after": {}}

        for condition, root in [("before", config.BEFORE_PATH), ("after", config.AFTER_PATH)]:
            sub_folder    = os.path.join(root, subject)
            found_sessions = []

            for session in range(1, config.NUM_SESSIONS + 1):
                filepath = os.path.join(sub_folder, f"{session}.set")
                if not os.path.isfile(filepath):
                    continue

                window_powers = band_power.compute_band_power_per_window(
                    filepath, config.BANDS,
                    window_sec=config.WINDOW_SEC,
                    overlap=config.OVERLAP,
                    agg_mode=agg_mode,
                    channel_regions=channel_regions,
                )
                if not window_powers:
                    continue

                window_ratios = [ratios_module.compute_ratio(pw) for pw in window_powers]
                results[subject][condition][session] = window_ratios
                found_sessions.append(session)

                n_valid = sum(1 for r in window_ratios if not np.isnan(r))
                print(f"    [{condition}] Session {session} -> "
                      f"{len(window_ratios)} windows, {n_valid} valid")

            print(f"    [{condition}] Sessions loaded: {found_sessions}")

    print("\n[OK] Per-window ratios computed.")
    return results