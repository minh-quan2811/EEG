import os
import numpy as np

from . import band_power
from . import ratios as ratios_module


def compute_subject_ratios(valid_subjects):
    results = {}
    for subject in valid_subjects:
        results[subject] = {"before": {}, "after": {}}
        for condition, root in [("before", band_power.config.BEFORE_PATH), ("after", band_power.config.AFTER_PATH)]:
            sub_folder = os.path.join(root, subject)
            found_sessions = []
            for session in range(1, band_power.config.NUM_SESSIONS + 1):
                filepath = os.path.join(sub_folder, f"{session}.set")
                if not os.path.isfile(filepath):
                    continue
                window_powers = band_power.compute_band_power_per_window(
                    filepath,
                    band_power.config.BANDS,
                    window_sec=band_power.config.WINDOW_SEC,
                    overlap=band_power.config.OVERLAP,
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
    return results
