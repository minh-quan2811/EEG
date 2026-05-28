from .fatigue import compute_all_labels


def label_fatigue(results: dict, subjects: list[str], cfg) -> dict:
    """Compute fatigue classifications for all windows."""
    fatigue_levels = compute_all_labels(results, subjects, cfg)
    print("[OK] Fatigue levels computed.")
    return fatigue_levels
