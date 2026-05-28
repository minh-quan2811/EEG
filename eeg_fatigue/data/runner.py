from .loader import get_valid_subjects


def load_subjects(cfg) -> list[str]:
    """Load and validate subjects from configured paths."""
    valid_subjects = get_valid_subjects(cfg.AFTER_PATH, cfg.BEFORE_PATH)
    if not valid_subjects:
        print("[ABORT] No valid subjects found. Check AFTER_PATH / BEFORE_PATH in config.py.")
        return []
    return valid_subjects
