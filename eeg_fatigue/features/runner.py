from .feature_extraction import extract_all_subjects


def extract_features(subjects: list[str], cfg) -> dict:
    """Extract band powers and ratios for all subjects."""
    return extract_all_subjects(subjects, cfg)
