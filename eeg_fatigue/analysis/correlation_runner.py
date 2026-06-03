import pandas as pd
from .correlation import (
    run_pearson_analysis, run_spearman_analysis,
    run_anova_analysis, generate_correlation_visualizations,
)


# Excluded from feature_cols in all modes.
_EXCLUDE: set[str] = {
    "subject", "condition", "session", "window",
    "fatigue_level", "fatigue_label",
    "z_score",
    "ratio",
}


def _detect_agg_mode(df):
    cols       = set(df.columns)
    core_bands = {"delta", "theta", "alpha", "beta", "gamma"}
    suffixed   = any(
        any(col.startswith(b + "_") for b in core_bands)
        for col in cols
    )
    if not suffixed:
        return "global"
    regions    = {"Frontal", "Central", "Parietal", "Temporal", "Occipital", "Other"}
    return "region" if any(col.endswith("_" + r) for col in cols for r in regions) else "channel"


def run_correlation_analysis(df_features):
    if df_features is None or df_features.empty:
        print("  [WARN] Feature DataFrame is empty — skipping correlation analysis.")
        return None, None

    agg_mode = _detect_agg_mode(df_features)
    print(f"  Total windows with features : {len(df_features)}")
    print(f"  Detected aggregation mode   : {agg_mode}")
    print(f"  Columns in DataFrame        : {len(df_features.columns)}")

    feature_cols = [c for c in df_features.columns if c not in _EXCLUDE]

    print(f"  Features for correlation    : {len(feature_cols)}")
    print(f"  Excluded                    : {sorted(_EXCLUDE & set(df_features.columns))}")
    for fc in feature_cols:
        print(f"    - {fc}")

    df_corr     = run_pearson_analysis(df_features, feature_cols)
    df_spearman = run_spearman_analysis(df_features, feature_cols)
    run_anova_analysis(df_features, feature_cols)
    generate_correlation_visualizations(df_features, df_corr)

    return df_corr, df_spearman