from .correlation import (
    run_pearson_analysis, run_spearman_analysis,
    run_anova_analysis, generate_correlation_visualizations
)


def run_correlation_analysis(df_features: object) -> tuple:
    """Run correlation analysis (Pearson, Spearman, ANOVA) and generate visualizations."""
    if df_features is None or df_features.empty:
        print("  [WARN] Feature DataFrame is empty — skipping correlation analysis.")
        return None, None

    print(f"  Total windows with features: {len(df_features)}")
    print(f"  Columns: {list(df_features.columns)}\n")

    exclude_cols = {
        "subject", "condition", "session", "window",
        "fatigue_level", "fatigue_label", "z_score",
    }
    feature_cols = [c for c in df_features.columns if c not in exclude_cols]
    print(f"  Features for correlation: {len(feature_cols)}")
    for fc in feature_cols:
        print(f"    - {fc}")

    df_corr = run_pearson_analysis(df_features, feature_cols)
    df_spearman = run_spearman_analysis(df_features, feature_cols)
    run_anova_analysis(df_features, feature_cols)
    generate_correlation_visualizations(df_features, df_corr)
    return df_corr, df_spearman
