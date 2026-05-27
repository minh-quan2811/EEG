import numpy as np
import pandas as pd
from scipy import stats

def pearson_correlation(df_features, feature_cols, target="fatigue_level"):
    """Compute Pearson correlation for each feature with target. Returns DataFrame."""
    correlations = []

    for feature in feature_cols:
        try:
            mask = df_features[feature].notna() & df_features[target].notna()
            x = pd.to_numeric(df_features.loc[mask, feature], errors='coerce').dropna().values
            y = pd.to_numeric(df_features.loc[mask, target], errors='coerce').dropna().values

            if len(x) < 3 or len(y) < 3:
                continue

            r, p_value = stats.pearsonr(x, y)
            correlations.append({
                "feature": feature,
                "pearson_r": r,
                "p_value": p_value,
                "significant": "Yes" if p_value < 0.05 else "No",
                "strength": "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak"
            })
        except (ValueError, TypeError):
            continue

    df_corr = pd.DataFrame(correlations)
    return df_corr.sort_values("pearson_r", ascending=False, key=abs) if len(correlations) > 0 else df_corr


def spearman_correlation(df_features, feature_cols, target="fatigue_level"):
    """Compute Spearman correlation (rank-based) for each feature with target. Returns DataFrame."""
    correlations = []

    for feature in feature_cols:
        try:
            mask = df_features[feature].notna() & df_features[target].notna()
            x = pd.to_numeric(df_features.loc[mask, feature], errors='coerce').dropna().values
            y = pd.to_numeric(df_features.loc[mask, target], errors='coerce').dropna().values

            if len(x) < 3 or len(y) < 3:
                continue

            rho, p_value = stats.spearmanr(x, y)
            correlations.append({
                "feature": feature,
                "spearman_rho": rho,
                "p_value": p_value,
                "significant": "Yes" if p_value < 0.05 else "No"
            })
        except (ValueError, TypeError):
            continue

    df_spearman = pd.DataFrame(correlations)
    return df_spearman.sort_values("spearman_rho", ascending=False, key=abs) if len(correlations) > 0 else df_spearman


def anova_kruskal_wallis(df_features, feature_cols, target="fatigue_level"):
    """Kruskal-Wallis test for each feature across fatigue levels. Returns DataFrame."""
    anova_results = []

    for feature in feature_cols:
        # Group by fatigue level
        groups = []
        for level in sorted(df_features[target].unique()):
            if not np.isnan(level):
                group = df_features[df_features[target] == level][feature].dropna()
                if len(group) >= 2:
                    groups.append(group)

        if len(groups) < 2:
            continue

        h_stat, p_value = stats.kruskal(*groups)
        anova_results.append({
            "feature": feature,
            "h_statistic": h_stat,
            "p_value": p_value,
            "significant": "Yes" if p_value < 0.05 else "No"
        })

    df_anova = pd.DataFrame(anova_results)
    return df_anova.sort_values("p_value")
