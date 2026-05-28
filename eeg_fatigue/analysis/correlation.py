import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from ..visualization.correlation_plots import plot_correlation_heatmap, plot_top_6_features_boxplots


def pearson_correlation(df_features, feature_cols, target="fatigue_level"):
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
    anova_results = []
    for feature in feature_cols:
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


def run_pearson_analysis(df_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Pearson Correlation Analysis")
    print("=" * 60)
    df_corr = pearson_correlation(df_features, feature_cols)
    if not df_corr.empty:
        print("\n  Top 10 features most correlated with fatigue level:")
        print(df_corr.head(10).to_string(index=False))
        print("\n  Bottom 10 features (least correlated):")
        print(df_corr.tail(10).to_string(index=False))
    else:
        print("  [WARN] No Pearson results.")
    return df_corr


def run_spearman_analysis(df_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Spearman Correlation Analysis")
    print("=" * 60)
    df_spearman = spearman_correlation(df_features, feature_cols)
    if not df_spearman.empty:
        print("\n  Top 10 features (Spearman):")
        print(df_spearman.head(10).to_string(index=False))
    else:
        print("  [WARN] No Spearman results.")
    return df_spearman


def run_anova_analysis(df_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("ANOVA (Kruskal-Wallis) per Feature across Fatigue Levels")
    print("=" * 60)
    df_anova = anova_kruskal_wallis(df_features, feature_cols)
    if not df_anova.empty:
        n_sig = (df_anova["significant"] == "Yes").sum()
        print(f"\n  Significant features (p < 0.05): {n_sig} / {len(df_anova)}")
        print("\n  Top 10 features with strongest group differences:")
        print(df_anova.head(10).to_string(index=False))
    else:
        print("  [WARN] No ANOVA results.")
    return df_anova


def generate_correlation_visualizations(df_features: pd.DataFrame, df_corr: pd.DataFrame, config=None) -> None:
    if not df_corr.empty:
        top_features = df_corr.head(15)["feature"].tolist()
        plot_correlation_heatmap(df_features, top_features)
        plot_top_6_features_boxplots(df_features, df_corr)
        plt.close("all")
        print("[OK] Correlation visualizations saved.")

