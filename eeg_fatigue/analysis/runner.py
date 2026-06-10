from .statistics import run_wilcoxon_analysis, run_kruskal_wallis_analysis, run_chi_square_analysis
from .correlation import run_pearson_analysis, run_spearman_analysis, run_anova_analysis


def run_analysis(results, fatigue_levels, subjects, cfg) -> dict:
    _, w_dict = run_wilcoxon_analysis(results, subjects, cfg)
    h_stat, p_kw, level_ratios = run_kruskal_wallis_analysis(results, subjects, cfg)
    chi2_result, contingency_table = run_chi_square_analysis(fatigue_levels, subjects, cfg)

    return {
        "w_dict": w_dict,
        "h_stat": h_stat,
        "p_kw": p_kw,
        "level_ratios": level_ratios,
        "chi2_result": chi2_result,
        "contingency_table": contingency_table,
    }


def run_correlation(df_features) -> dict:
    if df_features is None or df_features.empty:
        return {"df_corr": None, "df_spearman": None}

    exclude = {"subject", "condition", "session", "window", "fatigue_level", "fatigue_label", "z_score", "ratio"}
    feature_cols = [c for c in df_features.columns if c not in exclude]

    df_corr = run_pearson_analysis(df_features, feature_cols)
    df_spearman = run_spearman_analysis(df_features, feature_cols)
    run_anova_analysis(df_features, feature_cols)

    return {"df_corr": df_corr, "df_spearman": df_spearman}
