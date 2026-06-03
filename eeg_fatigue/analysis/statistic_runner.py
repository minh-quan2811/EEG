from .statistics import (
    run_wilcoxon_analysis, run_kruskal_wallis_analysis,
    run_chi_square_analysis
)


def run_statistical_analysis(results: dict, fatigue_levels: dict, subjects: list[str], cfg) -> tuple:
    """Run all statistical tests (Wilcoxon, Kruskal-Wallis, Chi-square)."""
    _, w_dict = run_wilcoxon_analysis(results, subjects, cfg)
    h_stat, p_kw, level_ratios = run_kruskal_wallis_analysis(results, subjects, cfg)
    chi2_result, contingency_table = run_chi_square_analysis(fatigue_levels, subjects, cfg)
    return w_dict, h_stat, p_kw, level_ratios, chi2_result, contingency_table
