import matplotlib.pyplot as plt
from pathlib import Path

from .ratio_plots import plot_ratio_analysis
from .stats_plots import plot_kruskal_wallis_boxplot, plot_chi_square_distribution
from .correlation_plots import plot_correlation_heatmap, plot_top_6_features_boxplots


def visualize_all(results, fatigue_levels, subjects, analysis_results: dict, results_dir: Path):
    plot_ratio_analysis(results, fatigue_levels, subjects, results_dir)
    plt.close("all")

    plot_kruskal_wallis_boxplot(analysis_results["level_ratios"], results_dir)
    plot_chi_square_distribution(analysis_results["contingency_table"], analysis_results["chi2_result"], results_dir)
    plt.close("all")

    df_corr = analysis_results.get("df_corr")
    df_features = analysis_results.get("df_features")
    if df_corr is not None and not df_corr.empty and df_features is not None:
        top_features = df_corr.head(15)["feature"].tolist()
        plot_correlation_heatmap(df_features, top_features, results_dir=results_dir)
        plot_top_6_features_boxplots(df_features, df_corr, results_dir=results_dir)
        plt.close("all")

    print("[OK] All visualizations saved.")
