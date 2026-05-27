import io
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .data.loader import get_valid_subjects
from .labeling.fatigue import compute_all_labels
from .visualization.ratio_plots import plot_ratio_analysis
from .visualization.stats_plots import plot_kruskal_wallis_boxplot, plot_chi_square_distribution
from .export.results import build_dataframe, build_feature_dataframe, save_results_csv
from .features.band_power import compute_band_power_per_window
from .features.ratios import compute_ratios
from .analysis.pipeline import load_and_compute_ratios
from .analysis.statistics import (
    run_wilcoxon_analysis, run_kruskal_wallis_analysis,
    run_chi_square_analysis, effect_size_interpretation
)
from .analysis.correlation import (
    run_pearson_analysis, run_spearman_analysis,
    run_anova_analysis, generate_correlation_visualizations
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class _Logger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def run():
    os.makedirs("results", exist_ok=True)
    log_file = open("results/analysis_output.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    try:
        safe_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        safe_stdout = original_stdout
    sys.stdout = _Logger(safe_stdout, log_file)
    try:
        _execute_pipeline()
    except Exception as e:
        import traceback
        print("\n[ERROR] Pipeline failed:")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print("Console output saved -> results/analysis_output.txt")


def _execute_pipeline():
    print("=" * 60)
    print("EEG FATIGUE ANALYSIS  -  Main Pipeline")
    print("=" * 60)

    print("\n[1/8] Loading subjects...")
    valid_subjects = get_valid_subjects(config.AFTER_PATH, config.BEFORE_PATH)
    if not valid_subjects:
        print("[ABORT] No valid subjects found. Check AFTER_PATH / BEFORE_PATH in config.py.")
        return

    results = load_and_compute_ratios(valid_subjects)

    print("\n[3/8] Computing fatigue levels...")
    fatigue_levels = compute_all_labels(results, valid_subjects)
    print("[OK] Fatigue levels computed.")

    print("\n[4/8] Building summary DataFrame...")
    df_summary = build_dataframe(results, fatigue_levels, valid_subjects, config)
    if df_summary.empty:
        print("[WARN] Summary DataFrame is empty — check .set files loaded correctly.")
        return
    print(f"  Total rows (windows): {len(df_summary)}")
    print(df_summary.head(10).to_string(index=False))

    print("\n[5/8] Running statistical tests...")
    w_stat, w_dict = run_wilcoxon_analysis(results, valid_subjects)
    h_stat, p_kw, level_ratios = run_kruskal_wallis_analysis(results, valid_subjects)
    chi2_result, contingency_table = run_chi_square_analysis(fatigue_levels, valid_subjects)

    print("\n[6/8] Generating visualizations...")
    plot_ratio_analysis(results, fatigue_levels, valid_subjects)
    plot_kruskal_wallis_boxplot(level_ratios)
    plot_chi_square_distribution(contingency_table, chi2_result)
    plt.close("all")
    print("[OK] Main visualizations saved.")

    print("\n[7/8] Running full-feature correlation analysis...")
    df_features = build_feature_dataframe(
        valid_subjects=valid_subjects,
        fatigue_levels=fatigue_levels,
        config=config,
        band_power_func=compute_band_power_per_window,
        ratios_func=compute_ratios,
    )

    if df_features.empty:
        print("  [WARN] Feature DataFrame is empty — skipping correlation analysis.")
        df_corr = None
        df_spearman = None
    else:
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

    print("\n[8/8] Exporting results...")
    _print_statistical_summary(w_dict, level_ratios, h_stat, p_kw, chi2_result, contingency_table, df_corr, df_spearman)
    save_results_csv(df_features if df_features is not None and not df_features.empty else df_summary)
    _print_completion_info()


def _print_statistical_summary(w_dict, level_ratios, h_stat, p_kw, chi2_result, contingency_table, df_corr, df_spearman):
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    print("\n-- Wilcoxon Signed-Rank (Before vs After) --")
    if w_dict is not None:
        sig = "[SIG] p<0.05" if w_dict["p_value"] < 0.05 else "[NS]"
        eff = effect_size_interpretation(w_dict["r_effect"])
        print(f"  N={w_dict['N']},  W={w_dict['stat']:.4f},  p={w_dict['p_value']:.4f}  {sig}")
        print(f"  Before : {w_dict['before_mean']:.4f} +/- {w_dict['before_std']:.4f}")
        print(f"  After  : {w_dict['after_mean']:.4f} +/- {w_dict['after_std']:.4f}")
        print(f"  Effect size r = {w_dict['r_effect']:.4f}  ({eff})")
    else:
        print("  Not enough subjects for Wilcoxon test.")
    print(f"\n-- Kruskal-Wallis ({len(config.LEVEL_LABELS)} Level Distinction) --")
    for lvl, label in config.LEVEL_LABELS.items():
        vals = level_ratios[lvl]
        if vals:
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            print(f"  {label:15s}: n={len(vals):4d}, "
                  f"median={np.median(vals):.4f}, IQR={iqr:.4f}")
    if h_stat is not None:
        sig = "[SIG] p<0.05" if p_kw < 0.05 else "[NS]"
        print(f"  H={h_stat:.4f},  p={p_kw:.4f}  {sig}")
    print("\n-- Chi-Square (Before vs After Level Shift) --")
    chi2, p_chi, dof, cramers_v = chi2_result
    sig = "[SIG] p<0.05" if p_chi < 0.05 else "[NS]"
    strength = effect_size_interpretation(cramers_v)
    print(f"  chi2={chi2:.4f},  df={dof},  p={p_chi:.4f}  {sig}")
    print(f"  Cramer's V = {cramers_v:.4f}  ({strength} effect)")
    print("\n  Contingency table:")
    for i, cond in enumerate(["Before", "After "]):
        row = contingency_table[i]
        total = row.sum()
        if total > 0:
            pcts = " | ".join([f"{v:3d} ({100*v/total:4.1f}%)" for v in row])
        else:
            pcts = "no data"
        print(f"    {cond}: {pcts}   total={total}")
    if df_corr is not None and not df_corr.empty:
        print("\n-- Pearson Correlation Top 10 --")
        print(df_corr.head(10).to_string(index=False))
        print("\n-- Spearman Correlation Top 10 --")
        print(df_spearman.head(10).to_string(index=False))


def _print_completion_info():
    print("\n" + "=" * 60)
    print("[OK] ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput files in results/:")
    print("  eeg_fatigue_analysis.png")
    print("  method1_kruskal_boxplot.png")
    print("  method3_chisquare_distribution.png")
    print("  correlation_heatmap.png")
    print("  top_6_features_boxplots.png")
    print("  eeg_fatigue_results.csv")
    print("  analysis_output.txt")


if __name__ == "__main__":
    run()