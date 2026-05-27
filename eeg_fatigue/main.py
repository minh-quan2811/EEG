import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from . import config
from .data.loader import get_valid_subjects
from .features.band_power import compute_band_power_per_window, compute_band_power
from .features.ratios import compute_ratio, compute_ratios
from .labeling.fatigue import compute_all_labels, build_baseline, classify_fatigue_level
from .analysis.statistics import wilcoxon_test, kruskal_wallis_test, dunns_post_hoc, chi_square_test
from .analysis.correlation import pearson_correlation, spearman_correlation, anova_kruskal_wallis
from .visualization.ratio_plots import plot_ratio_analysis
from .visualization.stats_plots import plot_kruskal_wallis_boxplot, plot_chi_square_distribution
from .visualization.correlation_plots import plot_correlation_heatmap, plot_top_6_features_boxplots
from .export.results import build_dataframe, build_feature_dataframe, save_results_csv

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Create results folder
os.makedirs("results", exist_ok=True)

# Tee stdout to file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("results/analysis_output.txt", "w")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

print("="*60)
print("EEG FATIGUE ANALYSIS — Main Pipeline")
print("="*60)

# 1. Load valid subjects
print("\n[1/8] Loading subjects...")
valid_subjects = get_valid_subjects(config.AFTER_PATH, config.BEFORE_PATH)

# 2. Compute per-window ratios
print("\n[2/8] Computing per-window ratios...")
results = {}
for subject in valid_subjects:
    print(f"Processing subject: {subject}")
    results[subject] = {"before": {}, "after": {}}

    for condition, root in [("before", config.BEFORE_PATH), ("after", config.AFTER_PATH)]:
        sub_folder = os.path.join(root, subject)
        for session in range(1, config.NUM_SESSIONS + 1):
            filepath = os.path.join(sub_folder, f"{session}.set")
            if not os.path.isfile(filepath):
                continue

            window_powers = compute_band_power_per_window(
                filepath, config.BANDS_SIMPLE, window_sec=config.WINDOW_SEC, overlap=config.OVERLAP
            )
            if not window_powers:
                continue

            window_ratios = [compute_ratio(pw) for pw in window_powers]
            results[subject][condition][session] = window_ratios

            n_valid = sum(1 for r in window_ratios if not np.isnan(r))
            print(f"  [{condition}] Session {session} → {len(window_ratios)} windows, {n_valid} valid ratios")

print("✓ Per-window ratios computed.")

# 3. Compute fatigue levels
print("\n[3/8] Computing fatigue levels...")
fatigue_levels = compute_all_labels(results, valid_subjects)
print("✓ Fatigue levels computed.")

# 4. Build results DataFrame
print("\n[4/8] Building results DataFrame...")
df = build_dataframe(results, fatigue_levels, valid_subjects, config)
print(f"Total rows (windows): {len(df)}")
print(df.head(10))

# 5. Statistical tests — Wilcoxon
print("\n[5/8] Running statistical tests...")
print("\n" + "="*55)
print("Wilcoxon Signed-Rank Test: Before vs After")
print("="*55)

before_means = []
after_means = []
for subject in valid_subjects:
    b = [r for sess_ratios in results[subject]["before"].values() for r in sess_ratios if not np.isnan(r)]
    a = [r for sess_ratios in results[subject]["after"].values() for r in sess_ratios if not np.isnan(r)]
    if b and a:
        before_means.append(np.mean(b))
        after_means.append(np.mean(a))

before_means = np.array(before_means)
after_means = np.array(after_means)

w_stat, w_p, w_r, w_dict = wilcoxon_test(before_means, after_means)
if w_stat is not None:
    print(f"  N subjects         : {w_dict['N']}")
    print(f"  Before mean ratio  : {w_dict['before_mean']:.4f} ± {w_dict['before_std']:.4f}")
    print(f"  After mean ratio   : {w_dict['after_mean']:.4f} ± {w_dict['after_std']:.4f}")
    print(f"  Wilcoxon W stat    : {w_dict['stat']:.4f}")
    print(f"  p-value            : {w_dict['p_value']:.4f}  {'✓ Significant' if w_dict['p_value'] < 0.05 else '✗ Not significant'}")
    print(f"  Effect size r      : {w_dict['r_effect']:.4f}")

# Kruskal-Wallis
print("\n" + "="*60)
print("Kruskal-Wallis Test: Are 4 Levels Distinct?")
print("="*60)

level_ratios = {0: [], 1: [], 2: [], 3: []}
for subject in valid_subjects:
    baseline_mean, baseline_std = build_baseline(results, subject)
    if baseline_mean is None:
        continue

    for condition in ["before", "after"]:
        for session, window_ratios in results[subject][condition].items():
            for ratio in window_ratios:
                if not np.isnan(ratio):
                    z = (ratio - baseline_mean) / baseline_std
                    level = classify_fatigue_level(z)
                    if not np.isnan(level):
                        level_ratios[int(level)].append(ratio)

for lvl, label in config.LEVEL_LABELS.items():
    vals = level_ratios[lvl]
    if vals:
        print(f"  {label:15s} (Level {lvl}): n={len(vals):3d}, median={np.median(vals):.4f}")

groups = [level_ratios[lvl] for lvl in [0, 1, 2, 3] if len(level_ratios[lvl]) > 0]
if len(groups) >= 2:
    h_stat, p_kw = kruskal_wallis_test(*groups)
    print(f"\n  Kruskal-Wallis H = {h_stat:.4f}, p = {p_kw:.4f}  "
          f"{'✓ Significant' if p_kw < 0.05 else '✗ Not significant'}")

# Chi-Square
print("\n" + "="*60)
print("Chi-Square Test: Do Levels Shift Before vs After?")
print("="*60)

contingency_data = {"before": {0: 0, 1: 0, 2: 0, 3: 0}, "after": {0: 0, 1: 0, 2: 0, 3: 0}}
for subject in valid_subjects:
    for condition in ["before", "after"]:
        for session, window_levels in fatigue_levels[subject][condition].items():
            for level in window_levels:
                if not np.isnan(level):
                    contingency_data[condition][int(level)] += 1

contingency_table = np.array([[contingency_data[c][lvl] for lvl in [0, 1, 2, 3]] for c in ["before", "after"]])
chi2_result = chi_square_test(contingency_table)
chi2, p_chi, dof, cramers_v = chi2_result

print(f"  Chi-square χ² = {chi2:.4f}, p = {p_chi:.4f}  "
      f"{'✓ Significant' if p_chi < 0.05 else '✗ Not significant'}")
print(f"  Cramér's V = {cramers_v:.4f}")

# 6. Visualizations
print("\n[6/8] Generating visualizations...")
fig1 = plot_ratio_analysis(results, fatigue_levels, valid_subjects)
fig2 = plot_kruskal_wallis_boxplot(level_ratios)
fig3 = plot_chi_square_distribution(contingency_table, chi2_result)
print("✓ Main visualizations saved.")

# 7. Correlation analysis
print("\n[7/8] Running correlation analysis...")
exclude_cols = ["subject", "condition", "session", "fatigue_level", "fatigue_label", "z_score", "window"]

print("\n" + "="*60)
print("Pearson Correlation Analysis")
print("="*60)
df_corr = pearson_correlation(df, [c for c in df.columns if c not in exclude_cols])
print("\nTop 10 Features Most Correlated with Fatigue:")
print(df_corr.head(10).to_string(index=False))

print("\n" + "="*60)
print("Spearman Correlation Analysis")
print("="*60)
df_spearman = spearman_correlation(df, [c for c in df.columns if c not in exclude_cols])
print("\nTop 10 Features (Spearman):")
print(df_spearman.head(10).to_string(index=False))

# Heatmap and boxplots
top_features = df_corr.head(15)["feature"].tolist()
plot_correlation_heatmap(df, top_features)
plot_top_6_features_boxplots(df, df_corr)
print("✓ Correlation visualizations saved.")

# 8. Export results
print("\n[8/8] Exporting results...")
save_results_csv(df)

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)
print("\nOutput files saved to results/:")
print("  - eeg_fatigue_analysis.png")
print("  - method1_kruskal_boxplot.png")
print("  - method3_chisquare_distribution.png")
print("  - correlation_heatmap.png")
print("  - top_6_features_boxplots.png")
print("  - eeg_fatigue_results.csv")
