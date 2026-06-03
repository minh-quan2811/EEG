import numpy as np
from .statistics import effect_size_interpretation


def print_statistical_summary(w_dict, level_ratios, h_stat, p_kw, chi2_result,
                             contingency_table, df_corr, df_spearman, cfg):
    """Print comprehensive statistical summary of analysis results."""
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    _print_wilcoxon_summary(w_dict)
    _print_kruskal_wallis_summary(level_ratios, h_stat, p_kw, cfg)
    _print_chi_square_summary(chi2_result, contingency_table)
    _print_correlation_summary(df_corr, df_spearman)


def _print_wilcoxon_summary(w_dict):
    """Print Wilcoxon Signed-Rank test results."""
    print("\n-- Wilcoxon Signed-Rank (Before vs After) --")
    if w_dict is not None and "p_value" in w_dict:
        sig = "[SIGNIFICANT] p<0.05" if w_dict["p_value"] < 0.05 else "[NOT SIGNIFICANT]"
        eff = effect_size_interpretation(w_dict["r_effect"])
        print(f"  N={w_dict['N']},  W={w_dict['stat']:.4f},  p={w_dict['p_value']:.4f}  {sig}")
        print(f"  Before : {w_dict['before_mean']:.4f} +/- {w_dict['before_std']:.4f}")
        print(f"  After  : {w_dict['after_mean']:.4f} +/- {w_dict['after_std']:.4f}")
        print(f"  Effect size r = {w_dict['r_effect']:.4f}  ({eff})")
    else:
        print("  Not enough subjects for Wilcoxon test.")


def _print_kruskal_wallis_summary(level_ratios, h_stat, p_kw, cfg):
    """Print Kruskal-Wallis test results."""
    print(f"\n-- Kruskal-Wallis ({len(cfg.LEVEL_LABELS)} Level Distinction) --")
    for lvl, label in cfg.LEVEL_LABELS.items():
        vals = level_ratios[lvl]
        if vals:
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            print(f"  {label:15s}: n={len(vals):4d}, "
                  f"median={np.median(vals):.4f}, IQR={iqr:.4f}")
    if h_stat is not None:
        sig = "[SIGNIFICANT] p<0.05" if p_kw < 0.05 else "[NOT SIGNIFICANT]"
        print(f"  H={h_stat:.4f},  p={p_kw:.4f}  {sig}")


def _print_chi_square_summary(chi2_result, contingency_table):
    """Print Chi-Square test results."""
    print("\n-- Chi-Square (Before vs After Level Shift) --")
    chi2, p_chi, dof, cramers_v = chi2_result
    sig = "[SIGNIFICANT] p<0.05" if p_chi < 0.05 else "[NOT SIGNIFICANT]"
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


def _print_correlation_summary(df_corr, df_spearman):
    """Print correlation analysis results."""
    if df_corr is not None and not df_corr.empty:
        print("\n-- Pearson Correlation Top 10 --")
        print(df_corr.head(10).to_string(index=False))
        print("\n-- Spearman Correlation Top 10 --")
        print(df_spearman.head(10).to_string(index=False))
