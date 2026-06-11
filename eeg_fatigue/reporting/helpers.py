import numpy as np
from ..analysis.statistics import effect_size_interpretation


def config_lines(cfg, train_cfg=None):
    lines = [
        "=" * 60,
        "EEG FATIGUE -- CONFIG",
        "=" * 60,
        "",
        f"BASE_PATH    : {cfg.BASE_PATH}",
        f"AFTER_PATH   : {cfg.AFTER_PATH}",
        f"BEFORE_PATH  : {cfg.BEFORE_PATH}",
        "",
        f"SUBJECTS ({len(cfg.SUBJECTS)}):",
    ]
    for s in cfg.SUBJECTS:
        lines.append(f"  {s}")
    lines += [
        "",
        f"NUM_SESSIONS : {cfg.NUM_SESSIONS}",
        f"WINDOW_SEC   : {cfg.WINDOW_SEC}",
        f"OVERLAP      : {cfg.OVERLAP}",
        "",
        "BANDS:",
    ]
    for band, (lo, hi) in cfg.BANDS.items():
        lines.append(f"  {band:6s}: {lo} - {hi} Hz")
    lines += [
        "",
        f"AGG_MODE     : {cfg.AGG_MODE}",
        f"Z_THRESHOLDS : {cfg.Z_THRESHOLDS}",
        f"LEVEL_LABELS : {cfg.LEVEL_LABELS}",
        "",
    ]

    if train_cfg is not None:
        lines += [
            "=" * 60,
            "TRAINING CONFIG",
            "=" * 60,
            "",
            f"MODEL_CHOICE       : {train_cfg.MODEL_CHOICE}",
            f"N_FOLDS            : {train_cfg.N_FOLDS}",
            f"MIN_CORRELATION    : {train_cfg.MIN_CORRELATION}",
            f"SIGNIFICANCE_ALPHA : {train_cfg.SIGNIFICANCE_ALPHA}",
            "",
            "MODEL_CONFIG:",
        ]
        for k, v in train_cfg.MODEL_CONFIG.items():
            lines.append(f"  {k:20s}: {v}")
        lines.append("")

    return lines


def wilcoxon_lines(w_dict):
    lines = ["", "-- Wilcoxon Signed-Rank (Before vs After) --"]
    if w_dict and "p_value" in w_dict:
        sig = "[SIGNIFICANT] p<0.05" if w_dict["p_value"] < 0.05 else "[NOT SIGNIFICANT]"
        eff = effect_size_interpretation(w_dict["r_effect"])
        lines.append(f"  N={w_dict['N']},  W={w_dict['stat']:.4f},  p={w_dict['p_value']:.4f}  {sig}")
        lines.append(f"  Before : {w_dict['before_mean']:.4f} +/- {w_dict['before_std']:.4f}")
        lines.append(f"  After  : {w_dict['after_mean']:.4f} +/- {w_dict['after_std']:.4f}")
        lines.append(f"  Effect size r = {w_dict['r_effect']:.4f}  ({eff})")
    else:
        lines.append("  Not enough subjects for Wilcoxon test.")
    return lines


def kruskal_lines(level_ratios, h_stat, p_kw):
    lines = ["", "-- Kruskal-Wallis --"]
    if level_ratios:
        for lvl, vals in level_ratios.items():
            if vals:
                iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
                lines.append(f"  Level {lvl}: n={len(vals):4d}, median={np.median(vals):.4f}, IQR={iqr:.4f}")
    if h_stat is not None:
        sig = "[SIGNIFICANT] p<0.05" if p_kw < 0.05 else "[NOT SIGNIFICANT]"
        lines.append(f"  H={h_stat:.4f},  p={p_kw:.4f}  {sig}")
    return lines


def chi2_lines(chi2_result, contingency_table):
    lines = ["", "-- Chi-Square (Before vs After Level Shift) --"]
    if chi2_result:
        chi2, p_chi, dof, cramers_v = chi2_result
        sig = "[SIGNIFICANT] p<0.05" if p_chi < 0.05 else "[NOT SIGNIFICANT]"
        strength = effect_size_interpretation(cramers_v)
        lines.append(f"  chi2={chi2:.4f},  df={dof},  p={p_chi:.4f}  {sig}")
        lines.append(f"  Cramer's V = {cramers_v:.4f}  ({strength} effect)")
    return lines


def correlation_lines(df_corr, df_spearman):
    lines = []
    if df_corr is not None and not df_corr.empty:
        lines += ["", "-- Pearson Correlation Top 10 --"]
        lines.append(df_corr.head(10).to_string(index=False))
    if df_spearman is not None and not df_spearman.empty:
        lines += ["", "-- Spearman Correlation Top 10 --"]
        lines.append(df_spearman.head(10).to_string(index=False))
    return lines