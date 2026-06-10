import numpy as np
from pathlib import Path
from ..analysis.statistics import effect_size_interpretation


def save_analysis_summary(analysis_results: dict, results_dir: Path):
    lines = [
        "=" * 60,
        "STATISTICAL SUMMARY",
        "=" * 60,
    ]
    lines += _wilcoxon_lines(analysis_results.get("w_dict"))
    lines += _kruskal_lines(analysis_results.get("level_ratios"), analysis_results.get("h_stat"), analysis_results.get("p_kw"))
    lines += _chi2_lines(analysis_results.get("chi2_result"), analysis_results.get("contingency_table"))
    lines += _correlation_lines(analysis_results.get("df_corr"), analysis_results.get("df_spearman"))

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "analysis_summary.txt").write_text(summary)


def save_training_summary(train_results: dict, results_dir: Path):
    fold_histories = train_results["fold_histories"]
    fold_val_losses = train_results["fold_val_losses"]
    best_fold_num = train_results["best_fold_num"]
    df_per_subject = train_results["df_per_subject"]
    selected_features = train_results["selected_features"]
    device = train_results["device"]
    model_choice = train_results["model_choice"]
    X_all = train_results["X_all"]
    y_all = train_results["y_all"]
    X_train_val = train_results["X_train_val"]
    X_test = train_results["X_test"]
    y_train_val = train_results["y_train_val"]
    y_test = train_results["y_test"]
    n_folds = train_results["n_folds"]
    level_labels = train_results["level_labels"]

    best_fold_idx = best_fold_num - 1
    best_val_loss = fold_val_losses[best_fold_idx]
    best_val_acc = max(fold_histories[best_fold_idx]["val_acc"])
    final_train_acc = fold_histories[best_fold_idx]["train_acc"][-1]
    vl_arr = np.array(fold_val_losses)

    total_samples = X_all.shape[0]
    unique_labels, counts = np.unique(y_all, return_counts=True)
    train_val_pct = (X_train_val.shape[0] / total_samples) * 100
    test_pct = (X_test.shape[0] / total_samples) * 100
    val_pct_kfold = (1 / n_folds) * 100
    train_pct_kfold = ((n_folds - 1) / n_folds) * 100
    approx_train = int(X_train_val.shape[0] * train_pct_kfold / 100)
    approx_val = int(X_train_val.shape[0] * val_pct_kfold / 100)

    suffix = ".pkl" if model_choice in ("svm", "rf") else ".pt"

    lines = [
        "DATASET OVERVIEW",
        "─" * 36,
        f"Total Samples: {total_samples:,}",
        "",
        "Class Distribution:",
    ]
    for label, count in zip(unique_labels, counts):
        pct = (count / total_samples) * 100
        lines.append(f"  {level_labels[label]:<12s}: {count:>5d} ({pct:>5.1f}%)")
    lines += [
        "",
        "DATA SPLIT",
        "─" * 36,
        f"Train + Validation: {X_train_val.shape[0]:,} samples ({train_val_pct:.0f}%)",
        f"Test Set          : {X_test.shape[0]:,} samples ({test_pct:.0f}%)",
        "",
        "CROSS-VALIDATION",
        "─" * 36,
        f"Method: {n_folds}-Fold Cross-Validation",
        f"  Training   : {train_pct_kfold:.0f}% (~{approx_train:,} samples)",
        f"  Validation : {val_pct_kfold:.0f}% (~{approx_val:,} samples)",
        "",
        "=" * 50,
        "TRAINING SUMMARY",
        "=" * 50,
        "",
        f"K-Fold splits: {n_folds}",
        "",
        "Val loss per fold:",
    ]
    for i, vl in enumerate(fold_val_losses):
        marker = "  <-- BEST" if i == best_fold_idx else ""
        lines.append(f"  Fold {i+1}: {vl:.4f}{marker}")
    lines += [
        f"  Mean: {vl_arr.mean():.4f}  Std: {vl_arr.std():.4f}",
        "",
        f"Best fold         : {best_fold_num}",
        f"Best val loss     : {best_val_loss:.4f}",
        f"Best val accuracy : {best_val_acc:.4f}",
        f"Final train acc   : {final_train_acc:.4f}",
        "",
    ]

    report_path = results_dir / "classification_report.txt"
    if report_path.exists():
        lines.append("Classification Report (Test Set):\n")
        lines.append(report_path.read_text())
        lines.append("")

    lines.append("Per-subject accuracy (best model):")
    for _, row in df_per_subject.iterrows():
        lines.append(f"  {row['subject']}  acc={row['accuracy']:.4f}  n={int(row['n_windows'])}")
    lines += [
        f"  Mean: {df_per_subject['accuracy'].mean():.4f}",
        f"  Std : {df_per_subject['accuracy'].std():.4f}",
        "",
        f"Selected features : {selected_features}",
        f"Device used       : {device}",
        "",
        f"Models saved to   : {results_dir / 'models'}",
        f"  best_model{suffix}",
        "=" * 50,
    ]

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "training_summary.txt").write_text(summary)


def _wilcoxon_lines(w_dict):
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


def _kruskal_lines(level_ratios, h_stat, p_kw):
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


def _chi2_lines(chi2_result, contingency_table):
    lines = ["", "-- Chi-Square (Before vs After Level Shift) --"]
    if chi2_result:
        chi2, p_chi, dof, cramers_v = chi2_result
        sig = "[SIGNIFICANT] p<0.05" if p_chi < 0.05 else "[NOT SIGNIFICANT]"
        strength = effect_size_interpretation(cramers_v)
        lines.append(f"  chi2={chi2:.4f},  df={dof},  p={p_chi:.4f}  {sig}")
        lines.append(f"  Cramer's V = {cramers_v:.4f}  ({strength} effect)")
    return lines


def _correlation_lines(df_corr, df_spearman):
    lines = []
    if df_corr is not None and not df_corr.empty:
        lines += ["", "-- Pearson Correlation Top 10 --"]
        lines.append(df_corr.head(10).to_string(index=False))
    if df_spearman is not None and not df_spearman.empty:
        lines += ["", "-- Spearman Correlation Top 10 --"]
        lines.append(df_spearman.head(10).to_string(index=False))
    return lines
