import numpy as np
from pathlib import Path

from .helpers import config_lines, correlation_lines


def save_training_summary(train_results: dict, results_dir: Path, cfg, train_cfg):
    fold_histories    = train_results["fold_histories"]
    fold_val_losses   = train_results["fold_val_losses"]
    best_fold_num     = train_results["best_fold_num"]
    df_per_subject    = train_results["df_per_subject"]
    selected_features = train_results["selected_features"]
    device            = train_results["device"]
    model_choice      = train_results["model_choice"]
    X_all             = train_results["X_all"]
    y_all             = train_results["y_all"]
    X_train_val       = train_results["X_train_val"]
    X_test            = train_results["X_test"]
    y_train_val       = train_results["y_train_val"]
    y_test            = train_results["y_test"]
    n_folds           = train_results["n_folds"]
    level_labels      = train_results["level_labels"]
    stat_validation   = train_results.get("stat_validation", [])
    corr_lines        = train_results.get("correlation_lines", [])

    best_fold_idx   = best_fold_num - 1
    best_val_loss   = fold_val_losses[best_fold_idx]
    best_val_acc    = max(fold_histories[best_fold_idx]["val_acc"])
    final_train_acc = fold_histories[best_fold_idx]["train_acc"][-1]
    vl_arr          = np.array(fold_val_losses)

    total_samples         = X_all.shape[0]
    unique_labels, counts = np.unique(y_all, return_counts=True)
    train_val_pct         = (X_train_val.shape[0] / total_samples) * 100
    test_pct              = (X_test.shape[0] / total_samples) * 100
    val_pct_kfold         = (1 / n_folds) * 100
    train_pct_kfold       = ((n_folds - 1) / n_folds) * 100
    approx_train          = int(X_train_val.shape[0] * train_pct_kfold / 100)
    approx_val            = int(X_train_val.shape[0] * val_pct_kfold / 100)
    suffix                = ".pkl" if model_choice in ("svm", "rf") else ".pt"

    lines = config_lines(cfg, train_cfg)

    # statistical validation
    lines += [
        "=" * 60,
        "STATISTICAL VALIDATION",
        "=" * 60,
        "",
    ]
    lines += stat_validation if stat_validation else ["  (validation results not captured)"]
    lines.append("")

    # correlation
    lines += [
        "=" * 60,
        "FEATURE CORRELATION",
        "=" * 60,
        "",
    ]
    lines += corr_lines if corr_lines else ["  (correlation results not captured)"]
    lines.append("")

    # dataset overview
    lines += [
        "=" * 60,
        "DATASET OVERVIEW",
        "=" * 60,
        "",
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
        "-" * 36,
        f"Train + Validation: {X_train_val.shape[0]:,} samples ({train_val_pct:.0f}%)",
        f"Test Set          : {X_test.shape[0]:,} samples ({test_pct:.0f}%)",
        "",
        "CROSS-VALIDATION",
        "-" * 36,
        f"Method: {n_folds}-Fold Cross-Validation",
        f"  Training   : {train_pct_kfold:.0f}% (~{approx_train:,} samples)",
        f"  Validation : {val_pct_kfold:.0f}% (~{approx_val:,} samples)",
    ]

    # training results
    lines += [
        "",
        "=" * 60,
        "TRAINING RESULTS",
        "=" * 60,
        "",
        f"Model             : {model_choice.upper()}",
        f"Device            : {device}",
        f"K-Fold splits     : {n_folds}",
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
    ]

    # classification report -- read then delete
    report_path = results_dir / "classification_report.txt"
    if report_path.exists():
        lines += ["", "Classification Report (Test Set):", ""]
        lines.append(report_path.read_text(encoding="utf-8"))
        report_path.unlink()

    # per-subject accuracy
    lines += [
        "",
        "Per-subject accuracy (best model):",
        "-" * 36,
    ]
    for _, row in df_per_subject.iterrows():
        lines.append(f"  {row['subject']}  acc={row['accuracy']:.4f}  n={int(row['n_windows'])}")
    lines += [
        f"  Mean: {df_per_subject['accuracy'].mean():.4f}",
        f"  Std : {df_per_subject['accuracy'].std():.4f}",
        "",
        f"Selected features : {selected_features}",
        "",
        f"Models saved to   : {results_dir / 'models'}",
        f"  best_model{suffix}",
        "=" * 60,
    ]

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "training_summary.txt").write_text(summary, encoding="utf-8")

    for name in ("per_subject_accuracy.csv", "stat_validation.txt", "config.txt"):
        p = results_dir / name
        if p.exists():
            p.unlink()