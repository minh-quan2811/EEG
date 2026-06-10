import numpy as np
import torch
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_builder import extract_features_and_labels
from .validator import run_statistical_validation
from .trainer import run_kfold
from .evaluator import evaluate_model, evaluate_per_subject
from .plots import plot_kfold_curves, plot_confusion_matrix, plot_per_subject_accuracy


def build_feature_dataframe(subjects, fatigue_levels, cfg):
    return extract_features_and_labels(subjects, cfg)


def validate(df, train_cfg, results_dir: Path) -> bool:
    return run_statistical_validation(df, train_cfg, results_dir)


def _select_features(df, min_corr):
    exclude = {"subject", "condition", "session", "window", "fatigue_level"}
    feature_cols = [c for c in df.columns if c not in exclude]

    correlations = []
    for col in feature_cols:
        x = df[col].apply(lambda v: float(v) if str(v).replace('.', '', 1).lstrip('-').isdigit() else np.nan)
        y = df["fatigue_level"]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        correlations.append({"feature": col, "pearson_r": r, "abs_r": abs(r), "p_value": p})

    import pandas as pd
    df_corr = pd.DataFrame(correlations).sort_values("abs_r", ascending=False)

    best_r = df_corr["abs_r"].max() if not df_corr.empty else 0.0
    if best_r < min_corr:
        print(f"\n[EARLY STOP] Best |r| = {best_r:.4f} < {min_corr}. No feature is predictive enough.")
        return None, df_corr

    selected_df = df_corr[df_corr["abs_r"] > 0.4]
    if selected_df.empty:
        selected_df = df_corr.head(5)
        print(f"\n[WARN] No Moderate/Strong features — falling back to top {len(selected_df)}.")
    else:
        print(f"\n[OK] Selected {len(selected_df)} features:")
        for _, row in selected_df.iterrows():
            print(f"  {row['feature']:30s}  r={row['pearson_r']:+.4f}")

    return selected_df["feature"].tolist(), df_corr


def run_training(df, cfg, train_cfg, results_dir: Path) -> dict:
    train_cfg.MODEL_CONFIG["num_classes"] = len(cfg.LEVEL_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  Model: {train_cfg.MODEL_CHOICE}")

    selected_features, _ = _select_features(df, train_cfg.MIN_CORRELATION)
    if selected_features is None:
        return {}

    train_cfg.MODEL_CONFIG["input_size"] = len(selected_features)
    print(f"  Input size: {len(selected_features)} features")

    X_all = df[selected_features].values.astype(np.float32)
    y_all = df["fatigue_level"].values.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.10, random_state=42, stratify=y_all
    )

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fold_histories, fold_val_losses, best_fold_num = run_kfold(
        X_train_val, y_train_val,
        train_cfg.MODEL_CHOICE, train_cfg.MODEL_CONFIG,
        train_cfg.N_FOLDS, models_dir, device
    )

    plot_kfold_curves(fold_histories, train_cfg.MODEL_CHOICE, results_dir)

    label_names = {i: cfg.LEVEL_LABELS[i] for i in sorted(cfg.LEVEL_LABELS)}
    y_pred = evaluate_model(
        train_cfg.MODEL_CHOICE, train_cfg.MODEL_CONFIG,
        X_test, y_test, label_names,
        best_fold_num, fold_val_losses, models_dir, results_dir, device
    )
    plot_confusion_matrix(y_test, y_pred, label_names, best_fold_num, results_dir)

    df_per_subject = evaluate_per_subject(
        train_cfg.MODEL_CHOICE, train_cfg.MODEL_CONFIG,
        df, selected_features, scaler,
        label_names, best_fold_num, models_dir, results_dir, device
    )
    plot_per_subject_accuracy(df_per_subject, best_fold_num, results_dir)

    return {
        "fold_histories": fold_histories,
        "fold_val_losses": fold_val_losses,
        "best_fold_num": best_fold_num,
        "df_per_subject": df_per_subject,
        "selected_features": selected_features,
        "device": str(device),
        "model_choice": train_cfg.MODEL_CHOICE,
        "X_all": X_all,
        "y_all": y_all,
        "X_train_val": X_train_val,
        "X_test": X_test,
        "y_train_val": y_train_val,
        "y_test": y_test,
        "n_folds": train_cfg.N_FOLDS,
        "level_labels": cfg.LEVEL_LABELS,
    }
