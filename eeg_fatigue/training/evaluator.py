import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from .dataset import FatigueDataset
from .models import ResNetFatigue, MLPFatigue


def _load_model(model_choice, model_config, models_dir, device):
    suffix = ".pkl" if model_choice in ("svm", "rf") else ".pt"
    model_path = models_dir / f"best_model{suffix}"

    if model_choice in ("svm", "rf"):
        import joblib
        return joblib.load(model_path)
    elif model_choice == "mlp":
        model = MLPFatigue(model_config["input_size"], model_config["num_classes"],
                           model_config["hidden_dim"], model_config["dropout"]).to(device)
    else:
        model = ResNetFatigue(model_config["input_size"], model_config["num_classes"],
                              model_config["hidden_dim"], model_config["num_blocks"],
                              model_config["dropout"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _predict(model, model_choice, X, y, batch_size, device):
    if model_choice in ("svm", "rf"):
        return np.array(model.predict(X))
    loader = DataLoader(FatigueDataset(X, y), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            out = model(X_batch.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.array(preds)


def evaluate_model(model_choice, model_config, X_test, y_test, level_labels,
                   best_fold_num, fold_val_losses, models_dir: Path, results_dir: Path, device):
    model = _load_model(model_choice, model_config, models_dir, device)
    preds = _predict(model, model_choice, X_test, y_test, model_config["batch_size"], device)

    best_fold_idx = best_fold_num - 1
    present = sorted(np.unique(np.concatenate([y_test, preds])))
    label_names = [level_labels[i] for i in present]

    print(f"\n  Best model: Fold {best_fold_num}  (val loss = {fold_val_losses[best_fold_idx]:.4f})")
    print(f"  Test windows: {len(y_test)}\n")

    report = classification_report(y_test, preds, target_names=label_names)
    print(report)
    (results_dir / "classification_report.txt").write_text(
        f"Best model: Fold {best_fold_num}\n\n" + report
    )
    return preds


def evaluate_per_subject(model_choice, model_config, df, selected_features, scaler,
                         level_labels, best_fold_num, models_dir: Path, results_dir: Path, device):
    model = _load_model(model_choice, model_config, models_dir, device)
    subjects = sorted(df["subject"].unique())
    rows = []

    print("\n" + "=" * 60)
    print(f"PER-SUBJECT ACCURACY  (Best model: Fold {best_fold_num})")
    print("=" * 60)

    for subject in subjects:
        df_sub = df[df["subject"] == subject]
        if df_sub.empty:
            continue
        X_sub = scaler.transform(df_sub[selected_features].values.astype(np.float32))
        y_sub = df_sub["fatigue_level"].values.astype(np.int64)

        preds = _predict(model, model_choice, X_sub, y_sub, model_config["batch_size"], device)
        acc = (preds == y_sub).mean()

        class_accs = {}
        for lvl, lbl in level_labels.items():
            mask = y_sub == lvl
            class_accs[lbl] = (preds[mask] == y_sub[mask]).mean() if mask.sum() > 0 else float("nan")

        rows.append({"subject": subject, "accuracy": acc, "n_windows": len(y_sub), **class_accs})
        class_str = "  ".join(
            f"{lbl}: {v:.2f}" if not np.isnan(v) else f"{lbl}: N/A"
            for lbl, v in class_accs.items()
        )
        print(f"  {subject}  |  overall={acc:.4f}  n={len(y_sub):4d}  |  {class_str}")

    import pandas as pd
    df_subj = pd.DataFrame(rows)
    print(f"\n  Mean: {df_subj['accuracy'].mean():.4f}  Std: {df_subj['accuracy'].std():.4f}")
    df_subj.to_csv(results_dir / "per_subject_accuracy.csv", index=False)
    print("[OK] Per-subject accuracy saved.")
    return df_subj
