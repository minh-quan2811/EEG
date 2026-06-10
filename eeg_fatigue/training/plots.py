import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


def plot_kfold_curves(fold_histories, model_choice, results_dir: Path):
    if not fold_histories or "train_loss" not in fold_histories[0]:
        return

    epochs = len(fold_histories[0]["train_loss"])
    x = range(1, epochs + 1)

    def mean_std(key):
        mat = np.array([h[key] for h in fold_histories])
        return mat.mean(axis=0), mat.std(axis=0)

    t_loss_m, t_loss_s = mean_std("train_loss")
    v_loss_m, v_loss_s = mean_std("val_loss")
    t_acc_m, t_acc_s = mean_std("train_acc")
    v_acc_m, v_acc_s = mean_std("val_acc")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, t_loss_m, color="#3498db", label="Train Loss")
    axes[0].fill_between(x, t_loss_m - t_loss_s, t_loss_m + t_loss_s, color="#3498db", alpha=0.15)
    axes[0].plot(x, v_loss_m, color="#e74c3c", label="Val Loss")
    axes[0].fill_between(x, v_loss_m - v_loss_s, v_loss_m + v_loss_s, color="#e74c3c", alpha=0.15)
    for h in fold_histories:
        axes[0].plot(x, h["val_loss"], color="#e74c3c", alpha=0.2, linewidth=0.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss — mean ± std across folds")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(x, t_acc_m, color="#3498db", label="Train Acc")
    axes[1].fill_between(x, t_acc_m - t_acc_s, t_acc_m + t_acc_s, color="#3498db", alpha=0.15)
    axes[1].plot(x, v_acc_m, color="#e74c3c", label="Val Acc")
    axes[1].fill_between(x, v_acc_m - v_acc_s, v_acc_m + v_acc_s, color="#e74c3c", alpha=0.15)
    for h in fold_histories:
        axes[1].plot(x, h["val_acc"], color="#e74c3c", alpha=0.2, linewidth=0.8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy — mean ± std across folds")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    n_folds = len(fold_histories)
    plt.suptitle(f"{model_choice.upper()} K-Fold Training Curves ({n_folds} folds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()
    print("[OK] Training curves saved.")


def plot_confusion_matrix(y_test, y_pred, label_names, best_fold_num, results_dir: Path):
    present = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names = [label_names[i] for i in present]
    cm = confusion_matrix(y_test, y_pred, labels=present)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Test Set  (Best model: Fold {best_fold_num})")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("[OK] Confusion matrix saved.")


def plot_per_subject_accuracy(df_subj, best_fold_num, results_dir: Path):
    fig, ax = plt.subplots(figsize=(max(8, len(df_subj) * 0.9), 5))
    colors = ["#2ecc71" if a >= 0.7 else "#f39c12" if a >= 0.5 else "#e74c3c"
              for a in df_subj["accuracy"]]
    bars = ax.bar(df_subj["subject"], df_subj["accuracy"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(df_subj["accuracy"].mean(), color="#2c3e50", linestyle="--",
               linewidth=1.5, label=f"Mean = {df_subj['accuracy'].mean():.3f}")
    for bar, acc in zip(bars, df_subj["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Subject Accuracy — Best Model (Fold {best_fold_num})")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "per_subject_accuracy.png", dpi=150)
    plt.close()
    print("[OK] Per-subject accuracy chart saved.")
