import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score
from pathlib import Path

from .dataset import FatigueDataset
from .models import ResNetFatigue, MLPFatigue, build_svm_model, build_rf_model


def _make_resnet(model_config):
    return ResNetFatigue(
        input_size=model_config["input_size"],
        num_classes=model_config["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        num_blocks=model_config["num_blocks"],
        dropout=model_config["dropout"],
    )


def _make_mlp(model_config):
    return MLPFatigue(
        input_size=model_config["input_size"],
        num_classes=model_config["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
    )


def run_sklearn_fold(build_fn, X_tv, y_tv, train_idx, val_idx, fold, models_dir: Path):
    import joblib
    X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
    X_vl, y_vl = X_tv[val_idx], y_tv[val_idx]

    model = build_fn()
    model.fit(X_tr, y_tr)

    val_acc = accuracy_score(y_vl, model.predict(X_vl))
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_loss = 1.0 - val_acc

    fold_path = models_dir / f"fold_{fold}_model.pkl"
    joblib.dump(model, fold_path)

    history = {
        "train_loss": [1.0 - train_acc],
        "val_loss": [val_loss],
        "train_acc": [train_acc],
        "val_acc": [val_acc],
    }
    print(f"    train acc={train_acc:.4f} | val acc={val_acc:.4f}")
    return history, val_loss, fold_path


def run_torch_fold(build_fn, X_tv, y_tv, train_idx, val_idx, fold, model_config, models_dir: Path, device):
    X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
    X_vl, y_vl = X_tv[val_idx], y_tv[val_idx]

    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights))

    bs = model_config["batch_size"]
    train_loader = DataLoader(FatigueDataset(X_tr, y_tr), batch_size=bs, sampler=sampler)
    val_loader = DataLoader(FatigueDataset(X_vl, y_vl), batch_size=bs, shuffle=False)

    model = build_fn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config["learning_rate"],
                           weight_decay=model_config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    fold_path = models_dir / f"fold_{fold}_model.pt"

    for epoch in range(1, model_config["epochs"] + 1):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_batch)
            t_correct += (out.argmax(1) == y_batch).sum().item()
            t_total += len(y_batch)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                v_loss += loss.item() * len(y_batch)
                v_correct += (out.argmax(1) == y_batch).sum().item()
                v_total += len(y_batch)

        t_loss /= t_total; t_acc = t_correct / t_total
        v_loss /= v_total; v_acc = v_correct / v_total
        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), fold_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{model_config['epochs']} | "
                  f"train loss={t_loss:.4f} acc={t_acc:.4f} | "
                  f"val loss={v_loss:.4f} acc={v_acc:.4f}")

    return history, best_val_loss, fold_path


def run_kfold(X_train_val, y_train_val, model_choice, model_config, n_folds, models_dir: Path, device):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_histories, fold_val_losses, fold_model_paths = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val), start=1):
        print(f"\n{'='*55}")
        print(f"  FOLD {fold} / {n_folds}  —  train: {len(train_idx)}  val: {len(val_idx)}")
        print(f"{'='*55}")

        if model_choice == "svm":
            history, val_loss, fold_path = run_sklearn_fold(
                build_svm_model, X_train_val, y_train_val, train_idx, val_idx, fold, models_dir)
        elif model_choice == "rf":
            history, val_loss, fold_path = run_sklearn_fold(
                build_rf_model, X_train_val, y_train_val, train_idx, val_idx, fold, models_dir)
        elif model_choice == "mlp":
            history, val_loss, fold_path = run_torch_fold(
                lambda: _make_mlp(model_config), X_train_val, y_train_val,
                train_idx, val_idx, fold, model_config, models_dir, device)
        else:
            history, val_loss, fold_path = run_torch_fold(
                lambda: _make_resnet(model_config), X_train_val, y_train_val,
                train_idx, val_idx, fold, model_config, models_dir, device)

        fold_histories.append(history)
        fold_val_losses.append(val_loss)
        fold_model_paths.append(fold_path)
        print(f"\n  [Fold {fold}] Best val loss: {val_loss:.4f}")

    best_fold_idx = int(np.argmin(fold_val_losses))
    best_fold_num = best_fold_idx + 1
    suffix = ".pkl" if model_choice in ("svm", "rf") else ".pt"
    shutil.copy(fold_model_paths[best_fold_idx], models_dir / f"best_model{suffix}")

    print(f"\n{'='*55}")
    for i, vl in enumerate(fold_val_losses):
        marker = "  <-- BEST" if i == best_fold_idx else ""
        print(f"  Fold {i+1}: best val loss = {vl:.4f}{marker}")
    print(f"\n  Best model saved from Fold {best_fold_num} -> best_model{suffix}")

    return fold_histories, fold_val_losses, best_fold_num
