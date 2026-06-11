from pathlib import Path

from .. import data, features, labeling
from ..training import runner as training
from ..reporting import runner as reporting


def run_train(cfg, train_cfg, results_dir: Path):
    print("=" * 60)
    print("EEG FATIGUE TRAINING")
    print("=" * 60)

    print("\n[1/6] Loading subjects...")
    subjects = data.runner.load_subjects(cfg)
    if not subjects:
        return

    print("\n[2/6] Extracting features...")
    results = features.runner.extract_features(subjects, cfg)

    print("\n[3/6] Computing fatigue levels...")
    fatigue_levels = labeling.runner.label_fatigue(results, subjects, cfg)

    print("\n[4/6] Building training DataFrame...")
    df = training.build_feature_dataframe(subjects, fatigue_levels, cfg)
    if df is None or df.empty:
        print("[ABORT] Empty feature DataFrame.")
        return

    print("\n[5/6] Statistical validation gate...")
    passed, stat_lines = training.validate(df, train_cfg, results_dir)
    if not passed:
        return

    print("\n[6/6] Training...")
    train_results = training.run_training(df, cfg, train_cfg, results_dir)
    if not train_results:
        return

    train_results["stat_validation"] = stat_lines

    reporting.save_all(cfg, train_results, results_dir, train_cfg=train_cfg)