from pathlib import Path
from .analysis_writer import save_analysis_summary
from .training_writer import save_training_summary


def save_all(cfg, results: dict, results_dir: Path, train_cfg=None):
    if train_cfg is not None:
        save_training_summary(results, results_dir, cfg, train_cfg)
    else:
        save_analysis_summary(results, results_dir, cfg)

    print(f"\n[OK] All results saved -> {results_dir}")
