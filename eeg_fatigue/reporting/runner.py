from pathlib import Path
from .config_writer import save_config_txt
from .summary_writer import save_analysis_summary, save_training_summary


def save_all(cfg, results: dict, results_dir: Path, train_cfg=None):
    save_config_txt(cfg, results_dir, train_cfg=train_cfg)

    if train_cfg is not None:
        save_training_summary(results, results_dir)
    else:
        save_analysis_summary(results, results_dir)

    print(f"\n[OK] All results saved -> {results_dir}")
