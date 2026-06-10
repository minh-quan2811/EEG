import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _build_parser():
    parser = argparse.ArgumentParser(prog="eeg_fatigue", description="EEG Fatigue Analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    # shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--results-dir", type=Path, default=None)
    shared.add_argument("--subjects", nargs="+", default=None)
    shared.add_argument("--agg-mode", choices=["global", "channel", "region"], default=None)
    shared.add_argument("--window-sec", type=float, default=None)
    shared.add_argument("--overlap", type=float, default=None)

    sub.add_parser("analyze", parents=[shared], help="Run analysis pipeline")

    train_p = sub.add_parser("train", parents=[shared], help="Run training pipeline")
    train_p.add_argument("--model", choices=["resnet", "mlp", "svm", "rf"], default="mlp")
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--folds", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--hidden-dim", type=int, default=None)
    train_p.add_argument("--lr", type=float, default=None)
    train_p.add_argument("--min-corr", type=float, default=None)

    return parser


def _apply_overrides(cfg, args):
    if args.subjects:
        cfg.SUBJECTS = args.subjects
    if args.agg_mode:
        cfg.AGG_MODE = args.agg_mode
    if args.window_sec:
        cfg.WINDOW_SEC = args.window_sec
    if args.overlap:
        cfg.OVERLAP = args.overlap


def main():
    parser = _build_parser()
    args = parser.parse_args()

    from . import config as cfg
    _apply_overrides(cfg, args)

    from .reporting.results_dir import get_next_results_dir
    from .utils.logging import setup_logging

    if args.command == "analyze":
        results_dir = args.results_dir or get_next_results_dir("analyze", cfg.AGG_MODE)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger, original_stdout = setup_logging(str(results_dir / "analysis_output.txt"))
        try:
            sys.stdout = logger
            from .pipelines.analyze import run_analyze
            run_analyze(cfg, results_dir)
        finally:
            sys.stdout = original_stdout

    elif args.command == "train":
        from .training import config as train_cfg
        train_cfg.MODEL_CHOICE = args.model
        if args.epochs:
            train_cfg.MODEL_CONFIG["epochs"] = args.epochs
        if args.folds:
            train_cfg.N_FOLDS = args.folds
        if args.batch_size:
            train_cfg.MODEL_CONFIG["batch_size"] = args.batch_size
        if args.hidden_dim:
            train_cfg.MODEL_CONFIG["hidden_dim"] = args.hidden_dim
        if args.lr:
            train_cfg.MODEL_CONFIG["learning_rate"] = args.lr
        if args.min_corr:
            train_cfg.MIN_CORRELATION = args.min_corr

        results_dir = args.results_dir or get_next_results_dir(args.model, cfg.AGG_MODE)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger, original_stdout = setup_logging(str(results_dir / "training_output.txt"))
        try:
            sys.stdout = logger
            from .pipelines.train import run_train
            run_train(cfg, train_cfg, results_dir)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
