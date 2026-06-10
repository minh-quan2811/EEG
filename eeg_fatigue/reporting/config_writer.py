from pathlib import Path


def save_config_txt(cfg, results_dir: Path, train_cfg=None):
    lines = [
        "=" * 50,
        "EEG FATIGUE — CONFIG",
        "=" * 50,
        "",
        f"BASE_PATH    : {cfg.BASE_PATH}",
        f"AFTER_PATH   : {cfg.AFTER_PATH}",
        f"BEFORE_PATH  : {cfg.BEFORE_PATH}",
        "",
        f"SUBJECTS ({len(cfg.SUBJECTS)}):",
    ]
    for s in cfg.SUBJECTS:
        lines.append(f"  {s}")
    lines += [
        "",
        f"NUM_SESSIONS : {cfg.NUM_SESSIONS}",
        f"WINDOW_SEC   : {cfg.WINDOW_SEC}",
        f"OVERLAP      : {cfg.OVERLAP}",
        "",
        "BANDS:",
    ]
    for band, (lo, hi) in cfg.BANDS.items():
        lines.append(f"  {band:6s}: {lo} - {hi} Hz")
    lines += [
        "",
        f"AGG_MODE     : {cfg.AGG_MODE}",
        f"Z_THRESHOLDS : {cfg.Z_THRESHOLDS}",
        f"LEVEL_LABELS : {cfg.LEVEL_LABELS}",
    ]

    if train_cfg is not None:
        lines += [
            "",
            "=" * 50,
            "TRAINING CONFIG",
            "=" * 50,
            "",
            f"MODEL_CHOICE       : {train_cfg.MODEL_CHOICE}",
            f"N_FOLDS            : {train_cfg.N_FOLDS}",
            f"MIN_CORRELATION    : {train_cfg.MIN_CORRELATION}",
            f"SIGNIFICANCE_ALPHA : {train_cfg.SIGNIFICANCE_ALPHA}",
            "",
            "MODEL_CONFIG:",
        ]
        for k, v in train_cfg.MODEL_CONFIG.items():
            lines.append(f"  {k:20s}: {v}")

    lines.append("")
    txt = "\n".join(lines)
    (results_dir / "config.txt").write_text(txt)
    print(txt)
