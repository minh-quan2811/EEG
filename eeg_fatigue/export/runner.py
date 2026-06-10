from pathlib import Path
from .dataframe import build_summary_df, build_feature_df
from .csv_writer import save_results_csv
from ..features.band_power import compute_band_power_per_window
from ..features.ratios import compute_ratios


def build_summary_dataframe(results, fatigue_levels, subjects, cfg):
    df = build_summary_df(results, fatigue_levels, subjects, cfg)
    if df is None or df.empty:
        print("[WARN] Summary DataFrame is empty.")
        return None
    print(f"  Total rows (windows): {len(df)}")
    print(df.head(10).to_string(index=False))
    return df


def build_feature_dataframe(subjects, fatigue_levels, cfg):
    return build_feature_df(
        valid_subjects=subjects,
        fatigue_levels=fatigue_levels,
        config=cfg,
        band_power_func=compute_band_power_per_window,
        ratios_func=compute_ratios,
    )


def export_all(df, results_dir: Path):
    save_results_csv(df, results_dir=results_dir)
