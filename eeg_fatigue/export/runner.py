from pathlib import Path
from .results import build_dataframe, build_feature_dataframe as _build_feature_df, save_results_csv
from ..features.band_power import compute_band_power_per_window
from ..features.ratios import compute_ratios


def build_summary_dataframe(results, fatigue_levels, subjects, cfg):
    df_summary = build_dataframe(results, fatigue_levels, subjects, cfg)
    if df_summary.empty:
        print("[WARN] Summary DataFrame is empty — check .set files loaded correctly.")
        return None
    print(f"  Total rows (windows): {len(df_summary)}")
    print(df_summary.head(10).to_string(index=False))
    return df_summary


def build_feature_dataframe(subjects, fatigue_levels, cfg):
    return _build_feature_df(
        valid_subjects=subjects,
        fatigue_levels=fatigue_levels,
        config=cfg,
        band_power_func=compute_band_power_per_window,
        ratios_func=compute_ratios,
    )


def export_results(df_features, results_dir):
    save_results_csv(df_features, results_dir=results_dir)