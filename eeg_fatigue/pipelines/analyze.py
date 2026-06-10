from pathlib import Path

from .. import data, features, labeling, export, analysis, visualization
from ..reporting import runner as reporting


def run_analyze(cfg, results_dir: Path):
    print("=" * 60)
    print("EEG FATIGUE ANALYSIS")
    print("=" * 60)

    print("\n[1/7] Loading subjects...")
    subjects = data.runner.load_subjects(cfg)
    if not subjects:
        return

    print("\n[2/7] Extracting features...")
    results = features.runner.extract_features(subjects, cfg)

    print("\n[3/7] Computing fatigue levels...")
    fatigue_levels = labeling.runner.label_fatigue(results, subjects, cfg)

    print("\n[4/7] Building DataFrames...")
    df_summary = export.runner.build_summary_dataframe(results, fatigue_levels, subjects, cfg)
    if df_summary is None:
        return

    print("\n[5/7] Running statistical analysis...")
    analysis_results = analysis.runner.run_analysis(results, fatigue_levels, subjects, cfg)

    print("\n[6/7] Building feature DataFrame + correlation...")
    df_features = export.runner.build_feature_dataframe(subjects, fatigue_levels, cfg)
    corr_results = analysis.runner.run_correlation(df_features)
    analysis_results.update(corr_results)

    print("\n[7/7] Visualizing + exporting + reporting...")
    visualization.runner.visualize_all(results, fatigue_levels, subjects, analysis_results, results_dir)
    export.runner.export_all(df_features if df_features is not None and not df_features.empty else df_summary, results_dir)
    reporting.save_all(cfg, analysis_results, results_dir)
