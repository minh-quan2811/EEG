import os
import sys
import warnings
from pathlib import Path

from . import config
from .utils.logging import setup_logging
from .analysis.summary import print_statistical_summary
from .utils.helpers import print_completion_info
from . import data, features, labeling, export, analysis, visualization

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run(results_dir: Path = Path("results")) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    logger, original_stdout = setup_logging(str(results_dir / "analysis_output.txt"))
    try:
        sys.stdout = logger
        _execute_pipeline(config, results_dir)
    except Exception as e:
        import traceback
        print("\n[ERROR] Pipeline failed:")
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        print(f"Console output saved -> {results_dir / 'analysis_output.txt'}")


def _execute_pipeline(cfg, results_dir: Path = Path("results")) -> None:
    print("=" * 60)
    print("EEG FATIGUE ANALYSIS  -  Main Pipeline")
    print("=" * 60)

    print("\n[1/8] Loading subjects...")
    subjects = data.runner.load_subjects(cfg)
    if not subjects:
        return

    print("\n[2/8] Extracting features...")
    results = features.runner.extract_features(subjects, cfg)

    print("\n[3/8] Computing fatigue levels...")
    fatigue_levels = labeling.runner.label_fatigue(results, subjects, cfg)

    print("\n[4/8] Building summary DataFrame...")
    df_summary = export.runner.build_summary_dataframe(results, fatigue_levels, subjects, cfg)
    if df_summary is None:
        return

    print("\n[5/8] Running statistical tests...")
    w_dict, h_stat, p_kw, level_ratios, chi2_result, contingency_table = \
        analysis.statistic_runner.run_statistical_analysis(results, fatigue_levels, subjects, cfg)

    print("\n[5B/8] Visualizing ratio analysis...")
    visualization.runner.visualize_ratio_analysis(results, fatigue_levels, subjects, results_dir)

    print("\n[5C/8] Generating statistical visualizations...")
    visualization.runner.visualize_statistics(level_ratios, contingency_table, chi2_result, results_dir)

    print("\n[6/8] Building feature DataFrame...")
    df_features = export.runner.build_feature_dataframe(subjects, fatigue_levels, cfg)

    print("\n[7/8] Running correlation analysis...")
    df_corr, df_spearman = analysis.correlation_runner.run_correlation_analysis(df_features)

    print_statistical_summary(
        w_dict, level_ratios, h_stat, p_kw, chi2_result, contingency_table, df_corr, df_spearman, cfg
    )

    print("\n[8/8] Exporting results...")
    export.runner.export_results(df_features if df_features is not None and not df_features.empty else df_summary, results_dir)
    print_completion_info()


if __name__ == "__main__":
    run()
