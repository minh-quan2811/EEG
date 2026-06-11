from pathlib import Path

from .helpers import config_lines, wilcoxon_lines, kruskal_lines, chi2_lines, correlation_lines


def save_analysis_summary(analysis_results: dict, results_dir: Path, cfg):
    lines = config_lines(cfg)

    lines += [
        "=" * 60,
        "STATISTICAL SUMMARY",
        "=" * 60,
    ]
    lines += wilcoxon_lines(analysis_results.get("w_dict"))
    lines += kruskal_lines(
        analysis_results.get("level_ratios"),
        analysis_results.get("h_stat"),
        analysis_results.get("p_kw"),
    )
    lines += chi2_lines(
        analysis_results.get("chi2_result"),
        analysis_results.get("contingency_table"),
    )
    lines += correlation_lines(
        analysis_results.get("df_corr"),
        analysis_results.get("df_spearman"),
    )

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "analysis_summary.txt").write_text(summary, encoding="utf-8")

    cfg_path = results_dir / "config.txt"
    if cfg_path.exists():
        cfg_path.unlink()