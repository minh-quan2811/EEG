from pathlib import Path
from scipy import stats


def run_statistical_validation(df, train_cfg, results_dir: Path) -> tuple[bool, list[str]]:
    """
    Statistic validation.
    """
    print("\n" + "=" * 55)
    print("STATISTICAL VALIDATION")
    print("=" * 55)

    passed = True
    report_lines = []
    alpha = train_cfg.SIGNIFICANCE_ALPHA

    before_means = df[df.condition == "before"].groupby("subject")["fatigue_level"].mean()
    after_means  = df[df.condition == "after"].groupby("subject")["fatigue_level"].mean()
    common = before_means.index.intersection(after_means.index)

    if len(common) >= 3:
        stat, p_w = stats.wilcoxon(before_means[common].values, after_means[common].values)
        sig  = p_w < alpha
        line = f"Wilcoxon  W={stat:.4f}  p={p_w:.4f}  {'[SIG]' if sig else '[NS - FAIL]'}"
        print(line)
        report_lines.append(line)
        if not sig:
            passed = False
    else:
        line = "Wilcoxon: not enough subjects"
        print(line)
        report_lines.append(line)
        passed = False

    exclude = {"subject", "condition", "session", "window", "fatigue_level"}
    feature_cols = [c for c in df.columns if c not in exclude]
    kw_feature   = feature_cols[0] if feature_cols else "fatigue_level"
    groups = [df[df.fatigue_level == lvl][kw_feature].values
              for lvl in sorted(df.fatigue_level.unique())]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) >= 2:
        h, p_kw = stats.kruskal(*groups)
        sig  = p_kw < alpha
        line = f"Kruskal-W H={h:.4f}  p={p_kw:.4f}  {'[SIG]' if sig else '[NS - FAIL]'}"
        print(line)
        report_lines.append(line)
        if not sig:
            passed = False
    else:
        line = "Kruskal-W: not enough groups"
        print(line)
        report_lines.append(line)
        passed = False

    if not passed:
        print("\n[EARLY STOP] Statistical tests failed. Fatigue levels are not significantly different.")
    else:
        print("\n[OK] Statistical validation passed.")

    return passed, report_lines