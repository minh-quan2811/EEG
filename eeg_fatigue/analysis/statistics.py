import numpy as np
from scipy import stats
from itertools import combinations

from .. import config
from ..labeling.fatigue import build_baseline, classify_fatigue_level

def wilcoxon_test(before_means, after_means):
    if len(before_means) < 3:
        return None, None, None, {"N": len(before_means), "status": "insufficient"}
    stat, p_value = stats.wilcoxon(before_means, after_means)
    n = len(before_means)
    r_effect = 1 - (2 * stat) / (n * (n + 1))
    return stat, p_value, r_effect, {
        "N": n,
        "before_mean": np.mean(before_means),
        "before_std": np.std(before_means),
        "after_mean": np.mean(after_means),
        "after_std": np.std(after_means),
        "stat": stat,
        "p_value": p_value,
        "r_effect": r_effect,
    }


def kruskal_wallis_test(*groups):
    if len(groups) < 2:
        return None, None
    h_stat, p_kw = stats.kruskal(*groups)
    return h_stat, p_kw


def dunns_post_hoc(level_ratios, level_pairs):
    n_pairs = len(level_pairs)
    results = []
    for a, b in level_pairs:
        g1, g2 = level_ratios.get(a, []), level_ratios.get(b, [])
        if len(g1) < 2 or len(g2) < 2:
            results.append({"pair": (a, b), "status": "insufficient_data"})
            continue
        u_stat, p_raw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        p_corrected = min(p_raw * n_pairs, 1.0)
        results.append({
            "pair": (a, b),
            "u_stat": u_stat,
            "p_raw": p_raw,
            "p_corrected": p_corrected,
            "significant": p_corrected < 0.05
        })
    return results


def chi_square_test(contingency_table):
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    n_total = contingency_table.sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(contingency_table.shape) - 1)))
    return chi2, p_chi, dof, cramers_v


def effect_size_interpretation(value):
    abs_v = abs(value)
    if abs_v > 0.5:
        return "Large"
    elif abs_v > 0.3:
        return "Medium"
    elif abs_v > 0.1:
        return "Small"
    else:
        return "Negligible"


def run_wilcoxon_analysis(results, valid_subjects):
    print("\n" + "=" * 55)
    print("Wilcoxon Signed-Rank Test: Before vs After")
    print("=" * 55)
    before_means, after_means = [], []
    for subject in valid_subjects:
        b = [r for sess in results[subject]["before"].values()
             for r in sess if not np.isnan(r)]
        a = [r for sess in results[subject]["after"].values()
             for r in sess if not np.isnan(r)]
        if b and a:
            before_means.append(np.mean(b))
            after_means.append(np.mean(a))
    before_means = np.array(before_means)
    after_means = np.array(after_means)
    w_stat, w_p, w_r, w_dict = wilcoxon_test(before_means, after_means)
    if w_stat is not None:
        sig = "[SIG]" if w_dict["p_value"] < 0.05 else "[NS]"
        eff = effect_size_interpretation(w_dict["r_effect"])
        print(f"  N subjects        : {w_dict['N']}")
        print(f"  Before mean ratio : {w_dict['before_mean']:.4f} +/- {w_dict['before_std']:.4f}")
        print(f"  After  mean ratio : {w_dict['after_mean']:.4f} +/- {w_dict['after_std']:.4f}")
        print(f"  W statistic       : {w_dict['stat']:.4f}")
        print(f"  p-value           : {w_dict['p_value']:.4f}  {sig}")
        print(f"  Effect size r     : {w_dict['r_effect']:.4f}  ({eff})")
    else:
        print(f"  [SKIP] Not enough subjects (need >= 3, have {w_dict['N']}).")
    return w_stat, w_dict


def run_kruskal_wallis_analysis(results, valid_subjects):
    print("\n" + "=" * 60)
    n_levels = len(config.LEVEL_LABELS)
    print(f"Kruskal-Wallis Test: Are {n_levels} Levels Distinct?")
    print("=" * 60)
    level_ratios = {lvl: [] for lvl in config.LEVEL_LABELS}
    for subject in valid_subjects:
        bm, bsd = build_baseline(results, subject)
        if bm is None:
            continue
        for condition in ["before", "after"]:
            for session_ratios in results[subject][condition].values():
                for ratio in session_ratios:
                    if not np.isnan(ratio):
                        z = (ratio - bm) / bsd
                        level = classify_fatigue_level(z)
                        if not np.isnan(level):
                            level_ratios[int(level)].append(ratio)
    for lvl, label in config.LEVEL_LABELS.items():
        vals = level_ratios[lvl]
        if vals:
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            print(f"  {label:15s} (Level {lvl}): "
                  f"n={len(vals):4d}, median={np.median(vals):.4f}, IQR={iqr:.4f}")
    h_stat, p_kw = None, None
    groups = [level_ratios[lvl] for lvl in sorted(config.LEVEL_LABELS) if len(level_ratios[lvl]) > 0]
    if len(groups) >= 2:
        h_stat, p_kw = kruskal_wallis_test(*groups)
        sig = "[SIG]" if p_kw < 0.05 else "[NS]"
        print(f"\n  H = {h_stat:.4f},  p = {p_kw:.4f}  {sig}")
        all_level_ids = sorted(config.LEVEL_LABELS.keys())
        all_pairs = [(a, b) for a in all_level_ids for b in all_level_ids if b > a]
        dunn_results = dunns_post_hoc(level_ratios, all_pairs)
        print("\n  Post-hoc Dunn's (Bonferroni corrected, 6 pairs):")
        for res in dunn_results:
            if res.get("status") == "insufficient_data":
                print(f"    Level {res['pair'][0]} vs {res['pair'][1]}: insufficient data")
            else:
                a, b = res["pair"]
                la, lb = config.LEVEL_LABELS[a], config.LEVEL_LABELS[b]
                sig2 = "[SIG]" if res["significant"] else "[NS]"
                print(f"    {la} vs {lb}: "
                      f"U={res['u_stat']:.1f}, p(corr)={res['p_corrected']:.4f}  {sig2}")
    else:
        print("  [SKIP] Not enough non-empty groups.")
    return h_stat, p_kw, level_ratios


def run_chi_square_analysis(fatigue_levels, valid_subjects):
    print("\n" + "=" * 60)
    print("Chi-Square Test: Do Levels Shift Before vs After?")
    print("=" * 60)
    all_levels = sorted(config.LEVEL_LABELS.keys())
    contingency_data = {
        "before": {lvl: 0 for lvl in all_levels},
        "after": {lvl: 0 for lvl in all_levels},
    }
    for subject in valid_subjects:
        for condition in ["before", "after"]:
            for window_levels in fatigue_levels[subject][condition].values():
                for level in window_levels:
                    if not np.isnan(level):
                        contingency_data[condition][int(level)] += 1
    contingency_table = np.array([
        [contingency_data[c][lvl] for lvl in all_levels]
        for c in ["before", "after"]
    ])
    chi2_result = chi_square_test(contingency_table)
    chi2, p_chi, dof, cramers_v = chi2_result
    sig = "[SIG]" if p_chi < 0.05 else "[NS]"
    strength = effect_size_interpretation(cramers_v)
    print(f"  chi2={chi2:.4f},  df={dof},  p={p_chi:.4f}  {sig}")
    print(f"  Cramer's V = {cramers_v:.4f}  ({strength} effect)")
    print("\n  Contingency table (rows = condition, cols = Level 0-3):")
    for i, cond in enumerate(["Before", "After "]):
        row = contingency_table[i]
        total = row.sum()
        if total > 0:
            pcts = " | ".join([f"{v:3d} ({100*v/total:4.1f}%)" for v in row])
        else:
            pcts = "no data"
        print(f"    {cond}: {pcts}   total={total}")
    return chi2_result, contingency_table