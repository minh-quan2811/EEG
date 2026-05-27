import numpy as np
from scipy import stats
from itertools import combinations

def wilcoxon_test(before_means, after_means):
    """Wilcoxon signed-rank test. Returns (stat, p_value, r_effect, results_dict)."""
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
    """Kruskal-Wallis test. Returns (h_stat, p_value)."""
    if len(groups) < 2:
        return None, None
    h_stat, p_kw = stats.kruskal(*groups)
    return h_stat, p_kw


def dunns_post_hoc(level_ratios, level_pairs):
    """Dunn's post-hoc pairwise comparisons with Bonferroni correction."""
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
    """Chi-square test on 2D contingency table. Returns (chi2, p_value, dof, cramers_v)."""
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    n_total = contingency_table.sum()
    cramers_v = np.sqrt(chi2 / (n_total * (min(contingency_table.shape) - 1)))
    return chi2, p_chi, dof, cramers_v


def effect_size_interpretation(value):
    """Interpret effect size (Cramér's V or abs(r))."""
    abs_v = abs(value)
    if abs_v > 0.5:
        return "Large"
    elif abs_v > 0.3:
        return "Medium"
    elif abs_v > 0.1:
        return "Small"
    else:
        return "Negligible"
