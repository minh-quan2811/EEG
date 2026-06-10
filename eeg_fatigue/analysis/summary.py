def build_summary(w_dict, level_ratios, h_stat, p_kw, chi2_result,
                  contingency_table, df_corr, df_spearman) -> dict:
    return {
        "w_dict": w_dict,
        "level_ratios": level_ratios,
        "h_stat": h_stat,
        "p_kw": p_kw,
        "chi2_result": chi2_result,
        "contingency_table": contingency_table,
        "df_corr": df_corr,
        "df_spearman": df_spearman,
    }
