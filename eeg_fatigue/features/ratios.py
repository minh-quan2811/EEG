import numpy as np

def compute_ratio(power_dict):
    """Compute (Theta + Alpha) / Beta ratio. Returns NaN if power_dict is None or beta is zero."""
    if power_dict is None:
        return np.nan
    beta = power_dict.get("beta", 0)
    if beta == 0:
        return np.nan
    return (power_dict["theta"] + power_dict["alpha"]) / beta


def compute_ratios(power_dict):
    """Compute various ratios from band powers. Returns dict with ratio values."""
    if power_dict is None:
        return None

    ratios = {}

    if power_dict.get("beta", 0) != 0:
        ratios["theta_beta"] = power_dict["theta"] / power_dict["beta"]
        ratios["alpha_beta"] = power_dict["alpha"] / power_dict["beta"]
        ratios["theta_alpha_beta"] = (power_dict["theta"] + power_dict["alpha"]) / power_dict["beta"]

    if power_dict.get("alpha", 0) != 0:
        ratios["theta_alpha"] = power_dict["theta"] / power_dict["alpha"]

    ratios["total_power"] = sum(power_dict.values())

    total = ratios["total_power"]
    if total != 0:
        for band in power_dict.keys():
            ratios[f"{band}_relative"] = (power_dict[band] / total) * 100

    return ratios
