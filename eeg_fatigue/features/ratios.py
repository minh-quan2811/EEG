import numpy as np


def _ratios_from_bands(bands):
    if bands is None:
        return None
    ratios = {}
    beta  = bands.get("beta",  0)
    alpha = bands.get("alpha", 0)
    theta = bands.get("theta", 0)

    if beta != 0:
        ratios["theta_beta"]       = theta / beta
        ratios["alpha_beta"]       = alpha / beta
        ratios["theta_alpha_beta"] = (theta + alpha) / beta
    if alpha != 0:
        ratios["theta_alpha"] = theta / alpha

    total = sum(bands.values())
    ratios["total_power"] = total
    if total != 0:
        for band, val in bands.items():
            ratios[f"{band}_relative"] = (val / total) * 100

    return ratios


def _detect_mode(power_dict):
    core_bands = {"delta", "theta", "alpha", "beta", "gamma"}
    if all(k in core_bands for k in power_dict):
        return "global", []

    suffixes = set()
    for key in power_dict:
        for band in core_bands:
            if key.startswith(band + "_"):
                suffixes.add(key[len(band) + 1:])
                break
    return "suffixed", sorted(suffixes)


def compute_ratios(power_dict):
    if power_dict is None:
        return None

    mode, suffixes = _detect_mode(power_dict)

    if mode == "global":
        return _ratios_from_bands(power_dict)

    # channel or region mode
    result = {}
    tab_values = []

    for suffix in suffixes:
        bands = {
            band: power_dict[f"{band}_{suffix}"]
            for band in ("delta", "theta", "alpha", "beta", "gamma")
            if f"{band}_{suffix}" in power_dict
        }
        ratios = _ratios_from_bands(bands)
        if ratios is None:
            continue
        for name, val in ratios.items():
            result[f"{name}_{suffix}"] = val
        if "theta_alpha_beta" in ratios:
            tab_values.append(ratios["theta_alpha_beta"])

    # scalar used by fatigue labelling — always present
    result["theta_alpha_beta"] = float(np.mean(tab_values)) if tab_values else np.nan

    return result if result else None


def compute_ratio(power_dict):
    """Returns scalar theta_alpha_beta. Used by labelling pipeline."""
    result = compute_ratios(power_dict)
    if result is None:
        return np.nan
    return result.get("theta_alpha_beta", np.nan)