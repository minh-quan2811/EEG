import numpy as np
import mne
from scipy.signal import welch

from .windowing import segment_signal

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _power_matrix(data, sfreq, bands, window_sec, overlap):
    n_channels = data.shape[0]
    nperseg    = int(window_sec * sfreq)
    n_windows  = len(segment_signal(data[0], sfreq, window_sec, overlap))
    if n_windows == 0:
        return {}, 0

    band_ch_powers = {b: np.zeros((n_windows, n_channels)) for b in bands}
    for ch in range(n_channels):
        windows = segment_signal(data[ch], sfreq, window_sec, overlap)
        for w_idx, win in enumerate(windows):
            freqs, psd = welch(win, fs=sfreq, nperseg=nperseg)
            for band, (fmin, fmax) in bands.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_ch_powers[band][w_idx, ch] = _trapz(psd[idx], freqs[idx])

    return band_ch_powers, n_windows


def _aggregate_global(band_ch_powers, n_windows, bands, **_):
    return [
        {band: float(np.mean(band_ch_powers[band][w])) for band in bands}
        for w in range(n_windows)
    ]


def _aggregate_channel(band_ch_powers, n_windows, bands, ch_names, **_):
    result = []
    for w in range(n_windows):
        pw = {}
        for band in bands:
            for ch_idx, ch in enumerate(ch_names):
                pw[f"{band}_{ch}"] = float(band_ch_powers[band][w, ch_idx])
        result.append(pw)
    return result


def _aggregate_region(band_ch_powers, n_windows, bands, ch_names, channel_regions, **_):
    # map each channel to its region
    ch_to_region = {}
    for region, members in channel_regions.items():
        upper = {m.upper() for m in members}
        for ch in ch_names:
            if ch.upper() in upper:
                ch_to_region[ch] = region

    region_indices = {}
    for ch_idx, ch in enumerate(ch_names):
        region = ch_to_region.get(ch, "Other")
        region_indices.setdefault(region, []).append(ch_idx)

    result = []
    for w in range(n_windows):
        pw = {}
        for band in bands:
            for region, indices in region_indices.items():
                pw[f"{band}_{region}"] = float(np.mean(band_ch_powers[band][w, indices]))
        result.append(pw)
    return result


_AGGREGATORS = {
    "global":  _aggregate_global,
    "channel": _aggregate_channel,
    "region":  _aggregate_region,
}


def compute_band_power_per_window(filepath, bands, window_sec=4.0, overlap=0.5,
                                   agg_mode="global", channel_regions=None):
    if agg_mode not in _AGGREGATORS:
        raise ValueError(f"Unknown agg_mode '{agg_mode}'. Choose: {list(_AGGREGATORS)}")
    if agg_mode == "region" and channel_regions is None:
        raise ValueError("channel_regions required when agg_mode='region'.")

    try:
        raw      = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        sfreq    = raw.info["sfreq"]
        ch_names = raw.info["ch_names"]
        data     = raw.get_data() * 1e6

        band_ch_powers, n_windows = _power_matrix(data, sfreq, bands, window_sec, overlap)
        if n_windows == 0:
            return []

        return _AGGREGATORS[agg_mode](
            band_ch_powers=band_ch_powers,
            n_windows=n_windows,
            bands=bands,
            ch_names=ch_names,
            channel_regions=channel_regions,
        )
    except Exception as e:
        print(f"    [WARN] Could not load {filepath}: {e}")
        return []