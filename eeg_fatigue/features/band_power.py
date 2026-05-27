import numpy as np
import mne
from scipy.signal import welch
from .windowing import segment_signal

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def compute_band_power_per_window(filepath, bands, window_sec=4.0, overlap=0.5):
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        data = raw.get_data() * 1e6
        nperseg = int(window_sec * sfreq)
        n_channels = data.shape[0]
        n_windows = len(segment_signal(data[0], sfreq, window_sec, overlap))
        if n_windows == 0:
            return []
        band_ch_powers = {b: np.zeros((n_windows, n_channels)) for b in bands}
        for ch in range(n_channels):
            windows = segment_signal(data[ch], sfreq, window_sec, overlap)
            for w_idx, win in enumerate(windows):
                freqs, psd = welch(win, fs=sfreq, nperseg=nperseg)
                for band_name, (fmin, fmax) in bands.items():
                    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_ch_powers[band_name][w_idx, ch] = _trapz(psd[idx], freqs[idx])
        window_powers = []
        for w_idx in range(n_windows):
            pw = {band: float(np.mean(band_ch_powers[band][w_idx])) for band in bands}
            window_powers.append(pw)
        return window_powers
    except Exception as e:
        print(f"    [WARN] Could not load {filepath}: {e}")
        return []