import numpy as np

def segment_signal(signal, sfreq, window_sec=4.0, overlap=0.5):
    """Slice a 1D signal into overlapping windows. Returns 2D array of shape (n_windows, window_samples)."""
    window_samples = int(window_sec * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    starts = range(0, len(signal) - window_samples + 1, step_samples)
    return np.array([signal[s : s + window_samples] for s in starts])
