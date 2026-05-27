import os

BASE_PATH = r"C:\Users\Admin\Desktop\School_Projects\git repositories\EEG\data\counting"
AFTER_PATH = os.path.join(BASE_PATH, "after")
BEFORE_PATH = os.path.join(BASE_PATH, "before")

NUM_SESSIONS = 10

WINDOW_SEC = 4.0
OVERLAP = 0.5

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

BANDS_SIMPLE = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

LEVEL_LABELS = {
    0: "No Fatigue",
    1: "Low Fatigue",
    2: "Mild Fatigue",
    3: "High Fatigue"
}

LEVEL_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c"
}
