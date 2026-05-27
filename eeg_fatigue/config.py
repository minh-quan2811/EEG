import os

BASE_PATH = r"C:\Users\Admin\Desktop\School_Projects\git repositories\EEG\data\counting"
AFTER_PATH = os.path.join(BASE_PATH, "after")
BEFORE_PATH = os.path.join(BASE_PATH, "before")

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13"]

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

Z_THRESHOLDS = [1, 2]

LEVEL_LABELS = {
    0: "No Fatigue",
    1: "Low Fatigue",
    2: "Mild Fatigue",
}

LEVEL_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8e44ad",
    5: "#2980b9",
}