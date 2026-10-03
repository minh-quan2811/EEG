import os
from dataclasses import dataclass


BASE_PATH = r"C:\Users\Admin\Desktop\School_Projects\git repositories\EEG\data\counting"
AFTER_PATH = os.path.join(BASE_PATH, "after")
BEFORE_PATH = os.path.join(BASE_PATH, "before")

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13"]
# SUBJECTS = ["sub-01", "sub-02"]

NUM_SESSIONS = 10
WINDOW_SEC = 2.0
OVERLAP = 0.6

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Z-score cut-offs: [1, 2] → 3 levels (Z<1, 1<=Z<2, Z>=2)
# Add a value to get more levels — e.g. [1, 2, 3] → 4 levels
Z_THRESHOLDS = [0.5, 2.5]

# Feature aggregation mode
AGG_MODE: str = "channel"    # "global" | "channel" | "region"
CORRELATION_THRESHOLD = 0.3

# 10-20 system
CHANNEL_REGIONS: dict[str, list[str]] = {
    "Frontal": [
        "Fp1", "Fp2",
        "F7", "F3", "Fz", "F4", "F8",
    ],
    "Central": [
        "C3", "Cz", "C4",
    ],
    "Parietal": [
        "P3", "Pz", "P4",
    ],
    "Temporal": [
        "T3", "T4", "T5", "T6",
    ],
    "Occipital": [
        "O1", "O2",
    ],
}


@dataclass(frozen=True)
class LevelDefinition:
    label: str
    color: str

# Count must match len(Z_THRESHOLDS) + 1.
LEVEL_DEFINITIONS: dict[int, LevelDefinition] = {
    0: LevelDefinition("Low Fatigue",   "#2ecc71"),
    1: LevelDefinition("Mild Fatigue",  "#f1c40f"),
    2: LevelDefinition("Severe Fatigue", "#e67e22"),
    # 3: LevelDefinition("High Fatigue", "#e74c3c"),
}

LEVEL_LABELS = {k: v.label for k, v in LEVEL_DEFINITIONS.items()}
LEVEL_COLORS = {k: v.color for k, v in LEVEL_DEFINITIONS.items()}