import os
from dataclasses import dataclass


BASE_PATH = r"C:\Users\Admin\Desktop\School_Projects\git repositories\EEG\data\counting"
AFTER_PATH = os.path.join(BASE_PATH, "after")
BEFORE_PATH = os.path.join(BASE_PATH, "before")

SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-06", "sub-07", "sub-09", "sub-10", "sub-11", "sub-12", "sub-13"]
# SUBJECTS = ["sub-01", "sub-02"]

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

# Z-score cut-offs: [1, 2] → 3 levels (Z<1, 1<=Z<2, Z>=2)
# Add a value to get more levels — e.g. [1, 2, 3] → 4 levels
Z_THRESHOLDS = [1, 2]

# Feature aggregation mode
AGG_MODE: str = "global"    # "global" | "channel" | "region"
 
CHANNEL_REGIONS: dict[str, list[str]] = {
    "Frontal": [
        "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
    ],
    "Central": [
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
    ],
    "Parietal": [
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz",
    ],
    "Temporal": [
        "T7", "T8", "TP7", "TP8", "TP9", "TP10",
        "FT7", "FT8", "FT9", "FT10",
    ],
    "Occipital": [
        "O1", "O2", "Oz",
        "PO3", "PO4", "PO7", "PO8", "POz",
    ],
}

@dataclass(frozen=True)
class LevelDefinition:
    label: str
    color: str

# Count must match len(Z_THRESHOLDS) + 1.
LEVEL_DEFINITIONS: dict[int, LevelDefinition] = {
    0: LevelDefinition("No Fatigue",   "#2ecc71"),
    1: LevelDefinition("Low Fatigue",  "#f1c40f"),
    2: LevelDefinition("Mild Fatigue", "#e67e22"),
}

LEVEL_LABELS = {k: v.label for k, v in LEVEL_DEFINITIONS.items()}
LEVEL_COLORS = {k: v.color for k, v in LEVEL_DEFINITIONS.items()}