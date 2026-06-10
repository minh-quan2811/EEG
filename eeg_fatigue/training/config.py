MODEL_CHOICE = "mlp"  # "resnet" | "mlp" | "svm" | "rf"

MODEL_CONFIG = {
    "input_size":    None,
    "num_classes":   3,     # updated from cfg.LEVEL_LABELS at runtime
    "hidden_dim":    64,
    "num_blocks":    3,     # resnet only
    "dropout":       0.3,
    "learning_rate": 1e-3,
    "batch_size":    64,
    "epochs":        50,
    "weight_decay":  1e-4,
}

N_FOLDS = 5
MIN_CORRELATION = 0.5
SIGNIFICANCE_ALPHA = 0.05
