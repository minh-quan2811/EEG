import torch.optim as optim


# Default settings used when a schedule type is picked but params are left empty.
SCHEDULER_DEFAULTS = {
    "plateau": {"patience": 5, "factor": 0.5, "min_lr": 1e-6},
    "cosine":  {"T_max": 50, "eta_min": 1e-6},
    "step":    {"step_size": 15, "gamma": 0.5},
    "none":    {},
}


def build_scheduler(optimizer, schedule_type: str = "plateau", schedule_params: dict | None = None):
    """Build a learning-rate scheduler based on schedule_type and schedule_params."""
    schedule_type = (schedule_type or "none").lower()
    if schedule_type not in SCHEDULER_DEFAULTS:
        raise ValueError(
            f"Unknown lr_schedule type: '{schedule_type}'. "
            f"Choose one of {list(SCHEDULER_DEFAULTS)}."
        )

    params = {**SCHEDULER_DEFAULTS[schedule_type], **(schedule_params or {})}

    if schedule_type == "none":
        return None, False

    if schedule_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=params["patience"],
            factor=params["factor"],
            min_lr=params["min_lr"],
        )
        return scheduler, True

    if schedule_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params["T_max"],
            eta_min=params["eta_min"],
        )
        return scheduler, False

    # schedule_type == "step"
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params["step_size"],
        gamma=params["gamma"],
    )
    return scheduler, False