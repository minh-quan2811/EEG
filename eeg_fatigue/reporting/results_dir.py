import re
from pathlib import Path


def get_next_results_dir(run_type: str, agg_mode: str, base: Path = Path("results")) -> Path:
    run_type = re.sub(r"[^a-zA-Z0-9_]", "", run_type)
    agg_mode = re.sub(r"[^a-zA-Z0-9_]", "", agg_mode)
    prefix = f"{run_type}_{agg_mode}"
    i = 1
    while (base / f"{prefix}_{i}").exists():
        i += 1
    path = base / f"{prefix}_{i}"
    path.mkdir(parents=True)
    print(f"  Results folder: {path}")
    return path
