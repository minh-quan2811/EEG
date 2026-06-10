from pathlib import Path


def save_results_csv(df, filename="eeg_fatigue_results.csv", results_dir=None):
    results_dir = Path(results_dir) if results_dir else Path("results")
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename
    df.to_csv(filepath, index=False)
    print(f"  Results saved -> {filepath}")
    return str(filepath)