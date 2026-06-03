# EEG Fatigue Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/MNE-EEG%20Processing-9cf" />
  <img src="https://img.shields.io/badge/SciPy-Statistics-orange" />
  <img src="" />
</p>

Research project conducted in collaboration with Universitas Padjadjaran (UNPAD), Indonesia, to investigate mental fatigue using EEG brain signals. The study analyzes changes in brainwave activity before and after work sessions and applies statistical methods to identify fatigue-related patterns in neural activity.

---

## What It Does

The brain produces different electrical signals depending on how tired it is. This project reads raw EEG files, extracts those signals, and classifies each moment into a fatigue level — No Fatigue, Low Fatigue, or Mild Fatigue.

It then runs statistical tests to confirm whether fatigue genuinely increased after work, and shows which brainwave features are the strongest indicators.

---

## How It Works

The pipeline runs in 8 steps:

1. **Load subjects** — finds valid subject folders in `before/` and `after/` data directories
2. **Extract features** — slices each EEG file into 4-second windows and computes band power (delta, theta, alpha, beta, gamma)
3. **Compute fatigue levels** — calculates a Z-score per window against each subject's personal baseline, then assigns a fatigue label
4. **Build summary DataFrame** — collects all windows, ratios, Z-scores, and labels into one table
5. **Run statistical tests** — Wilcoxon signed-rank, Kruskal-Wallis, and Chi-square tests
6. **Visualize ratio analysis** — before vs after charts, session trends, Z-score distributions
7. **Run correlation analysis** — Pearson, Spearman, and ANOVA to find the best EEG features
8. **Export results** — saves all data and plots to the `results/` folder

---

## How to Run

Install dependencies:

```bash
pip install requirements.txt
```

Run the pipeline:

```bash
python -m eeg_fatigue.main
```

Results are saved to `results/` by default. To use a custom output folder:

```python
from eeg_fatigue.main import run
run(results_dir="my_output_folder")
```
