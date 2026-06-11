# EEG Fatigue Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/MNE-EEG%20Processing-9cf" />
  <img src="https://img.shields.io/badge/SciPy-Statistics-orange" />
</p>

Research project conducted in collaboration with Universitas Padjadjaran (UNPAD), Indonesia, to investigate mental fatigue using EEG brain signals. The study analyzes changes in brainwave activity before and after work sessions and applies statistical methods to identify fatigue-related patterns in neural activity.

---

## What It Does

The brain produces different electrical signals depending on how tired it is. This project reads raw EEG files, extracts those signals, and classifies each moment into a fatigue level — No Fatigue, Low Fatigue, or Mild Fatigue.

It then runs statistical tests to confirm whether fatigue genuinely increased after work, and shows which brainwave features are the strongest indicators.

---

## How It Works

1. **Load subjects** — finds valid subject folders in `before/` and `after/` directories
2. **Extract features** — slices each EEG file into 4-second windows and computes band power (delta, theta, alpha, beta, gamma)
3. **Compute fatigue levels** — calculates a Z-score per window against each subject's personal baseline, then assigns a fatigue label
4. **Run statistical tests** — Wilcoxon signed-rank, Kruskal-Wallis, and Chi-square
5. **Run correlation analysis** — Pearson, Spearman, and ANOVA to find the best EEG features
6. **Train a classifier** — ResNet, MLP, SVM, or Random Forest with k-fold cross-validation
7. **Export results** — saves all plots, CSVs, config, and summary to the results folder

---

## Usage

```bash
pip install -r requirements.txt
```

```bash
# Run analysis
python -m eeg_fatigue analyze

# Run training
python -m eeg_fatigue train --model mlp
python -m eeg_fatigue train --model resnet
python -m eeg_fatigue train --model svm
python -m eeg_fatigue train --model rf
```

Optional overrides:
```bash
python -m eeg_fatigue analyze --agg-mode region --window-sec 2.0
python -m eeg_fatigue train --model mlp --folds 10 --epochs 100 --subjects sub-01 sub-02
```

Results are saved to `results/<command>_<agg_mode>_<n>/`.