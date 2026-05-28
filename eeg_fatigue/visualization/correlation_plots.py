import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_correlation_heatmap(df_features, top_features: list[str], target: str = "fatigue_level",
                             results_dir: Path = Path("results")) -> None:
    """Heatmap of top features vs fatigue level."""
    corr_matrix = df_features[top_features + [target]].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5
    )
    plt.title("Correlation Heatmap - Top 15 Features vs Fatigue Level", fontsize=14, fontweight="bold")
    plt.tight_layout()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_6_features_boxplots(df_features, df_corr, target: str = "fatigue_level",
                                 results_dir: Path = Path("results")) -> plt.Figure:
    """Box plots for top 6 features across fatigue levels."""
    top_6_features = df_corr.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_6_features):
        ax = axes[i]

        data_by_level = [
            df_features[df_features[target] == 0][feature].dropna(),
            df_features[df_features[target] == 1][feature].dropna(),
            df_features[df_features[target] == 2][feature].dropna()
        ]

        bp = ax.boxplot(data_by_level, patch_artist=True, labels=["Level 0", "Level 1", "Level 2"])

        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        r_val = df_corr[df_corr['feature'] == feature]['pearson_r'].values
        r_val = r_val[0] if len(r_val) > 0 else 0
        ax.set_title(f"{feature}\n(r = {r_val:.3f})", fontsize=10)
        ax.set_ylabel(feature)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Top 6 Features vs Fatigue Level", fontsize=14, fontweight="bold")
    plt.tight_layout()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "top_6_features_boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig

