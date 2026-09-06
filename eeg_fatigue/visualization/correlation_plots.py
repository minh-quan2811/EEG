import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Global font settings
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


def plot_correlation_heatmap(
    df_features,
    top_features: list[str],
    target: str = "fatigue_level",
    results_dir: Path = Path("results"),
) -> None:
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
        linewidths=0.5,
        annot_kws={"fontsize": 10, "fontname": "Times New Roman"},
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "Correlation Heatmap of Top 15 Features and Fatigue Level",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    plt.savefig(
        results_dir / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def plot_top_6_features_boxplots(
    df_features,
    df_corr,
    target: str = "fatigue_level",
    results_dir: Path = Path("results"),
) -> plt.Figure:
    """Box plots for top 6 features across fatigue levels."""

    top_6_features = df_corr.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    for i, feature in enumerate(top_6_features):
        ax = axes[i]

        data_by_level = [
            df_features[df_features[target] == 0][feature].dropna(),
            df_features[df_features[target] == 1][feature].dropna(),
            df_features[df_features[target] == 2][feature].dropna(),
        ]

        bp = ax.boxplot(
            data_by_level,
            patch_artist=True,
            labels=["Level 0", "Level 1", "Level 2"],
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        r_val = df_corr[df_corr["feature"] == feature]["pearson_r"].values
        r_val = r_val[0] if len(r_val) > 0 else 0

        ax.set_title(
            f"{feature}\n(r = {r_val:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_ylabel(feature, fontsize=12)
        ax.set_xlabel("Fatigue Level", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Top 6 EEG Features Across Fatigue Levels",
        fontsize=20,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    plt.savefig(
        results_dir / "top_6_features_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return fig