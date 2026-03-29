from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
VISUALS_DIR = ROOT / "visuals"


def load_dataset() -> pd.DataFrame:
    for name in ("regime_dataset_v2.csv", "feature_dataset.csv"):
        path = RESULTS_DIR / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No dataset CSV found in results/")


def load_cluster_summary() -> dict:
    path = RESULTS_DIR / "cluster_summary_v2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_histogram(series: pd.Series, title: str, xlabel: str, output_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=40)
    ax.axvline(series.mean(), linestyle="--", label=f"Mean = {series.mean():.4f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / output_name, dpi=150)
    plt.close(fig)


def make_cluster_plot(summary: dict) -> None:
    clusters = sorted(summary["clusters"], key=lambda row: row["cluster_id"])
    x = [row["cluster_id"] for row in clusters]
    y = [row["mean_pearson_residual"] for row in clusters]
    yerr = [row["std_pearson_residual"] for row in clusters]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, yerr=yerr, capsize=4)
    ax.set_title("Cluster Mean Residual Correlation (V2)")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean Pearson Residual")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "cluster_comparison.png", dpi=150)
    plt.close(fig)


def make_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["beta"], df["pearson_residual"], s=12, alpha=0.6)
    ax.set_title("Spot Beta vs Residual Pearson Correlation")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Pearson Residual Correlation")
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "beta_vs_correlation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    summary = load_cluster_summary()

    make_histogram(
        df["pearson_residual"],
        "Distribution of Residual Pearson Correlations",
        "pearson_residual",
        "correlation_distribution.png",
    )
    make_histogram(
        df["spearman_residual"],
        "Distribution of Residual Spearman Correlations",
        "spearman_residual",
        "spearman_distribution.png",
    )
    make_cluster_plot(summary)
    make_scatter(df)


if __name__ == "__main__":
    main()
