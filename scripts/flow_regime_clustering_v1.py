#!/usr/bin/env python3
"""Unsupervised regime discovery over flow_validation_v1 per-file results."""

from __future__ import annotations

import os
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR_RAW = os.environ.get(
    "FLOW_MASTER_ROOT",
    r"<DATASET_PATH>",
)
ALLOWED_MONTHS = {"feb2026", "march2026"}
N_CLUSTERS = 4
RANDOM_STATE = 42
RESULT_FILE_RE = re.compile(r"^\d{8}_result\.json$")
STRIKE_RE = re.compile(r"MASTER_\d+_(\d+)_(CE|PE)\.csv$", re.IGNORECASE)


def safe_float(value, default=np.nan):
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def to_python_float(value):
    value = safe_float(value, default=np.nan)
    return float(value) if np.isfinite(value) else None


def normalize_root_path(path_like: str | Path) -> Path:
    raw = str(path_like).strip()
    if os.name == "nt":
        if raw.startswith("/mnt/") and len(raw) > 7 and raw[5].isalpha() and raw[6] == "/":
            drive = raw[5].upper() + ":"
            rest = raw[7:].replace("/", "\\")
            return Path(drive + "\\" + rest)
        return Path(raw)

    if len(raw) >= 3 and raw[1] == ":" and raw[0].isalpha():
        drive = raw[0].lower()
        rest = raw[2:].replace("\\", "/")
        if rest.startswith("/"):
            rest = rest[1:]
        return Path(f"/mnt/{drive}/{rest}")

    return Path(raw)


def parse_file_metadata(file_name: str) -> tuple[float, int | None]:
    match = STRIKE_RE.match(str(file_name).strip())
    if not match:
        return float("nan"), None

    strike = safe_float(match.group(1), default=np.nan)
    option_type_encoded = 1 if match.group(2).upper() == "CE" else 0
    return strike, option_type_encoded


def collect_result_files(root_dir: Path) -> list[Path]:
    result_files = []
    for month_dir in sorted(root_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        if ALLOWED_MONTHS and month_dir.name not in ALLOWED_MONTHS:
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            result_path = day_dir / f"{day_dir.name}_result.json"
            if result_path.exists() and RESULT_FILE_RE.match(result_path.name):
                result_files.append(result_path)
    return result_files


def load_feature_rows(result_files: list[Path]) -> pd.DataFrame:
    rows = []
    for result_path in result_files:
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[FLOW-CLUSTER] failed to read {result_path.name}: {exc}")
            continue

        date = str(payload.get("date", result_path.parent.name))
        per_file = payload.get("per_file", [])
        if not isinstance(per_file, list):
            continue

        for row in per_file:
            if not isinstance(row, dict):
                continue

            strike, option_type_encoded = parse_file_metadata(row.get("file", ""))
            pearson_residual = safe_float(row.get("pearson_residual"), default=np.nan)
            spearman_residual = safe_float(row.get("spearman_residual"), default=np.nan)
            beta = safe_float(row.get("beta"), default=np.nan)
            n_samples = safe_float(row.get("n_samples"), default=np.nan)

            if option_type_encoded is None:
                continue
            if not all(np.isfinite(v) for v in (pearson_residual, spearman_residual, beta, n_samples, strike)):
                continue

            rows.append(
                {
                    "date": date,
                    "file": row.get("file"),
                    "option_type": row.get("option_type"),
                    "option_type_encoded": option_type_encoded,
                    "strike": float(strike),
                    "pearson_residual": float(pearson_residual),
                    "spearman_residual": float(spearman_residual),
                    "beta": float(beta),
                    "n_samples": float(n_samples),
                }
            )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df

    return df.sort_values(["date", "file"]).reset_index(drop=True)


def standard_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    scaled = (x - mean) / std_safe
    return scaled, mean, std_safe


def kmeans_numpy(x: np.ndarray, n_clusters: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = x.shape[0]
    if n_samples == 0:
        raise ValueError("No samples available for clustering")

    k = min(n_clusters, n_samples)
    initial_idx = rng.choice(n_samples, size=k, replace=False)
    centers = x[initial_idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(100):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for cluster_id in range(k):
            mask = new_labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = np.mean(x[mask], axis=0)
            else:
                new_centers[cluster_id] = x[rng.integers(0, n_samples)]

        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    return labels, centers


def fit_clusters(feature_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    feature_columns = [
        "spearman_residual",
        "beta",
        "n_samples",
        "option_type_encoded",
        "strike",
    ]
    x = feature_df[feature_columns].to_numpy(dtype=float)
    x_scaled, mean, scale = standard_scale(x)

    backend = "numpy"
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        mean = scaler.mean_
        scale = scaler.scale_
        model = KMeans(n_clusters=min(N_CLUSTERS, len(feature_df)), random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(x_scaled)
        centers = model.cluster_centers_
        backend = "sklearn"
    except Exception:
        labels, centers = kmeans_numpy(x_scaled, N_CLUSTERS, RANDOM_STATE)

    meta = {
        "feature_columns": feature_columns,
        "scaler_mean": [to_python_float(v) for v in mean],
        "scaler_scale": [to_python_float(v) for v in scale],
        "backend": backend,
        "n_clusters": int(len(np.unique(labels))),
        "cluster_centers_scaled": centers.tolist(),
    }
    return labels, meta


def cluster_summary(feature_df: pd.DataFrame) -> dict:
    total_files = int(len(feature_df))
    clusters = []

    for cluster_id in sorted(feature_df["cluster_id"].unique()):
        cluster_df = feature_df[feature_df["cluster_id"] == cluster_id]
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(cluster_df)),
                "mean_pearson_residual": to_python_float(cluster_df["pearson_residual"].mean()),
                "std_pearson_residual": to_python_float(cluster_df["pearson_residual"].std(ddof=0)),
                "mean_spearman_residual": to_python_float(cluster_df["spearman_residual"].mean()),
                "mean_beta": to_python_float(cluster_df["beta"].mean()),
                "mean_samples": to_python_float(cluster_df["n_samples"].mean()),
            }
        )

    return {
        "total_files": total_files,
        "clusters": clusters,
    }


def main():
    root_dir = normalize_root_path(ROOT_DIR_RAW)

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    result_files = collect_result_files(root_dir)
    if not result_files:
        raise FileNotFoundError(f"No <day>_result.json files found under {root_dir}")

    print(f"[FLOW-CLUSTER] root={root_dir}")
    print(f"[FLOW-CLUSTER] months={sorted(ALLOWED_MONTHS)}")
    print(f"[FLOW-CLUSTER] result_files={len(result_files)}")

    feature_df = load_feature_rows(result_files)
    if feature_df.empty:
        raise RuntimeError("No valid per-file rows available for clustering")

    labels, meta = fit_clusters(feature_df)
    feature_df = feature_df.copy()
    feature_df["cluster_id"] = labels

    dataset_path = root_dir / "feature_dataset.csv"
    feature_df.to_csv(dataset_path, index=False)

    summary = cluster_summary(feature_df)
    summary["root_dir"] = str(root_dir)
    summary["backend"] = meta["backend"]
    summary["feature_columns"] = meta["feature_columns"]
    summary["n_clusters"] = meta["n_clusters"]

    summary_path = root_dir / "cluster_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[FLOW-CLUSTER] dataset_rows={len(feature_df)}")
    print(f"[FLOW-CLUSTER] backend={meta['backend']} n_clusters={meta['n_clusters']}")
    print(f"[FLOW-CLUSTER] wrote {dataset_path}")
    print(f"[FLOW-CLUSTER] wrote {summary_path}")


if __name__ == "__main__":
    main()
