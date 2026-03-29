#!/usr/bin/env python3
"""Market-state regime clustering over raw tick files plus flow validation outputs."""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR_RAW = os.environ.get(
    "FLOW_MASTER_ROOT",
    r"<DATASET_PATH>",
)
ALLOWED_MONTHS = {"feb2026", "march2026"}
TAU_SECONDS = 5.0
N_CLUSTERS = 5
RANDOM_STATE = 42
DEFAULT_MAX_WORKERS = max(1, min(12, (os.cpu_count() or 12)))
MAX_WORKERS = max(1, int(os.environ.get("FLOW_CLUSTER_V2_WORKERS", str(DEFAULT_MAX_WORKERS))))
EPS = 1e-12
CSV_COLUMNS = [
    "timestamp",
    "last_traded_price",
    "last_traded_quantity",
    "Bid_Price_1",
    "Bid_Quantity_1",
    "Ask_Price_1",
    "Ask_Quantity_1",
    "spot_ltp",
]
RESULT_FILE_RE = re.compile(r"^\d{8}_result\.json$")
FILE_RE = re.compile(r"MASTER_\d+_(\d+)_(CE|PE)\.csv$", re.IGNORECASE)


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


def parse_file_metadata(file_name: str) -> tuple[float, str | None, int | None]:
    match = FILE_RE.match(str(file_name).strip())
    if not match:
        return float("nan"), None, None

    strike = safe_float(match.group(1), default=np.nan)
    option_type = match.group(2).upper()
    option_type_encoded = 1 if option_type == "CE" else 0
    return float(strike), option_type, option_type_encoded


def time_aware_ema(values: np.ndarray, dt_seconds: np.ndarray, tau_seconds: float) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    state = float("nan")

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            out[idx] = state
            continue

        if not np.isfinite(state):
            state = float(value)
            out[idx] = state
            continue

        dt = dt_seconds[idx]
        if not np.isfinite(dt) or dt <= 0.0:
            alpha = 0.0
        else:
            alpha = 1.0 - math.exp(-dt / tau_seconds)
        state = state + alpha * (float(value) - state)
        out[idx] = state

    return out


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


def load_result_map(result_files: list[Path]) -> dict[tuple[str, str], dict]:
    result_map: dict[tuple[str, str], dict] = {}
    for result_path in result_files:
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[FLOW-CLUSTER-V2] failed to read {result_path.name}: {exc}")
            continue

        date = str(payload.get("date", result_path.parent.name))
        per_file = payload.get("per_file", [])
        if not isinstance(per_file, list):
            continue

        for row in per_file:
            if not isinstance(row, dict):
                continue
            key = (date, str(row.get("file", "")))
            result_map[key] = {
                "pearson_residual": safe_float(row.get("pearson_residual"), default=np.nan),
                "spearman_residual": safe_float(row.get("spearman_residual"), default=np.nan),
                "beta": safe_float(row.get("beta"), default=np.nan),
                "n_samples": safe_float(row.get("n_samples"), default=np.nan),
            }
    return result_map


def collect_csv_tasks(root_dir: Path, result_map: dict[tuple[str, str], dict]) -> list[tuple[str, Path, dict]]:
    tasks = []
    for month_dir in sorted(root_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        if ALLOWED_MONTHS and month_dir.name not in ALLOWED_MONTHS:
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            date = day_dir.name
            for csv_path in sorted(day_dir.glob("MASTER_*.csv")):
                metrics = result_map.get((date, csv_path.name))
                if metrics is None:
                    continue
                tasks.append((date, csv_path, metrics))
    return tasks


def load_tick_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=CSV_COLUMNS, low_memory=False)
    except Exception as exc:
        print(f"[FLOW-CLUSTER-V2] failed to read {path.name}: {exc}")
        return None

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        return None

    numeric_cols = [col for col in CSV_COLUMNS if col != "timestamp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(df) < 3:
        return None

    ts_ns = df["timestamp"].astype("int64", copy=False).to_numpy()
    time_seconds = (ts_ns - ts_ns[0]) / 1e9
    dt_seconds = np.diff(time_seconds, prepend=np.nan)
    dt_safe = np.where(np.isfinite(dt_seconds) & (dt_seconds > 0.0), dt_seconds, 1e-6)

    df["time_seconds"] = time_seconds
    df["dt_seconds"] = dt_seconds
    df["dt_safe"] = dt_safe
    return df


def compute_state_features(df: pd.DataFrame, strike: float, option_type_encoded: int) -> dict:
    ltp = df["last_traded_price"].to_numpy(dtype=float)
    qty = df["last_traded_quantity"].to_numpy(dtype=float)
    bid = df["Bid_Price_1"].to_numpy(dtype=float)
    ask = df["Ask_Price_1"].to_numpy(dtype=float)
    bid_qty = df["Bid_Quantity_1"].to_numpy(dtype=float)
    ask_qty = df["Ask_Quantity_1"].to_numpy(dtype=float)
    spot = df["spot_ltp"].to_numpy(dtype=float)
    dt_safe = df["dt_safe"].to_numpy(dtype=float)
    time_seconds = df["time_seconds"].to_numpy(dtype=float)

    valid_bid = np.isfinite(bid) & (bid > 0.0)
    valid_ask = np.isfinite(ask) & (ask > 0.0)
    signed_volume = np.where(
        valid_ask & (ltp >= ask),
        qty,
        np.where(valid_bid & (ltp <= bid), -qty, 0.0),
    )

    intensity_raw = np.abs(signed_volume) / dt_safe
    intensity_raw[0] = np.nan
    intensity = time_aware_ema(intensity_raw, dt_safe, TAU_SECONDS)

    duration = max(float(time_seconds[-1] - time_seconds[0]), 1e-6)
    trade_count = int(np.sum(np.isfinite(qty) & (qty > 0.0)))
    volume_total = float(np.nansum(np.where(np.isfinite(qty), qty, 0.0)))

    spread = np.where(valid_bid & valid_ask, ask - bid, np.nan)
    depth = np.where(np.isfinite(bid_qty) & np.isfinite(ask_qty), bid_qty + ask_qty, np.nan)
    imbalance_denom = bid_qty + ask_qty
    book_imbalance = np.divide(
        bid_qty - ask_qty,
        imbalance_denom,
        out=np.full(len(df), np.nan, dtype=float),
        where=np.isfinite(imbalance_denom) & (np.abs(imbalance_denom) > EPS),
    )

    valid_ltp = np.isfinite(ltp) & (ltp > 0.0)
    log_ltp = np.where(valid_ltp, np.log(ltp), np.nan)
    returns = np.diff(log_ltp, prepend=np.nan)
    prev_valid = np.roll(valid_ltp, 1)
    prev_valid[0] = False
    returns[~(valid_ltp & prev_valid)] = np.nan

    moneyness = np.abs(spot - strike)

    return {
        "flow_intensity_mean": safe_float(np.nanmean(intensity), default=np.nan),
        "flow_intensity_std": safe_float(np.nanstd(intensity), default=np.nan),
        "trade_rate": safe_float(trade_count / duration, default=np.nan),
        "volume_rate": safe_float(volume_total / duration, default=np.nan),
        "spread_mean": safe_float(np.nanmean(spread), default=np.nan),
        "depth_mean": safe_float(np.nanmean(depth), default=np.nan),
        "book_imbalance_mean": safe_float(np.nanmean(book_imbalance), default=np.nan),
        "return_std": safe_float(np.nanstd(returns, ddof=1), default=np.nan),
        "realized_variance": safe_float(np.nansum(returns ** 2), default=np.nan),
        "moneyness": safe_float(np.nanmean(moneyness), default=np.nan),
        "option_type_encoded": float(option_type_encoded),
    }


def process_csv_task(task: tuple[str, Path, dict]) -> dict | None:
    date, csv_path, metrics = task
    strike, option_type, option_type_encoded = parse_file_metadata(csv_path.name)
    if option_type_encoded is None or not np.isfinite(strike):
        return None

    df = load_tick_file(csv_path)
    if df is None:
        return None

    state_features = compute_state_features(df, strike=float(strike), option_type_encoded=option_type_encoded)
    required_values = list(state_features.values()) + [
        metrics.get("pearson_residual"),
        metrics.get("spearman_residual"),
        metrics.get("beta"),
        metrics.get("n_samples"),
    ]
    if not all(np.isfinite(safe_float(v, default=np.nan)) for v in required_values):
        return None

    return {
        "date": date,
        "file": csv_path.name,
        "option_type": option_type,
        "strike": float(strike),
        **{k: float(v) for k, v in state_features.items()},
        "pearson_residual": float(metrics["pearson_residual"]),
        "spearman_residual": float(metrics["spearman_residual"]),
        "beta": float(metrics["beta"]),
        "n_samples": float(metrics["n_samples"]),
    }


def build_regime_dataset(tasks: list[tuple[str, Path, dict]]) -> pd.DataFrame:
    rows = []
    total = len(tasks)
    if total == 0:
        return pd.DataFrame()

    print(f"[FLOW-CLUSTER-V2] csv_tasks={total} workers={MAX_WORKERS}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_csv_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
                if row is not None:
                    rows.append(row)
            except Exception as exc:
                print(f"[FLOW-CLUSTER-V2] failed {task[1].name}: {exc}")

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"[FLOW-CLUSTER-V2] progress {completed}/{total}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows).sort_values(["date", "file"]).reset_index(drop=True)


def fit_clusters(dataset_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    state_feature_columns = [
        "flow_intensity_mean",
        "flow_intensity_std",
        "trade_rate",
        "volume_rate",
        "spread_mean",
        "depth_mean",
        "book_imbalance_mean",
        "return_std",
        "realized_variance",
        "moneyness",
        "option_type_encoded",
    ]
    x = dataset_df[state_feature_columns].to_numpy(dtype=float)
    x_scaled, mean, scale = standard_scale(x)

    backend = "numpy"
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        mean = scaler.mean_
        scale = scaler.scale_
        model = KMeans(n_clusters=min(N_CLUSTERS, len(dataset_df)), random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(x_scaled)
        centers = model.cluster_centers_
        backend = "sklearn"
    except Exception:
        labels, centers = kmeans_numpy(x_scaled, N_CLUSTERS, RANDOM_STATE)

    meta = {
        "backend": backend,
        "feature_columns": state_feature_columns,
        "n_clusters": int(len(np.unique(labels))),
        "scaler_mean": [to_python_float(v) for v in mean],
        "scaler_scale": [to_python_float(v) for v in scale],
        "cluster_centers_scaled": centers.tolist(),
    }
    return labels, meta


def build_cluster_summary(dataset_df: pd.DataFrame) -> dict:
    clusters = []
    for cluster_id in sorted(dataset_df["cluster_id"].unique()):
        cluster_df = dataset_df[dataset_df["cluster_id"] == cluster_id]
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(cluster_df)),
                "mean_pearson_residual": to_python_float(cluster_df["pearson_residual"].mean()),
                "std_pearson_residual": to_python_float(cluster_df["pearson_residual"].std(ddof=0)),
                "mean_spearman_residual": to_python_float(cluster_df["spearman_residual"].mean()),
                "mean_beta": to_python_float(cluster_df["beta"].mean()),
                "mean_flow_intensity": to_python_float(cluster_df["flow_intensity_mean"].mean()),
                "mean_spread": to_python_float(cluster_df["spread_mean"].mean()),
                "mean_moneyness": to_python_float(cluster_df["moneyness"].mean()),
            }
        )

    return {
        "total_files": int(len(dataset_df)),
        "clusters": clusters,
    }


def main():
    root_dir = normalize_root_path(ROOT_DIR_RAW)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    result_files = collect_result_files(root_dir)
    if not result_files:
        raise FileNotFoundError(f"No <day>_result.json files found under {root_dir}")

    print(f"[FLOW-CLUSTER-V2] root={root_dir}")
    print(f"[FLOW-CLUSTER-V2] months={sorted(ALLOWED_MONTHS)}")
    print(f"[FLOW-CLUSTER-V2] result_files={len(result_files)}")

    result_map = load_result_map(result_files)
    tasks = collect_csv_tasks(root_dir, result_map)
    dataset_df = build_regime_dataset(tasks)
    if dataset_df.empty:
        raise RuntimeError("No valid file-level dataset rows could be built")

    labels, meta = fit_clusters(dataset_df)
    dataset_df = dataset_df.copy()
    dataset_df["cluster_id"] = labels

    dataset_path = root_dir / "regime_dataset_v2.csv"
    dataset_df.to_csv(dataset_path, index=False)

    summary = build_cluster_summary(dataset_df)
    summary["root_dir"] = str(root_dir)
    summary["backend"] = meta["backend"]
    summary["feature_columns"] = meta["feature_columns"]
    summary["n_clusters"] = meta["n_clusters"]

    summary_path = root_dir / "cluster_summary_v2.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[FLOW-CLUSTER-V2] dataset_rows={len(dataset_df)}")
    print(f"[FLOW-CLUSTER-V2] backend={meta['backend']} n_clusters={meta['n_clusters']}")
    print(f"[FLOW-CLUSTER-V2] wrote {dataset_path}")
    print(f"[FLOW-CLUSTER-V2] wrote {summary_path}")


if __name__ == "__main__":
    main()
