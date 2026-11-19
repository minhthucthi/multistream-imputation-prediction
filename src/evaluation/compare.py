"""
src/evaluation/compare.py
==========================
Comparison framework: run all imputation methods and collect metrics.

Provides utilities to:
- Run all baseline methods + MPIN on a data chunk
- Collect MAE/MRE/RMSE for each method
- Print comparison tables
- Save results to CSV/JSON

Usage:
------
    results = run_all_baselines(X_chunk, M_chunk, X_true, config)
    mpin_results = run_mpin(X_chunk, M_chunk, X_true, config)
    all_results = {**results, "MPIN": mpin_results}
    print_comparison_table(all_results)
    save_results(all_results, "results/metrics/icu_results.json")
"""

import numpy as np
import json
import csv
import os
import time
from typing import Dict, Any, Optional
import logging

from src.models.baselines import (
    mean_imputation,
    knn_imputation,
    feature_propagation,
    mice_imputation,
    mf_imputation
)
from src.models.similarity_graph import build_similarity_graph, build_knn_edge_index
from src.models.mpin import train_mpin
from src.evaluation.metrics import evaluate_all, create_evaluation_mask

logger = logging.getLogger(__name__)


def run_all_baselines(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    X_true: np.ndarray,
    config: dict,
    M_eval: Optional[np.ndarray] = None,
    adj: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline imputation methods and compute evaluation metrics.

    Methods included:
        - MEAN: Global mean imputation
        - KNN: K-Nearest Neighbor (k=5)
        - MICE: Multiple Imputation by Chained Equations
        - MF: Matrix Factorization
        - FeaProp (FP): Feature Propagation on similarity graph

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Observed data with NaN at missing positions.
        This is what methods "see" as input.
    M_chunk : np.ndarray, shape (N, D)
        Original observation mask (1=observed, 0=missing).
    X_true : np.ndarray, shape (N, D)
        Ground truth complete data (used for evaluation only).
    config : dict
        Configuration dictionary.
    M_eval : np.ndarray, optional
        Mask of positions to evaluate on (1=evaluate here).
        If None, evaluates on all missing positions (1 - M_chunk).
    adj : np.ndarray, optional
        Pre-built adjacency matrix. If None, builds one from X_chunk.

    Returns
    -------
    results : dict
        {
            "MEAN": {"mae": ..., "mre": ..., "rmse": ..., "time_s": ...},
            "KNN":  {"mae": ..., "mre": ..., "rmse": ..., "time_s": ...},
            ...
        }
    """
    # Default: evaluate on all originally missing positions
    if M_eval is None:
        M_eval = (1 - M_chunk).astype(np.float32)

    # Build adjacency for FeaProp if not provided
    if adj is None:
        logger.info("Building similarity graph for FeaProp...")
        adj, _ = build_similarity_graph(
            X_chunk, M_chunk,
            k=config.get("mpin", {}).get("K", 10)
        )

    baseline_cfg = config.get("baselines", {})
    knn_k = baseline_cfg.get("knn", {}).get("k", 5)
    fp_iter = baseline_cfg.get("feature_propagation", {}).get("num_iterations", 10)
    mf_factors = baseline_cfg.get("mf", {}).get("num_factors", 20)

    methods = {}

    # --- MEAN ---
    logger.info("Running MEAN imputation...")
    t0 = time.time()
    X_mean = mean_imputation(X_chunk, M_chunk)
    methods["MEAN"] = {
        **evaluate_all(X_true, X_mean, M_eval),
        "time_s": time.time() - t0
    }

    # --- KNN ---
    logger.info(f"Running KNN imputation (k={knn_k})...")
    t0 = time.time()
    X_knn = knn_imputation(X_chunk, M_chunk, k=knn_k)
    methods["KNN"] = {
        **evaluate_all(X_true, X_knn, M_eval),
        "time_s": time.time() - t0
    }

    # --- MICE ---
    logger.info("Running MICE imputation...")
    t0 = time.time()
    try:
        X_mice = mice_imputation(X_chunk, M_chunk)
        methods["MICE"] = {
            **evaluate_all(X_true, X_mice, M_eval),
            "time_s": time.time() - t0
        }
    except Exception as e:
        logger.warning(f"MICE failed: {e}. Skipping.")
        methods["MICE"] = {"mae": float("nan"), "mre": float("nan"), "rmse": float("nan"),
                           "time_s": 0.0, "error": str(e)}

    # --- MF ---
    logger.info(f"Running Matrix Factorization imputation (K={mf_factors})...")
    t0 = time.time()
    X_mf = mf_imputation(X_chunk, M_chunk, num_factors=mf_factors)
    methods["MF"] = {
        **evaluate_all(X_true, X_mf, M_eval),
        "time_s": time.time() - t0
    }

    # --- FeaProp ---
    logger.info(f"Running FeaProp imputation ({fp_iter} iterations)...")
    t0 = time.time()
    X_fp = feature_propagation(X_chunk, M_chunk, adj=adj, num_iter=fp_iter)
    methods["FeaProp"] = {
        **evaluate_all(X_true, X_fp, M_eval),
        "time_s": time.time() - t0
    }

    logger.info(f"Baseline evaluation complete: {len(methods)} methods.")
    return methods


def run_mpin(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    X_true: np.ndarray,
    config: dict,
    M_eval: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run MPIN imputation and compute evaluation metrics.

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Input data with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask.
    X_true : np.ndarray, shape (N, D)
        Ground truth for evaluation.
    config : dict
        Configuration dictionary.
    M_eval : np.ndarray, optional
        Evaluation mask. Defaults to all missing positions.

    Returns
    -------
    result : dict
        {"mae": ..., "mre": ..., "rmse": ..., "time_s": ...}
    """
    if M_eval is None:
        M_eval = (1 - M_chunk).astype(np.float32)

    logger.info("Running MPIN imputation...")
    t0 = time.time()

    # Build sparse similarity graph (O(E) memory)
    mpin_cfg = config.get("mpin", {})
    edge_index, _ = build_knn_edge_index(
        X_chunk, M_chunk,
        k=mpin_cfg.get("K", 10),
        metric=mpin_cfg.get("graph_metric", "euclidean"),
        symmetric=mpin_cfg.get("symmetric", True)
    )

    # Train and impute
    model, X_imputed, train_history = train_mpin(X_chunk, M_chunk, edge_index, config)

    elapsed = time.time() - t0

    result = {
        **evaluate_all(X_true, X_imputed, M_eval),
        "time_s": elapsed
    }

    logger.info(
        f"MPIN complete: MAE={result['mae']:.6f}, MRE={result['mre']:.6f}, "
        f"time={elapsed:.1f}s"
    )
    return result, X_imputed, model


def print_comparison_table(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: list = ["mae", "mre", "rmse", "time_s"]
) -> None:
    """
    Print a formatted comparison table of all methods.

    Parameters
    ----------
    results_dict : dict
        {method_name: {metric_name: value, ...}}
    metrics : list
        Which metrics to show. Defaults to MAE, MRE, RMSE, time.

    Example Output:
    ---------------
    Method    │  MAE    │  MRE    │  RMSE   │  Time(s)
    ──────────┼─────────┼─────────┼─────────┼──────────
    MEAN      │  0.1234 │  0.0567 │  0.1890 │  0.01
    KNN       │  0.0987 │  0.0432 │  0.1456 │  2.34
    ...
    MPIN      │  0.0523 │  0.0210 │  0.0712 │  45.6
    """
    metric_labels = {
        "mae": "MAE↓",
        "mre": "MRE↓",
        "rmse": "RMSE↓",
        "time_s": "Time(s)"
    }
    col_width = 10
    method_width = 12

    # Header
    header = f"{'Method':<{method_width}}"
    for m in metrics:
        label = metric_labels.get(m, m)
        header += f"│ {label:^{col_width}}"
    print("\n" + "=" * (method_width + (col_width + 2) * len(metrics)))
    print(header)
    print("─" * method_width + ("┼" + "─" * (col_width + 2)) * len(metrics))

    # Find best (lowest) value for each metric (highlight)
    best = {}
    for m in metrics:
        vals = [v.get(m, float("nan")) for v in results_dict.values()
                if isinstance(v.get(m), (int, float))]
        best[m] = min(vals) if vals else float("nan")

    # Rows
    for method_name, result in results_dict.items():
        row = f"{method_name:<{method_width}}"
        for m in metrics:
            val = result.get(m, float("nan"))
            if isinstance(val, float) and not np.isnan(val):
                formatted = f"{val:.4f}"
                # Mark best with *
                if abs(val - best.get(m, float("nan"))) < 1e-9:
                    formatted = f"*{val:.4f}"
            else:
                formatted = "N/A"
            row += f"│ {formatted:^{col_width}}"
        print(row)

    print("=" * (method_width + (col_width + 2) * len(metrics)))
    print("* = best value for that metric\n")


def save_results(
    results_dict: Dict[str, Dict[str, Any]],
    path: str,
    format: str = "json"
) -> None:
    """
    Save comparison results to file.

    Parameters
    ----------
    results_dict : dict
        {method_name: {metric_name: value, ...}}
    path : str
        Output file path (e.g., "results/metrics/icu_results.json")
    format : str
        "json" or "csv".
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, default=float)
        logger.info(f"Results saved to {path} (JSON)")

    elif format == "csv":
        # Determine all metric columns
        all_metrics = set()
        for v in results_dict.values():
            all_metrics.update(v.keys())
        all_metrics = sorted(all_metrics)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["method"] + all_metrics)
            for method, result in results_dict.items():
                row = [method] + [result.get(m, "") for m in all_metrics]
                writer.writerow(row)
        logger.info(f"Results saved to {path} (CSV)")

    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


def load_results(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load previously saved results from JSON file.

    Parameters
    ----------
    path : str
        Path to saved JSON results file.

    Returns
    -------
    results_dict : dict
    """
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.info(f"Results loaded from {path}")
    return results
