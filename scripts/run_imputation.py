"""
scripts/run_imputation.py
==========================
Main script to run the full imputation pipeline.

Supports running MPIN or any baseline method on ICU or Airquality dataset.

Usage:
------
    # Run MPIN on ICU dataset
    python scripts/run_imputation.py --dataset icu --method mpin

    # Run all methods on Airquality
    python scripts/run_imputation.py --dataset airquality --method all

    # Run specific baseline with custom config
    python scripts/run_imputation.py --dataset icu --method knn \
        --config configs/config.yaml --output results/metrics/

    # Run with custom hyperparameters
    python scripts/run_imputation.py --dataset icu --method mpin \
        --epochs 200 --K 15
"""

import os
import sys
import argparse
import numpy as np
import logging
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import load_config, set_seed, create_logger
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.similarity_graph import build_similarity_graph
from src.models.mpin import train_mpin
from src.models.baselines import (
    mean_imputation, knn_imputation, feature_propagation,
    mice_imputation, mf_imputation
)
from src.evaluation.metrics import evaluate_all, create_evaluation_mask
from src.evaluation.compare import (
    run_all_baselines, run_mpin, print_comparison_table, save_results
)
from src.visualization.plots import (
    plot_missing_pattern, plot_comparison_bar,
    plot_imputed_vs_true, plot_training_loss
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run imputation pipeline for MPIN project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["icu", "airquality"],
        help="Dataset to use: 'icu' (PhysioNet 2012) or 'airquality'"
    )
    parser.add_argument(
        "--method", type=str, default="mpin",
        choices=["mpin", "mpin_dm", "mean", "knn", "mice", "mf", "feaprop", "all"],
        help="Imputation method to run. 'all' runs all methods for comparison."
    )

    # Config
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--output", type=str, default="results/metrics/",
        help="Output directory for results."
    )

    # Overrides (override config values)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override mpin.epochs in config.")
    parser.add_argument("--K", type=int, default=None,
                        help="Override mpin.K (KNN graph neighbors) in config.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override mpin.lr in config.")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Override dataset window_size in config.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # Evaluation
    parser.add_argument("--eval_missing_rate", type=float, default=0.1,
                        help="Fraction of observed values to hide for evaluation.")
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Max number of windows to process (for quick testing).")
    parser.add_argument("--max_streams", type=int, default=None,
                        help="Max number of streams/patients to load (to limit memory usage).")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Generate synthetic dataset for quick testing without raw files.")

    # Visualization
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots.")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively.")

    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Apply command-line argument overrides to config."""
    if args.epochs is not None:
        config.setdefault("mpin", {})["epochs"] = args.epochs
    if args.K is not None:
        config.setdefault("mpin", {})["K"] = args.K
    if args.lr is not None:
        config.setdefault("mpin", {})["lr"] = args.lr
    if args.window_size is not None:
        config.setdefault(args.dataset, {})["window_size"] = args.window_size
    if args.max_streams is not None:
        config.setdefault(args.dataset, {})["max_streams"] = args.max_streams
    config.setdefault("general", {})["seed"] = args.seed
    return config


def main():
    args = parse_args()

    # Setup logging
    log_file = os.path.join(args.output, "run_imputation.log")
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger("run_imputation", log_file=log_file, level=args.log_level)

    logger.info("=" * 60)
    logger.info("MPIN Imputation Pipeline")
    logger.info(f"Dataset: {args.dataset}, Method: {args.method}")
    logger.info("=" * 60)

    # Load and apply config
    config = load_config(args.config)
    config = apply_overrides(config, args)
    set_seed(config["general"].get("seed", 42))

    dataset_cfg = config.get(args.dataset, {})

    # -------------------------------------------------------
    # Step 1: Load Dataset
    # -------------------------------------------------------
    logger.info("Step 1: Loading dataset...")
    data_loader = DataLoader(config)

    if getattr(args, "synthetic", False):
        logger.info("Using synthetic dataset for testing (--synthetic enabled).")
        D = 37 if args.dataset == "icu" else 11
        X, M, y = data_loader.load_synthetic_dataset(num_streams=10, num_timesteps=48, num_attributes=D, sparsity=0.8)
    elif args.dataset == "icu":
        raw_path = dataset_cfg.get("raw_data_dir", "data/raw/icu/")
        X, M, y = data_loader.load_icu_dataset(raw_path)
    elif args.dataset == "airquality":
        raw_path = dataset_cfg.get("raw_data_dir", "data/raw/airquality/")
        X, M = data_loader.load_airquality_dataset(raw_path)
        y = None
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loaded data: X.shape={X.shape}, M.shape={M.shape}")
    if y is not None:
        logger.info(f"Labels: y.shape={y.shape}, pos_rate={y.mean():.3f}")

    # -------------------------------------------------------
    # Step 2: Normalize
    # -------------------------------------------------------
    logger.info("Step 2: Normalizing data...")
    norm_method = dataset_cfg.get("normalization", "minmax")
    X_norm, norm_stats = data_loader.normalize(X, M, method=norm_method)

    # -------------------------------------------------------
    # Step 3: Create tumbling windows
    # -------------------------------------------------------
    logger.info("Step 3: Creating tumbling windows...")
    window_size = dataset_cfg.get("window_size", 12)
    windows = data_loader.create_tumbling_windows(X_norm, M, window_size=window_size)
    logger.info(f"Total windows: {len(windows)}")

    if args.max_windows is not None:
        windows = windows[:args.max_windows]
        logger.info(f"Using first {args.max_windows} windows (--max_windows).")

    # -------------------------------------------------------
    # Step 4: Preprocessing (flatten windows)
    # -------------------------------------------------------
    preprocessor = Preprocessor(config)

    # -------------------------------------------------------
    # Step 5: Run imputation on each window
    # -------------------------------------------------------
    logger.info(f"Step 5: Running imputation: method={args.method}")

    all_results = {}
    all_imputed_chunks = []

    for window_idx, (X_win, M_win) in enumerate(windows):
        logger.info(f"--- Window {window_idx + 1}/{len(windows)} ---")

        # Flatten window to chunk
        X_chunk, M_chunk = preprocessor.create_data_chunk(X_win, M_win)

        # Create artificial evaluation mask
        X_eval, M_eval_obs, M_eval_missing = create_evaluation_mask(
            M_chunk, X_chunk,
            artificial_missing_rate=args.eval_missing_rate
        )

        # Ground truth for evaluation
        X_true = X_chunk.copy()

        # Build adjacency for dense-graph baseline (FeaProp); MPIN builds sparse edge_index internally
        if args.method in ["feaprop", "all"]:
            adj, _ = build_similarity_graph(
                X_eval, M_eval_obs,
                k=config.get("mpin", {}).get("K", 10)
            )
        else:
            adj = None

        # Run selected method(s)
        if args.method == "all":
            logger.info("Running all methods...")
            baseline_results = run_all_baselines(
                X_eval, M_eval_obs, X_true, config,
                M_eval=M_eval_missing, adj=adj
            )
            mpin_result, X_imputed, model = run_mpin(
                X_eval, M_eval_obs, X_true, config,
                M_eval=M_eval_missing
            )
            window_results = {**baseline_results, "MPIN": mpin_result}
            all_imputed_chunks.append(X_imputed)

        elif args.method == "mpin":
            mpin_result, X_imputed, model = run_mpin(
                X_eval, M_eval_obs, X_true, config,
                M_eval=M_eval_missing
            )
            window_results = {"MPIN": mpin_result}
            all_imputed_chunks.append(X_imputed)

        elif args.method == "mean":
            X_imp = mean_imputation(X_eval, M_eval_obs)
            window_results = {"MEAN": evaluate_all(X_true, X_imp, M_eval_missing)}

        elif args.method == "knn":
            k = config.get("baselines", {}).get("knn", {}).get("k", 5)
            X_imp = knn_imputation(X_eval, M_eval_obs, k=k)
            window_results = {"KNN": evaluate_all(X_true, X_imp, M_eval_missing)}

        elif args.method == "mice":
            X_imp = mice_imputation(X_eval, M_eval_obs)
            window_results = {"MICE": evaluate_all(X_true, X_imp, M_eval_missing)}

        elif args.method == "mf":
            X_imp = mf_imputation(X_eval, M_eval_obs)
            window_results = {"MF": evaluate_all(X_true, X_imp, M_eval_missing)}

        elif args.method == "feaprop":
            X_imp = feature_propagation(X_eval, M_eval_obs, adj=adj)
            window_results = {"FeaProp": evaluate_all(X_true, X_imp, M_eval_missing)}

        # Accumulate results
        for method_name, metrics in window_results.items():
            if method_name not in all_results:
                all_results[method_name] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_results[method_name][k].append(v)

    # -------------------------------------------------------
    # Step 6: Aggregate results across windows
    # -------------------------------------------------------
    logger.info("Step 6: Aggregating results across windows...")
    final_results = {}
    for method_name, metrics_lists in all_results.items():
        final_results[method_name] = {}
        for metric_name, values in metrics_lists.items():
            if values:
                final_results[method_name][metric_name] = float(np.mean(values))

    # -------------------------------------------------------
    # Step 7: Print and save results
    # -------------------------------------------------------
    logger.info("Step 7: Results summary")
    print_comparison_table(final_results)

    # Save to JSON and CSV
    out_prefix = os.path.join(args.output, f"{args.dataset}_{args.method}")
    save_results(final_results, out_prefix + "_results.json", format="json")
    save_results(final_results, out_prefix + "_results.csv", format="csv")

    # -------------------------------------------------------
    # Step 8: Plots (optional)
    # -------------------------------------------------------
    if args.plot and windows:
        logger.info("Step 8: Generating plots...")
        plot_dir = os.path.join("results", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Missing pattern of first window
        X_win0, M_win0 = windows[0]
        X_chunk0, M_chunk0 = preprocessor.create_data_chunk(X_win0, M_win0)
        plot_missing_pattern(
            M_chunk0,
            title=f"{args.dataset.upper()} Missing Pattern (Window 1)",
            save_path=os.path.join(plot_dir, f"{args.dataset}_missing_pattern.png"),
            show=args.show_plots
        )

        if args.method in ["mpin", "all"] and all_imputed_chunks:
            X_imputed0 = all_imputed_chunks[0]
            X_true0, _, M_eval0 = create_evaluation_mask(M_chunk0, X_chunk0, 0.1)
            plot_imputed_vs_true(
                X_true=X_chunk0, X_imputed=X_imputed0, M=M_eval0,
                feature_idx=0, stream_idx=0,
                title=f"MPIN Imputation ({args.dataset.upper()}) – Attribute 0",
                save_path=os.path.join(plot_dir, f"{args.dataset}_mpin_imputed_attr0.png"),
                show=args.show_plots
            )

        # Comparison bar chart
        if len(final_results) > 1:
            for metric in ["mae", "mre"]:
                plot_comparison_bar(
                    final_results, metric=metric,
                    title=f"{args.dataset.upper()} – {metric.upper()} Comparison",
                    save_path=os.path.join(plot_dir, f"{args.dataset}_{metric}_comparison.png"),
                    show=args.show_plots
                )

    logger.info("Pipeline complete!")
    logger.info(f"Results saved to: {args.output}")
    return final_results


if __name__ == "__main__":
    main()
