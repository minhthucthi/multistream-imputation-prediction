"""
scripts/run_prediction.py
==========================
Main script to run the downstream prediction pipeline (Task 2).

Loads imputed data and trains a classifier for ICU mortality prediction.
Compares prediction accuracy across different imputation methods to show
that better imputation → better downstream task performance.

Usage:
------
    # Train logistic regression on MPIN-imputed data
    python scripts/run_prediction.py --dataset icu --imputation_method mpin \
        --predictor logistic

    # Train LSTM on MPIN-imputed data
    python scripts/run_prediction.py --dataset icu --imputation_method mpin \
        --predictor lstm

    # Compare prediction across all imputation methods
    python scripts/run_prediction.py --dataset icu --imputation_method all \
        --predictor logistic

    # With custom config
    python scripts/run_prediction.py --dataset icu --imputation_method mpin \
        --predictor logistic --config configs/config.yaml \
        --output results/metrics/prediction/
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
from src.models.baselines import mean_imputation, knn_imputation
from src.models.predictor import LogisticPredictor, LSTMPredictor, train_predictor, evaluate_predictor
from src.evaluation.compare import save_results, print_comparison_table
from src.visualization.plots import plot_comparison_bar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run downstream prediction pipeline (Task 2 – ICU mortality).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset", type=str, default="icu",
        choices=["icu"],
        help="Dataset for prediction. Currently only ICU has mortality labels."
    )
    parser.add_argument(
        "--imputation_method", type=str, default="mpin",
        choices=["mpin", "mean", "knn", "all"],
        help="Which imputation method to use for filling missing values before prediction."
    )
    parser.add_argument(
        "--predictor", type=str, default="logistic",
        choices=["logistic", "lstm"],
        help="Predictor model type. 'logistic' = Logistic Regression, 'lstm' = LSTM."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--output", type=str, default="results/metrics/prediction/",
        help="Output directory for prediction results."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to use as test set."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison plots."
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    return parser.parse_args()


def impute_data(
    X: np.ndarray,
    M: np.ndarray,
    method: str,
    config: dict
) -> np.ndarray:
    """
    Impute missing values using the specified method.

    Parameters
    ----------
    X : np.ndarray, shape (J, T, D)
        Data tensor with NaN at missing positions.
    M : np.ndarray, shape (J, T, D)
        Observation mask.
    method : str
        Imputation method: "mpin", "mean", "knn".
    config : dict
        Configuration dictionary.

    Returns
    -------
    X_imputed : np.ndarray, shape (J, T, D)
        Fully imputed data.
    """
    J, T, D = X.shape
    N = J * T
    X_flat = X.reshape(N, D)
    M_flat = M.reshape(N, D)

    logger = logging.getLogger("run_prediction")
    logger.info(f"Imputing with method: {method}")

    if method == "mean":
        X_imputed_flat = mean_imputation(X_flat, M_flat)

    elif method == "knn":
        k = config.get("baselines", {}).get("knn", {}).get("k", 5)
        X_imputed_flat = knn_imputation(X_flat, M_flat, k=k)

    elif method == "mpin":
        # Build graph and run MPIN
        adj, _ = build_similarity_graph(
            X_flat, M_flat,
            k=config.get("mpin", {}).get("K", 10)
        )
        _, X_imputed_flat, _ = train_mpin(X_flat, M_flat, adj, config)

    else:
        raise ValueError(f"Unknown imputation method: {method}")

    X_imputed = X_imputed_flat.reshape(J, T, D)
    logger.info(f"Imputation complete: shape={X_imputed.shape}")
    return X_imputed.astype(np.float32)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42
):
    """
    Split data into train/test sets (stratified by label).

    Parameters
    ----------
    X : np.ndarray, shape (J, T, D) or (J, D)
    y : np.ndarray, shape (J,)
    test_size : float
    seed : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ImportError:
        # Fallback: simple split
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        n_test = int(len(y) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()

    # Setup
    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, "run_prediction.log")
    logger = create_logger("run_prediction", log_file=log_file, level=args.log_level)

    logger.info("=" * 60)
    logger.info("MPIN Downstream Prediction Pipeline (Task 2)")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Imputation: {args.imputation_method}")
    logger.info(f"Predictor: {args.predictor}")
    logger.info("=" * 60)

    config = load_config(args.config)
    config["general"]["seed"] = args.seed
    set_seed(args.seed)

    dataset_cfg = config.get(args.dataset, {})

    # -------------------------------------------------------
    # Step 1: Load Dataset
    # -------------------------------------------------------
    logger.info("Step 1: Loading ICU dataset...")
    data_loader = DataLoader(config)

    raw_path = dataset_cfg.get("raw_data_dir", "data/raw/icu/")
    X, M, y = data_loader.load_icu_dataset(raw_path)

    if y is None:
        raise ValueError("ICU labels (y) not found. Cannot run prediction task.")

    logger.info(f"X: {X.shape}, M: {M.shape}, y: {y.shape}")
    logger.info(f"Mortality rate: {y.mean():.3f} ({int(y.sum())} / {len(y)})")

    # Normalize
    X_norm, _ = data_loader.normalize(X, M, method=dataset_cfg.get("normalization", "minmax"))

    # -------------------------------------------------------
    # Step 2: Determine which imputation methods to run
    # -------------------------------------------------------
    if args.imputation_method == "all":
        imputation_methods = ["mean", "knn", "mpin"]
    else:
        imputation_methods = [args.imputation_method]

    all_pred_results = {}

    for imp_method in imputation_methods:
        logger.info(f"\n{'='*40}")
        logger.info(f"Imputation method: {imp_method.upper()}")
        logger.info(f"{'='*40}")

        # -------------------------------------------------------
        # Step 3: Impute missing values
        # -------------------------------------------------------
        logger.info("Step 3: Imputing missing values...")

        # Cache path to avoid re-imputation
        cache_path = os.path.join(
            args.output, f"X_imputed_{args.dataset}_{imp_method}.npy"
        )

        if os.path.exists(cache_path):
            logger.info(f"Loading cached imputed data from {cache_path}")
            X_imputed = np.load(cache_path)
        else:
            X_imputed = impute_data(X_norm, M, method=imp_method, config=config)
            np.save(cache_path, X_imputed)
            logger.info(f"Imputed data cached to {cache_path}")

        # -------------------------------------------------------
        # Step 4: Train/test split
        # -------------------------------------------------------
        logger.info("Step 4: Splitting data into train/test...")
        X_train, X_test, y_train, y_test = split_data(
            X_imputed, y, test_size=args.test_size, seed=args.seed
        )
        logger.info(
            f"Train: {X_train.shape[0]} samples, "
            f"Test: {X_test.shape[0]} samples"
        )

        # -------------------------------------------------------
        # Step 5: Train predictor
        # -------------------------------------------------------
        logger.info(f"Step 5: Training {args.predictor} predictor...")
        model = train_predictor(X_train, y_train, config, predictor_type=args.predictor)

        # -------------------------------------------------------
        # Step 6: Evaluate predictor
        # -------------------------------------------------------
        logger.info("Step 6: Evaluating predictor on test set...")
        metrics = evaluate_predictor(model, X_test, y_test, predictor_type=args.predictor)

        all_pred_results[imp_method.upper()] = metrics

        logger.info(
            f"[{imp_method.upper()}] "
            f"Accuracy={metrics.get('accuracy', 'N/A'):.4f}, "
            f"AUC={metrics.get('auc_roc', 'N/A'):.4f}, "
            f"F1={metrics.get('f1', 'N/A'):.4f}"
        )

    # -------------------------------------------------------
    # Step 7: Print and save results
    # -------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS SUMMARY")
    logger.info("=" * 60)
    print_comparison_table(all_pred_results, metrics=["accuracy", "auc_roc", "f1"])

    out_path = os.path.join(
        args.output,
        f"{args.dataset}_{args.imputation_method}_{args.predictor}_results"
    )
    save_results(all_pred_results, out_path + ".json", format="json")
    save_results(all_pred_results, out_path + ".csv", format="csv")
    logger.info(f"Results saved to: {out_path}.json")

    # -------------------------------------------------------
    # Step 8: Plots (optional)
    # -------------------------------------------------------
    if args.plot and len(all_pred_results) > 1:
        plot_dir = os.path.join("results", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for metric in ["accuracy", "auc_roc", "f1"]:
            plot_comparison_bar(
                all_pred_results,
                metric=metric,
                title=f"ICU Mortality Prediction – {metric.upper()} by Imputation Method",
                save_path=os.path.join(plot_dir, f"prediction_{metric}_comparison.png"),
                show=False
            )
        logger.info(f"Plots saved to {plot_dir}/")

    logger.info("Prediction pipeline complete!")
    return all_pred_results


if __name__ == "__main__":
    main()
