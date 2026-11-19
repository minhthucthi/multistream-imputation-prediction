"""
src/evaluation/metrics.py
==========================
Evaluation metrics for missing value imputation.

All metrics are computed ONLY at positions where values were originally missing
(M_missing == 1), not at observed positions. This ensures we measure imputation
accuracy rather than reconstruction of already-known values.

Metrics:
- MAE  (Mean Absolute Error)
- MRE  (Mean Relative Error)
- RMSE (Root Mean Squared Error)

Usage:
------
    from src.evaluation.metrics import evaluate_all
    results = evaluate_all(X_true, X_imputed, M_missing)
    print(results)  # {'mae': 0.123, 'mre': 0.045, 'rmse': 0.167}
"""

import numpy as np
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def mae(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M_missing: np.ndarray
) -> float:
    """
    Mean Absolute Error on missing positions only.

    MAE = (1/|M|) * Σ_{(i,d): M_missing[i,d]=1} |X_imputed[i,d] - X_true[i,d]|

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
        Ground truth values (complete data, before artificially masking).
    X_imputed : np.ndarray, shape (N, D)
        Imputed data (output of imputation method).
    M_missing : np.ndarray, shape (N, D)
        Binary mask: 1 at positions that WERE missing (now filled by imputer).
        0 at positions that were originally observed (not evaluated).

    Returns
    -------
    mae_val : float
        Mean Absolute Error. Lower is better.

    Notes
    -----
    M_missing is NOT the same as M_observed:
        M_observed[i,d] = 1 → observed (not evaluated here)
        M_missing[i,d]  = 1 → missing (evaluated here)
        M_missing = 1 - M_observed

    If no positions are missing (all M_missing=0), returns 0.0.
    """
    missing_mask = M_missing.astype(bool)
    num_missing = missing_mask.sum()

    if num_missing == 0:
        logger.warning("MAE: No missing positions found in M_missing. Returning 0.0.")
        return 0.0

    errors = np.abs(X_imputed[missing_mask] - X_true[missing_mask])
    mae_val = errors.mean()

    return float(mae_val)


def mre(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M_missing: np.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Mean Relative Error on missing positions only.

    MRE = Σ_{(i,d): M_missing[i,d]=1} |X_imputed[i,d] - X_true[i,d]| / (Σ_{(i,d): M_missing[i,d]=1} |X_true[i,d]| + ε)

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
        Ground truth values.
    X_imputed : np.ndarray, shape (N, D)
        Imputed values.
    M_missing : np.ndarray, shape (N, D)
        1 at positions that were missing.
    epsilon : float, default 1e-6
        Small constant added to denominator to avoid division by zero.

    Returns
    -------
    mre_val : float
        Mean Relative Error. Lower is better.

    Notes
    -----
    Using the sum-ratio definition (standard across BRITS/SAITS/MPIN) prevents
    division-by-zero blowups when normalized ground truth entries equal zero.
    """
    missing_mask = M_missing.astype(bool)
    num_missing = missing_mask.sum()

    if num_missing == 0:
        logger.warning("MRE: No missing positions found. Returning 0.0.")
        return 0.0

    true_vals = X_true[missing_mask]
    pred_vals = X_imputed[missing_mask]

    mre_val = np.abs(pred_vals - true_vals).sum() / (np.abs(true_vals).sum() + epsilon)

    return float(mre_val)


def rmse(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M_missing: np.ndarray
) -> float:
    """
    Root Mean Squared Error on missing positions only.

    RMSE = sqrt( (1/|M|) * Σ_{(i,d): M_missing[i,d]=1} (X_imputed[i,d] - X_true[i,d])^2 )

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
        Ground truth values.
    X_imputed : np.ndarray, shape (N, D)
        Imputed values.
    M_missing : np.ndarray, shape (N, D)
        1 at positions that were missing.

    Returns
    -------
    rmse_val : float
        Root Mean Squared Error. Lower is better.

    Notes
    -----
    RMSE penalizes large errors more heavily than MAE due to squaring.
    """
    missing_mask = M_missing.astype(bool)
    num_missing = missing_mask.sum()

    if num_missing == 0:
        logger.warning("RMSE: No missing positions found. Returning 0.0.")
        return 0.0

    squared_errors = (X_imputed[missing_mask] - X_true[missing_mask]) ** 2
    rmse_val = np.sqrt(squared_errors.mean())

    return float(rmse_val)


def per_attribute_mae(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M_missing: np.ndarray
) -> np.ndarray:
    """
    Compute MAE separately for each attribute d.

    Useful for understanding which attributes are harder to impute.

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
    X_imputed : np.ndarray, shape (N, D)
    M_missing : np.ndarray, shape (N, D)

    Returns
    -------
    mae_per_attr : np.ndarray, shape (D,)
        MAE for each attribute. NaN if attribute has no missing values.
    """
    N, D = X_true.shape
    mae_per_attr = np.full(D, np.nan)

    for d in range(D):
        missing_d = M_missing[:, d].astype(bool)
        if missing_d.sum() > 0:
            errors = np.abs(X_imputed[missing_d, d] - X_true[missing_d, d])
            mae_per_attr[d] = errors.mean()

    return mae_per_attr


def evaluate_all(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M_missing: np.ndarray,
    epsilon: float = 1e-6
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in one call.

    Parameters
    ----------
    X_true : np.ndarray, shape (N, D)
        Ground truth complete data.
    X_imputed : np.ndarray, shape (N, D)
        Imputed data from any method.
    M_missing : np.ndarray, shape (N, D)
        1 at positions that were missing (to be evaluated).
    epsilon : float
        Small constant for MRE stability.

    Returns
    -------
    metrics : dict
        Dictionary with keys: 'mae', 'mre', 'rmse', 'num_missing'.
        {
            'mae': float,
            'mre': float,
            'rmse': float,
            'num_missing': int,   # Total missing positions evaluated
            'missing_rate': float # Fraction of positions that were missing
        }

    Example
    -------
    >>> results = evaluate_all(X_true, X_imputed, M_missing)
    >>> print(f"MAE={results['mae']:.4f}, MRE={results['mre']:.4f}")
    """
    num_missing = int(M_missing.sum())
    total = M_missing.size
    missing_rate = num_missing / total if total > 0 else 0.0

    mae_val = mae(X_true, X_imputed, M_missing)
    mre_val = mre(X_true, X_imputed, M_missing, epsilon=epsilon)
    rmse_val = rmse(X_true, X_imputed, M_missing)

    metrics = {
        "mae": mae_val,
        "mre": mre_val,
        "rmse": rmse_val,
        "num_missing": num_missing,
        "missing_rate": missing_rate
    }

    logger.info(
        f"Metrics: MAE={mae_val:.6f}, MRE={mre_val:.6f}, RMSE={rmse_val:.6f} "
        f"(evaluated on {num_missing} missing positions, rate={100*missing_rate:.1f}%)"
    )

    return metrics


def create_evaluation_mask(
    M_observed: np.ndarray,
    X_true: np.ndarray,
    artificial_missing_rate: float = 0.1,
    seed: int = 42
) -> tuple:
    """
    Create artificial missing values for evaluation.

    For methods that require ground truth at missing positions (all methods do),
    this function takes a subset of observed values, hides them (pretends they
    are missing), runs imputation, and measures error at those positions.

    Parameters
    ----------
    M_observed : np.ndarray, shape (N, D)
        Original observation mask (1=observed, 0=missing).
    X_true : np.ndarray, shape (N, D)
        Ground truth data.
    artificial_missing_rate : float
        Fraction of observed values to artificially hide for evaluation.
    seed : int
        Random seed.

    Returns
    -------
    X_eval : np.ndarray
        X_true with artificial missing positions set to NaN.
    M_eval : np.ndarray
        New observation mask (original missing + artificial missing = 0).
    M_artificial : np.ndarray
        Mask of ONLY artificially created missing positions (1 = eval position).
    """
    rng = np.random.RandomState(seed)

    obs_positions = np.argwhere(M_observed == 1)  # (num_obs, 2)
    num_hide = int(len(obs_positions) * artificial_missing_rate)
    hide_idx = rng.choice(len(obs_positions), size=num_hide, replace=False)
    hide_positions = obs_positions[hide_idx]  # (num_hide, 2)

    X_eval = X_true.copy()
    M_eval = M_observed.copy()
    M_artificial = np.zeros_like(M_observed)

    for pos in hide_positions:
        i, d = pos[0], pos[1]
        X_eval[i, d] = np.nan
        M_eval[i, d] = 0
        M_artificial[i, d] = 1

    logger.info(
        f"Created artificial evaluation mask: {num_hide} positions hidden "
        f"({100*artificial_missing_rate:.0f}% of {len(obs_positions)} observed)."
    )
    return X_eval, M_eval, M_artificial
