"""
src/models/baselines.py
========================
Baseline imputation methods for comparison with MPIN.

Implements the following baselines from the MPIN paper:
- MEAN: Global mean imputation
- KNN: K-Nearest Neighbor imputation
- FeaProp (FP): Feature Propagation on similarity graph

External libraries needed for full baselines:
- MICE: sklearn.impute.IterativeImputer
- MF: sklearn.decomposition.NMF or custom ALS
- BRITS: Available at https://github.com/caow13/BRITS
- SAITS: pip install pyPOTS

Usage:
------
    X_imputed = mean_imputation(X_chunk, M_chunk)
    X_imputed = knn_imputation(X_chunk, M_chunk, k=5)
    X_imputed = feature_propagation(X_chunk, M_chunk, adj, num_iter=10)
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ===========================================================================
# MEAN Imputation
# ===========================================================================

def mean_imputation(
    X: np.ndarray,
    M: np.ndarray
) -> np.ndarray:
    """
    Global mean imputation: replace missing values with per-attribute mean.

    For each attribute d, compute the mean over all observed values across
    all instances, and use that as the fill value for missing positions.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data chunk. NaN where values are missing.
    M : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).

    Returns
    -------
    X_imputed : np.ndarray, shape (N, D)
        Data with missing values replaced by column-wise mean.
        No NaN values remain.

    Notes
    -----
    This is the simplest baseline and serves as the lower bound.
    It ignores temporal structure, cross-instance correlations, and
    produces a uniform fill that reduces data variance.
    """
    X_imputed = X.copy()
    N, D = X.shape

    for d in range(D):
        obs_mask = M[:, d].astype(bool)
        if obs_mask.sum() > 0:
            col_mean = np.nanmean(X[obs_mask, d])
        else:
            col_mean = 0.0
            logger.warning(f"Attribute {d} has no observed values. Using 0.0 as fill.")

        # Fill missing positions
        missing_mask = ~obs_mask
        X_imputed[missing_mask, d] = col_mean

    logger.info(f"Mean imputation complete. Filled {(M == 0).sum()} missing values.")
    return X_imputed


# ===========================================================================
# KNN Imputation
# ===========================================================================

def knn_imputation(
    X: np.ndarray,
    M: np.ndarray,
    k: int = 5
) -> np.ndarray:
    """
    K-Nearest Neighbor imputation.

    For each instance i with missing attribute d:
    1. Find K instances that have attribute d observed.
    2. Among those, find K nearest neighbors based on other shared observed attributes.
    3. Fill x_{i,d} with the weighted mean of neighbor values (weights ∝ 1/distance).

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data chunk. NaN where values are missing.
    M : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    k : int, default 5
        Number of nearest neighbors to use.

    Returns
    -------
    X_imputed : np.ndarray, shape (N, D)
        Data with missing values filled by KNN.

    Notes
    -----
    Distance is computed using only the attributes that are observed in BOTH
    the target instance and the neighbor (shared observed attributes).
    If no shared attributes exist, falls back to global mean.
    """
    # First do mean-fill as fallback
    X_imputed = mean_imputation(X, M)
    N, D = X.shape

    # For each instance i that has at least one missing attribute
    missing_instances = np.where(M.sum(axis=1) < D)[0]  # Instances with ≥1 missing

    for i in missing_instances:
        missing_attrs = np.where(M[i] == 0)[0]  # Attributes missing in instance i

        for d in missing_attrs:
            # Candidates: instances that have attribute d observed
            candidates = np.where(M[:, d] == 1)[0]
            candidates = candidates[candidates != i]  # Exclude self

            if len(candidates) == 0:
                continue  # Keep mean-fill value

            # Shared observed attributes between i and candidates
            # (attributes where both i and candidate have values)
            shared_obs = M[i] == 1  # Attributes observed in i (shape D,)

            # Compute distance on shared attributes only
            distances = []
            for c in candidates:
                # Shared attributes: obs in both i and candidate c
                shared = shared_obs & (M[c] == 1)  # shape (D,)
                if shared.sum() == 0:
                    distances.append(np.inf)
                else:
                    diff = X[i, shared] - X[c, shared]
                    distances.append(np.sqrt((diff ** 2).sum()))

            distances = np.array(distances)

            # Take K nearest (smallest distance)
            if len(distances) <= k:
                nn_idx = np.arange(len(candidates))
            else:
                nn_idx = np.argsort(distances)[:k]

            nn_distances = distances[nn_idx]
            nn_values = X[candidates[nn_idx], d]

            # Inverse distance weighting
            if np.any(nn_distances == 0):
                # Exact match: use that value
                exact = nn_values[nn_distances == 0]
                X_imputed[i, d] = exact[0]
            else:
                weights = 1.0 / (nn_distances + 1e-8)
                weights /= weights.sum()
                X_imputed[i, d] = (weights * nn_values).sum()

    logger.info(f"KNN imputation (k={k}) complete.")
    return X_imputed


# ===========================================================================
# Feature Propagation (FeaProp / FP)
# ===========================================================================

def feature_propagation(
    X: np.ndarray,
    M: np.ndarray,
    adj: Optional[np.ndarray] = None,
    num_iter: int = 10,
    alpha: float = 0.8
) -> np.ndarray:
    """
    Feature Propagation (FeaProp) imputation on a similarity graph.

    Propagates observed features through graph neighbors to fill missing values.
    Iterative process: missing values are updated by averaging neighbor values,
    while observed values are kept fixed (bound condition, same as MPIN).

    Algorithm:
        X^{(0)} = mean-fill(X)  (initialization)
        For l = 1, ..., num_iter:
            X^{(l)} = alpha * Adj_norm @ X^{(l-1)} + (1-alpha) * X^{(0)}
            X^{(l)} = M * X_observed + (1-M) * X^{(l)}  (bound condition)

    This is a simplified diffusion-based approach. The original FeaProp paper:
    "On the Unreasonable Effectiveness of Feature Propagation in Learning
    on Graphs with Missing Node Features" (Rossi et al., 2022).

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data chunk. NaN where values are missing.
    M : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    adj : np.ndarray, shape (N, N), optional
        Pre-built adjacency matrix. If None, uses a simple identity graph
        (no propagation, equivalent to mean-fill).
    num_iter : int, default 10
        Number of propagation iterations.
    alpha : float, default 0.8
        Propagation coefficient. Higher = more weight on neighbors.
        alpha=0 → no propagation (mean-fill). alpha=1 → full propagation.

    Returns
    -------
    X_imputed : np.ndarray, shape (N, D)
        Imputed data after feature propagation.
    """
    N, D = X.shape

    # Initialization: mean-fill
    X_obs = X.copy()
    for d in range(D):
        obs = M[:, d].astype(bool)
        fill_val = np.nanmean(X_obs[obs, d]) if obs.sum() > 0 else 0.0
        X_obs[~obs, d] = fill_val
        X_obs[np.isnan(X_obs[:, d]), d] = fill_val

    X_current = X_obs.copy()
    X_original_obs = X_obs.copy()  # Keep original mean-fill for residual term

    if adj is None:
        logger.warning("No adjacency matrix provided. FeaProp with identity graph = mean-fill.")
        return X_obs

    # Normalize adjacency: D^{-1} * A (row normalization for mean aggregation)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
    adj_norm = adj / row_sums  # Row-normalized: each row sums to 1

    M_float = M.astype(np.float32)

    for iteration in range(num_iter):
        # Propagation: aggregate neighbor features
        X_propagated = alpha * (adj_norm @ X_current) + (1 - alpha) * X_original_obs

        # Bound condition: keep observed values, only update missing
        X_current = M_float * X_original_obs + (1 - M_float) * X_propagated

        logger.debug(f"FeaProp iteration {iteration+1}/{num_iter} complete.")

    logger.info(f"Feature Propagation (FeaProp) complete: {num_iter} iterations.")
    return X_current


# ===========================================================================
# MICE Imputation (wrapper around scikit-learn)
# ===========================================================================

def mice_imputation(
    X: np.ndarray,
    M: np.ndarray,
    max_iter: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """
    MICE (Multiple Imputation by Chained Equations) using scikit-learn.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data with NaN at missing positions.
    M : np.ndarray, shape (N, D)
        Observation mask.
    max_iter : int
        Maximum iterations for IterativeImputer.
    random_state : int

    Returns
    -------
    X_imputed : np.ndarray, shape (N, D)
    """
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
    except ImportError:
        logger.error("scikit-learn required for MICE. Install: pip install scikit-learn")
        return mean_imputation(X, M)

    # Set masked positions to NaN (IterativeImputer expects NaN for missing)
    X_nan = X.copy()
    X_nan[M == 0] = np.nan

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    X_imputed = imputer.fit_transform(X_nan)

    logger.info(f"MICE imputation complete (max_iter={max_iter}).")
    return X_imputed.astype(np.float32)


# ===========================================================================
# Matrix Factorization Imputation (simple ALS)
# ===========================================================================

def mf_imputation(
    X: np.ndarray,
    M: np.ndarray,
    num_factors: int = 20,
    max_iter: int = 100,
    lr: float = 0.01,
    reg: float = 0.01
) -> np.ndarray:
    """
    Matrix Factorization imputation using Alternating Least Squares (ALS).

    Factorizes X ≈ U @ V^T where U ∈ R^{N×K}, V ∈ R^{D×K}.
    Only observed positions contribute to the loss.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Data with NaN at missing positions.
    M : np.ndarray, shape (N, D)
        Observation mask.
    num_factors : int
        Rank K of the factorization.
    max_iter : int
        Number of training iterations.
    lr : float
        Learning rate for SGD updates.
    reg : float
        L2 regularization coefficient.

    Returns
    -------
    X_imputed : np.ndarray, shape (N, D)
    """
    N, D = X.shape
    K = num_factors

    # Initialize factor matrices
    rng = np.random.RandomState(42)
    U = rng.randn(N, K) * 0.01
    V = rng.randn(D, K) * 0.01

    # Observed positions
    obs_i, obs_d = np.where(M == 1)
    obs_values = X[obs_i, obs_d]

    best_loss = float("inf")
    best_U, best_V = U.copy(), V.copy()

    for it in range(max_iter):
        # Compute predictions at observed positions
        pred = (U @ V.T)[obs_i, obs_d]
        error = obs_values - pred

        # Gradient updates (SGD)
        for idx in range(len(obs_i)):
            i, d, e = obs_i[idx], obs_d[idx], error[idx]
            U[i] += lr * (e * V[d] - reg * U[i])
            V[d] += lr * (e * U[i] - reg * V[d])

        # Loss
        pred_all = (U @ V.T)[obs_i, obs_d]
        loss = np.mean((obs_values - pred_all) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V = U.copy(), V.copy()

        if it % 20 == 0:
            logger.debug(f"MF iter {it}/{max_iter}, loss={loss:.6f}")

    X_imputed = best_U @ best_V.T
    # Keep observed values unchanged
    X_imputed[M == 1] = X[M == 1]

    logger.info(f"MF imputation complete: {max_iter} iters, best_loss={best_loss:.6f}")
    return X_imputed.astype(np.float32)
