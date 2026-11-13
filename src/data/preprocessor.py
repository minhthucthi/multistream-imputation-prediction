"""
src/data/preprocessor.py
========================
Preprocessing utilities for MPIN project.

Handles:
- Flattening windowed data into data chunks (J*T instances x D attributes)
- Computing instance importance scores (phi) for continuous framework
- Filtering instances by importance threshold eta
- Creating artificial missing patterns for validation

Usage:
    preprocessor = Preprocessor(config)
    X_chunk, M_chunk = preprocessor.create_data_chunk(X_window, M_window)
    phi = preprocessor.compute_importance_score(X_chunk, M_chunk)
    X_imp, M_imp = preprocessor.filter_by_importance(X_chunk, M_chunk, phi, eta=0.6)
"""

import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Data preprocessing class for MPIN pipeline.

    Converts windowed data into chunk format suitable for MPIN's
    graph-based imputation, and handles the data-update step of MPIN-DM.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("Preprocessor initialized.")

    def create_data_chunk(
        self,
        X_window: np.ndarray,
        M_window: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a windowed data tensor into a flat 2D data chunk.

        MPIN operates on a chunk where each row is one "instance" (j, t):
        a single stream at a single timestep, with D attribute values.

        Parameters
        ----------
        X_window : np.ndarray, shape (J, T, D)
            Data tensor for one tumbling window.
            NaN where values are missing.
        M_window : np.ndarray, shape (J, T, D)
            Observation mask (1=observed, 0=missing).

        Returns
        -------
        X_chunk : np.ndarray, shape (N, D)  where N = J * T
            Flattened data matrix. Row i = instance (j, t).
            Index mapping: i = j * T + t.
            NaN values preserved.
        M_chunk : np.ndarray, shape (N, D)
            Flattened mask matrix. M_chunk[i, d] = 1 if observed.

        Notes
        -----
        The original (j, t) index can be recovered:
            j = i // T
            t = i % T
        """
        J, T, D = X_window.shape
        N = J * T

        # Reshape from (J, T, D) to (N, D) using C-order (row-major)
        # i = j * T + t ensures consistent index mapping
        X_chunk = X_window.reshape(N, D).copy()
        M_chunk = M_window.reshape(N, D).astype(np.float32)

        logger.debug(
            f"Created data chunk: X_chunk={X_chunk.shape}, "
            f"M_chunk={M_chunk.shape}, N={N}, D={D}"
        )
        return X_chunk, M_chunk

    def flatten_windows(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert a list of windowed tensors to list of data chunks.

        Parameters
        ----------
        windows : List[Tuple[np.ndarray, np.ndarray]]
            List of (X_window, M_window) pairs, each shape (J, T, D).

        Returns
        -------
        chunks : List[Tuple[np.ndarray, np.ndarray]]
            List of (X_chunk, M_chunk) pairs, each shape (N, D).
        """
        chunks = []
        for i, (X_win, M_win) in enumerate(windows):
            X_chunk, M_chunk = self.create_data_chunk(X_win, M_win)
            chunks.append((X_chunk, M_chunk))
            logger.debug(f"Window {i+1}/{len(windows)} → chunk shape: {X_chunk.shape}")
        return chunks

    def compute_importance_score(
        self,
        X_chunk: np.ndarray,
        M_chunk: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance score phi for each instance in a data chunk.

        The importance score phi_i measures how much observed information
        instance i provides. It is used in MPIN-DM's data-update step to
        select "important" instances from previous chunks.

        Definition:
            phi_i = (number of observed attributes in instance i) / D

        Parameters
        ----------
        X_chunk : np.ndarray, shape (N, D)
            Data chunk. (Values not used directly, only M_chunk matters.)
        M_chunk : np.ndarray, shape (N, D)
            Observation mask. M_chunk[i, d] = 1 if attribute d of instance i
            is observed; 0 if missing.

        Returns
        -------
        phi : np.ndarray, shape (N,), dtype float
            Importance scores in [0, 1].
            phi[i] = 0 means all attributes of instance i are missing.
            phi[i] = 1 means all attributes are observed.

        Examples
        --------
        >>> M = np.array([[1, 1, 0, 1], [0, 0, 0, 1]])  # shape (2, 4)
        >>> phi = compute_importance_score(X, M)
        >>> phi  # [3/4, 1/4] = [0.75, 0.25]
        """
        N, D = M_chunk.shape
        # Sum observed (M=1) across attributes, normalize by D
        phi = M_chunk.sum(axis=1) / D  # shape (N,)
        logger.debug(
            f"Importance scores: mean={phi.mean():.4f}, "
            f"min={phi.min():.4f}, max={phi.max():.4f}"
        )
        return phi.astype(np.float32)

    def filter_by_importance(
        self,
        X_chunk: np.ndarray,
        M_chunk: np.ndarray,
        phi: np.ndarray,
        eta: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter instances from a data chunk by importance score threshold.

        Used in MPIN-DM's data-update step: keep only "important" instances
        from the previous chunk (phi >= eta) to augment the current chunk.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (N, D)
            Data chunk.
        M_chunk : np.ndarray, shape (N, D)
            Observation mask.
        phi : np.ndarray, shape (N,)
            Importance scores from compute_importance_score().
        eta : float, default 0.6
            Importance threshold. Instances with phi >= eta are retained.
            From the MPIN paper, eta ≈ 0.6 works well in practice.

        Returns
        -------
        X_important : np.ndarray, shape (N_keep, D)
            Filtered data chunk with only important instances.
        M_important : np.ndarray, shape (N_keep, D)
            Filtered mask.

        Notes
        -----
        If no instances pass the threshold (all phi < eta), returns empty arrays
        with shapes (0, D). The caller should handle this edge case.
        """
        keep_mask = phi >= eta  # Boolean mask of shape (N,)
        N_keep = keep_mask.sum()

        X_important = X_chunk[keep_mask]  # shape (N_keep, D)
        M_important = M_chunk[keep_mask]  # shape (N_keep, D)

        logger.info(
            f"Importance filter (eta={eta}): kept {N_keep}/{len(phi)} instances "
            f"({100*N_keep/len(phi):.1f}%)"
        )

        return X_important, M_important

    def augment_chunk(
        self,
        X_current: np.ndarray,
        M_current: np.ndarray,
        X_important: np.ndarray,
        M_important: np.ndarray,
        max_keep: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment current chunk with important instances from previous chunk.

        This is the data-update step in MPIN-DM: combine current window's
        data with important historical instances to provide richer context.

        Parameters
        ----------
        X_current : np.ndarray, shape (N_cur, D)
            Current window's data chunk.
        M_current : np.ndarray, shape (N_cur, D)
            Current window's mask.
        X_important : np.ndarray, shape (N_imp, D)
            Important instances from previous chunk (filtered by phi).
        M_important : np.ndarray, shape (N_imp, D)
            Masks for important instances.
        max_keep : int, optional
            Maximum number of historical instances to include.
            If set, randomly sample max_keep from X_important.

        Returns
        -------
        X_augmented : np.ndarray, shape (N_cur + N_keep, D)
        M_augmented : np.ndarray, shape (N_cur + N_keep, D)
        """
        N_imp = X_important.shape[0]

        if N_imp == 0:
            logger.debug("No important instances to augment with.")
            return X_current, M_current

        # Optionally cap the number of historical instances
        if max_keep is not None and N_imp > max_keep:
            idx = np.random.choice(N_imp, max_keep, replace=False)
            X_important = X_important[idx]
            M_important = M_important[idx]
            logger.debug(f"Sampled {max_keep} from {N_imp} important instances.")

        X_augmented = np.concatenate([X_current, X_important], axis=0)
        M_augmented = np.concatenate([M_current, M_important], axis=0)

        logger.info(
            f"Augmented chunk: {X_current.shape[0]} current + "
            f"{X_important.shape[0]} historical = {X_augmented.shape[0]} total instances."
        )
        return X_augmented, M_augmented

    def create_validation_mask(
        self,
        M_chunk: np.ndarray,
        val_ratio: float = 0.05,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a validation split from observed positions.

        In MPIN's transductive learning, a small fraction of observed
        positions are hidden during training and used for validation.
        This allows monitoring overfitting without a separate dataset.

        Parameters
        ----------
        M_chunk : np.ndarray, shape (N, D)
            Original observation mask (1=observed, 0=missing).
        val_ratio : float, default 0.05
            Fraction of observed positions to use as validation (5%).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        M_train : np.ndarray, shape (N, D)
            Training mask: 1 for training observed positions, 0 for missing
            and validation positions.
        M_val : np.ndarray, shape (N, D)
            Validation mask: 1 only at the held-out positions (val set).
            Use this to compute validation loss.
        """
        rng = np.random.RandomState(seed)

        # Find all observed positions
        observed_indices = np.argwhere(M_chunk == 1)  # shape (num_obs, 2)
        num_obs = len(observed_indices)
        num_val = max(1, int(num_obs * val_ratio))

        # Randomly select validation positions
        val_selection = rng.choice(num_obs, size=num_val, replace=False)
        val_indices = observed_indices[val_selection]  # shape (num_val, 2)

        # Build masks
        M_train = M_chunk.copy()
        M_val = np.zeros_like(M_chunk)

        for idx in val_indices:
            i, d = idx[0], idx[1]
            M_train[i, d] = 0   # Hide from training
            M_val[i, d] = 1     # Mark as validation

        logger.info(
            f"Validation split: {num_val}/{num_obs} observed positions "
            f"({100*val_ratio:.0f}%) held out for validation."
        )
        return M_train, M_val

    def fill_nan_with_mean(
        self,
        X_chunk: np.ndarray,
        M_chunk: np.ndarray
    ) -> np.ndarray:
        """
        Replace NaN values with column-wise mean (for graph construction).

        This is the initial mean-fill used before building the similarity
        graph in MPIN. It is NOT the final imputation result.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (N, D)
            Data with NaN at missing positions.
        M_chunk : np.ndarray, shape (N, D)
            Observation mask.

        Returns
        -------
        X_filled : np.ndarray, shape (N, D)
            Data with NaN replaced by attribute-wise mean (computed on
            observed values only).
        """
        X_filled = X_chunk.copy()
        N, D = X_chunk.shape

        for d in range(D):
            obs_mask = M_chunk[:, d].astype(bool)
            if obs_mask.sum() > 0:
                col_mean = X_chunk[obs_mask, d].mean()
            else:
                col_mean = 0.0  # Fallback if entire column is missing
            X_filled[~obs_mask, d] = col_mean

        return X_filled
