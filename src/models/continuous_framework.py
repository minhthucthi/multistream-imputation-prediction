"""
src/models/continuous_framework.py
===================================
MPIN-DM: Continuous/Dynamic MPIN Framework

Implements the continuous framework (MPIN-DM) described in Section 4 of
arXiv:2311.07344, enabling MPIN to handle streaming data efficiently.

Two core mechanisms:
1. Data Update: Augment incoming chunk with important historical instances
   (instances with high phi score from previous chunks)
2. Model Update: Warm-start fine-tuning of the message passing module
   on new data (transfer learning to avoid full retraining)

Workflow for each new time window W_k:
   prev_model, prev_important ← state from W_{k-1}
   X_aug = augment(W_k, prev_important)  ← Data Update
   model_k = fine_tune(prev_model, X_aug) ← Model Update
   X_imputed = model_k.infer(X_aug)
   Update state: prev_model = model_k, prev_important = filter(W_k, eta)

Usage:
------
    dm = MPINcontinuous(D=37, config=config)
    for X_chunk, M_chunk in streaming_chunks:
        X_imputed = dm.process_window(X_chunk, M_chunk)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import copy
import logging

from src.models.mpin import MPIN, train_mpin
from src.models.similarity_graph import build_knn_edge_index

logger = logging.getLogger(__name__)


class MPINcontinuous:
    """
    MPIN-DM: Dynamic/Continuous MPIN for streaming data imputation.

    Manages the full lifecycle of MPIN over a data stream:
    - Maintains a "warm" model from the previous window
    - Maintains a set of "important" historical instances
    - Applies data update and model update for each new window

    Parameters
    ----------
    D : int
        Number of attributes (feature dimension).
    config : dict
        Full configuration dictionary.
    """

    def __init__(self, D: int, config: dict):
        self.D = D
        self.config = config
        self.mpin_cfg = config.get("mpin", {})
        self.dm_cfg = config.get("mpin_dm", {})

        # Hyperparameters
        self.eta = self.dm_cfg.get("eta", 0.6)
        self.max_keep = self.dm_cfg.get("max_keep", 500)
        self.fine_tune_epochs = self.dm_cfg.get("fine_tune_epochs", 20)
        self.fine_tune_lr = self.dm_cfg.get("fine_tune_lr", 1e-4)

        self.device_str = config.get("general", {}).get("device", "cpu")
        self.device = torch.device(
            self.device_str if torch.cuda.is_available() else "cpu"
        )

        # Internal state
        self._prev_model: Optional[MPIN] = None          # Model from previous window
        self._prev_X_important: Optional[np.ndarray] = None  # Important instances
        self._prev_M_important: Optional[np.ndarray] = None  # Their masks
        self._window_count: int = 0                        # Number of processed windows

        logger.info(
            f"MPINcontinuous initialized: D={D}, eta={self.eta}, "
            f"max_keep={self.max_keep}, fine_tune_epochs={self.fine_tune_epochs}"
        )

    def process_window(
        self,
        X_chunk: np.ndarray,
        M_chunk: np.ndarray
    ) -> np.ndarray:
        """
        Process a new streaming window and return imputed data.

        This is the main entry point for each new time window.
        Performs data update, model update, and imputation.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (N, D)
            Current window's data chunk (NaN at missing positions).
        M_chunk : np.ndarray, shape (N, D)
            Current window's observation mask.

        Returns
        -------
        X_imputed : np.ndarray, shape (N_aug, D)
            Imputed data for the augmented chunk (current + historical).
            The first N rows correspond to the current window.
        """
        self._window_count += 1
        logger.info(f"Processing window {self._window_count}...")

        # --- Step 1: Data Update ---
        X_aug, M_aug = self.data_update(X_chunk, M_chunk)

        # --- Step 2: Build similarity graph ---
        edge_index, _ = build_knn_edge_index(
            X_aug, M_aug,
            k=self.mpin_cfg.get("K", 10),
            metric=self.mpin_cfg.get("graph_metric", "euclidean"),
            symmetric=self.mpin_cfg.get("symmetric", True)
        )

        # --- Step 3: Model Update (or initial training) ---
        if self._prev_model is None:
            # First window: train from scratch
            logger.info("First window – training MPIN from scratch.")
            model, X_imputed, _ = train_mpin(X_aug, M_aug, edge_index, self.config)
        else:
            # Subsequent windows: warm-start fine-tuning
            logger.info(f"Window {self._window_count} – fine-tuning from previous model.")
            model, X_imputed = self.model_update(
                self._prev_model, X_aug, M_aug, edge_index
            )

        # --- Step 4: Update internal state ---
        self._prev_model = model

        # Compute importance scores for current chunk (not augmented)
        phi_current = self._compute_phi(M_chunk)
        X_imp_current, M_imp_current = self.filter_by_importance(
            X_imputed[:X_chunk.shape[0]],  # Only current window rows
            M_chunk,
            phi_current
        )
        self._prev_X_important = X_imp_current
        self._prev_M_important = M_imp_current

        logger.info(
            f"Window {self._window_count} done. "
            f"Important instances saved: {X_imp_current.shape[0]}"
        )

        # Return only the imputed values for the current window (not historical)
        return X_imputed[:X_chunk.shape[0]]

    def data_update(
        self,
        X_current: np.ndarray,
        M_current: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment current chunk with important instances from previous chunk.

        This is the "Data Update" step of MPIN-DM:
            X_aug = X_current ∪ {x_i ∈ W_{k-1} : phi(x_i) >= eta}

        Parameters
        ----------
        X_current : np.ndarray, shape (N_cur, D)
            Current window's data chunk.
        M_current : np.ndarray, shape (N_cur, D)
            Current window's observation mask.

        Returns
        -------
        X_aug : np.ndarray, shape (N_cur + N_imp, D)
            Augmented chunk. If no historical data, returns current chunk only.
        M_aug : np.ndarray, shape (N_cur + N_imp, D)
            Augmented mask.
        """
        if self._prev_X_important is None or len(self._prev_X_important) == 0:
            logger.debug("No historical important instances. Using current chunk only.")
            return X_current.copy(), M_current.copy()

        N_imp = len(self._prev_X_important)

        # Optionally cap historical instances
        if self.max_keep is not None and N_imp > self.max_keep:
            rng = np.random.RandomState(self._window_count)
            idx = rng.choice(N_imp, size=self.max_keep, replace=False)
            X_hist = self._prev_X_important[idx]
            M_hist = self._prev_M_important[idx]
            logger.debug(f"Sampled {self.max_keep}/{N_imp} historical instances.")
        else:
            X_hist = self._prev_X_important
            M_hist = self._prev_M_important

        X_aug = np.concatenate([X_current, X_hist], axis=0)
        M_aug = np.concatenate([M_current, M_hist], axis=0)

        logger.info(
            f"Data update: {X_current.shape[0]} current + "
            f"{X_hist.shape[0]} historical = {X_aug.shape[0]} total."
        )
        return X_aug, M_aug

    def model_update(
        self,
        prev_model: MPIN,
        X_aug: np.ndarray,
        M_aug: np.ndarray,
        adj: np.ndarray
    ) -> Tuple[MPIN, np.ndarray]:
        """
        Fine-tune the previous MPIN model on new augmented data.

        This is the "Model Update" step of MPIN-DM:
        - Start from the previous window's model weights (warm-start)
        - Fine-tune via `train_mpin` with lower learning rate and fewer epochs
        - Avoids full retraining from scratch while reusing self-supervised masking

        Parameters
        ----------
        prev_model : MPIN
            Model trained on previous window(s).
        X_aug : np.ndarray, shape (N_aug, D)
            Augmented data chunk for current window.
        M_aug : np.ndarray, shape (N_aug, D)
            Augmented mask.
        adj : np.ndarray
            Sparse edge_index (2, E) or dense adjacency matrix (N_aug, N_aug).

        Returns
        -------
        model : MPIN
            Fine-tuned model.
        X_imputed : np.ndarray, shape (N_aug, D)
            Imputed data from the fine-tuned model.
        """
        model = copy.deepcopy(prev_model).to(self.device)
        for param in model.parameters():
            param.requires_grad = True

        ft_config = copy.deepcopy(self.config)
        ft_mpin = ft_config.setdefault("mpin", {})
        ft_mpin["epochs"] = self.fine_tune_epochs
        ft_mpin["lr"] = self.fine_tune_lr
        ft_mpin["patience"] = max(5, self.fine_tune_epochs // 2)

        model, X_imputed, history = train_mpin(
            X_aug, M_aug, adj, ft_config, init_model=model
        )
        val_losses = history.get("val_losses", [])
        best_val = min(val_losses) if val_losses else float("nan")
        logger.info(
            f"Model update complete: {len(val_losses)} epochs, "
            f"best_val_loss={best_val:.6f}"
        )
        return model, X_imputed

    def filter_by_importance(
        self,
        X_chunk: np.ndarray,
        M_chunk: np.ndarray,
        phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter instances by importance score.

        Parameters
        ----------
        X_chunk : np.ndarray, shape (N, D)
        M_chunk : np.ndarray, shape (N, D)
        phi : np.ndarray, shape (N,)

        Returns
        -------
        X_important : np.ndarray, shape (N_keep, D)
        M_important : np.ndarray, shape (N_keep, D)
        """
        keep = phi >= self.eta
        return X_chunk[keep].copy(), M_chunk[keep].copy()

    def _compute_phi(self, M_chunk: np.ndarray) -> np.ndarray:
        """Compute importance score phi_i = (observed_count_i) / D."""
        return M_chunk.mean(axis=1).astype(np.float32)

    def reset(self):
        """Reset internal state (start fresh stream)."""
        self._prev_model = None
        self._prev_X_important = None
        self._prev_M_important = None
        self._window_count = 0
        logger.info("MPINcontinuous state reset.")
