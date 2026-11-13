"""
src/data/loader.py
==================
Data loading utilities for MPIN project.

Supports:
- ICU dataset (PhysioNet Challenge 2012): D=37 attributes, 11988 streams, ~80% sparsity
- Airquality dataset: D=11 attributes, multi-station monitoring

Usage:
    loader = DataLoader(config)
    X, M = loader.load_icu_dataset("data/raw/icu/")
    windows = loader.create_tumbling_windows(X, M, window_size=12)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ICU: List of 37 physiological attributes in PhysioNet 2012
ICU_ATTRIBUTES = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS",
    "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
    "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
    "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"
]

# Airquality: 11 attributes
AIRQUALITY_ATTRIBUTES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "Temp", "Pres", "Dewp", "Rain", "WS"
]


class DataLoader:
    """
    Main data loader class for MPIN experiments.

    Handles loading raw datasets, applying normalization,
    creating tumbling windows, and generating masks for missing values.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from config.yaml.
        Expected keys: 'icu', 'airquality', etc.
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("DataLoader initialized.")

    # ------------------------------------------------------------------
    # ICU Dataset
    # ------------------------------------------------------------------

    def load_icu_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load ICU dataset from PhysioNet Challenge 2012 raw files.

        The ICU dataset consists of multiple patient records, each stored
        as a text file with (Time, Parameter, Value) format.

        Parameters
        ----------
        path : str
            Path to raw ICU directory containing set-a/, set-b/ subdirs
            and Outcomes-a.txt, Outcomes-b.txt label files.

        Returns
        -------
        X : np.ndarray, shape (J, T, D)
            Data tensor. J=num_patients, T=48 (hours), D=37 attributes.
            NaN where values are missing.
        M : np.ndarray, shape (J, T, D), dtype bool
            Observation mask. M[j,t,d]=True if X[j,t,d] is observed.
        y : np.ndarray, shape (J,), dtype int, optional
            Labels: 1=in-hospital death, 0=survived. None if no labels.
        """
        logger.info(f"Loading ICU dataset from: {path}")

        # TODO: Implement full ICU loading pipeline
        # Step 1: Scan all patient files in set-a/ and set-b/
        # Step 2: Parse each .txt file into a dict {param: [(time, value), ...]}
        # Step 3: Bucket timestamps into 1-hour bins (bin 0 = hours [0,1))
        # Step 4: Build tensor X of shape (J, 48, 37)
        # Step 5: Build mask M (True where not NaN)
        # Step 6: Load labels from Outcomes-a.txt / Outcomes-b.txt

        # --- Placeholder Implementation ---
        processed_X_path = os.path.join(
            self.config.get("icu", {}).get("processed_dir", "data/processed/icu/"),
            "X_icu.npy"
        )
        processed_M_path = os.path.join(
            self.config.get("icu", {}).get("processed_dir", "data/processed/icu/"),
            "M_icu.npy"
        )
        processed_y_path = os.path.join(
            self.config.get("icu", {}).get("processed_dir", "data/processed/icu/"),
            "y_icu.npy"
        )

        if os.path.exists(processed_X_path):
            logger.info("Loading pre-processed ICU data from cache.")
            X = np.load(processed_X_path, allow_pickle=True)
            M = np.load(processed_M_path, allow_pickle=True)
            y = np.load(processed_y_path, allow_pickle=True) if os.path.exists(processed_y_path) else None
            logger.info(f"Loaded X: {X.shape}, M: {M.shape}")
            return X, M, y

        # If raw data exists, parse it
        set_a_dir = os.path.join(path, "set-a")
        set_b_dir = os.path.join(path, "set-b")

        if not os.path.exists(set_a_dir):
            raise FileNotFoundError(
                f"ICU raw data not found at {set_a_dir}. "
                "Please run: python scripts/download_data.py --dataset icu"
            )

        # TODO: Implement _parse_icu_directory
        X, M, y = self._parse_icu_directory(path)

        # Save processed data
        os.makedirs(os.path.dirname(processed_X_path), exist_ok=True)
        np.save(processed_X_path, X)
        np.save(processed_M_path, M)
        if y is not None:
            np.save(processed_y_path, y)

        return X, M, y

    def _parse_icu_directory(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse ICU raw text files into tensors.

        Parameters
        ----------
        path : str
            Path to ICU raw directory.

        Returns
        -------
        X : np.ndarray, shape (J, 48, 37) with NaN for missing
        M : np.ndarray, shape (J, 48, 37), bool mask
        y : np.ndarray, shape (J,), int labels
        """
        # TODO: Full implementation
        # For each patient file in set-a/, set-b/:
        #   1. Read CSV-like content
        #   2. Convert 'Time' string "HH:MM" to hour index (0-47)
        #   3. Map 'Parameter' to attribute index using ICU_ATTRIBUTES
        #   4. Fill in X[patient_idx, hour_idx, attr_idx] = value
        # Then load labels from Outcomes-*.txt

        raise NotImplementedError(
            "TODO: Implement _parse_icu_directory. "
            "See doc/05_dataset_guide.md for file format details."
        )

    def load_icu_labels(self, path: str) -> np.ndarray:
        """
        Load in-hospital mortality labels from ICU Outcomes files.

        Parameters
        ----------
        path : str
            Path to ICU raw directory containing Outcomes-a.txt.

        Returns
        -------
        y : np.ndarray, shape (J,), dtype int
            1 = in-hospital death, 0 = survived.
        """
        # TODO: Parse Outcomes-a.txt and Outcomes-b.txt
        # Format: RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death
        outcomes_path = os.path.join(path, "Outcomes-a.txt")
        if not os.path.exists(outcomes_path):
            raise FileNotFoundError(f"Labels file not found: {outcomes_path}")

        # TODO: implement
        raise NotImplementedError("TODO: Implement load_icu_labels.")

    # ------------------------------------------------------------------
    # Airquality Dataset
    # ------------------------------------------------------------------

    def load_airquality_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Air Quality multi-station monitoring dataset.

        Parameters
        ----------
        path : str
            Path to airquality raw directory.

        Returns
        -------
        X : np.ndarray, shape (J, T_total, D)
            J=num_stations, T_total=total_timesteps, D=11 attributes.
            NaN where values are missing.
        M : np.ndarray, shape (J, T_total, D), bool
            Observation mask.
        """
        logger.info(f"Loading Airquality dataset from: {path}")

        processed_X_path = os.path.join(
            self.config.get("airquality", {}).get("processed_dir", "data/processed/airquality/"),
            "X_airquality.npy"
        )
        processed_M_path = os.path.join(
            self.config.get("airquality", {}).get("processed_dir", "data/processed/airquality/"),
            "M_airquality.npy"
        )

        if os.path.exists(processed_X_path):
            logger.info("Loading pre-processed Airquality data from cache.")
            X = np.load(processed_X_path, allow_pickle=True)
            M = np.load(processed_M_path, allow_pickle=True)
            logger.info(f"Loaded X: {X.shape}, M: {M.shape}")
            return X, M

        # TODO: Load raw CSV files, pivot by station
        # Expected structure:
        #   data/raw/airquality/
        #     ├── station_001.csv   (timestamp, PM2.5, PM10, ...)
        #     └── station_002.csv

        raise NotImplementedError(
            "TODO: Implement load_airquality_dataset. "
            "Download data first: python scripts/download_data.py --dataset airquality"
        )

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------

    def create_tumbling_windows(
        self,
        X: np.ndarray,
        M: np.ndarray,
        window_size: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Partition a time-series tensor into non-overlapping tumbling windows.

        Parameters
        ----------
        X : np.ndarray, shape (J, T_total, D)
            Full data tensor. NaN for missing.
        M : np.ndarray, shape (J, T_total, D)
            Observation mask.
        window_size : int
            Size of each tumbling window (number of timesteps T_w).

        Returns
        -------
        windows : List[Tuple[np.ndarray, np.ndarray]]
            List of (X_chunk, M_chunk) pairs.
            Each X_chunk has shape (J, T_w, D).
            Each M_chunk has shape (J, T_w, D).

        Notes
        -----
        Remaining timesteps at the end (< window_size) are discarded.

        Example
        -------
        >>> X.shape = (100, 48, 37), window_size=12
        >>> len(windows) = 4  # 48 / 12 = 4 windows
        >>> windows[0][0].shape = (100, 12, 37)
        """
        J, T_total, D = X.shape
        num_windows = T_total // window_size
        windows = []

        for w in range(num_windows):
            t_start = w * window_size
            t_end = t_start + window_size
            X_chunk = X[:, t_start:t_end, :]   # shape (J, T_w, D)
            M_chunk = M[:, t_start:t_end, :]   # shape (J, T_w, D)
            windows.append((X_chunk, M_chunk))

        logger.info(
            f"Created {num_windows} tumbling windows of size {window_size} "
            f"from {T_total} total timesteps."
        )

        if T_total % window_size != 0:
            discarded = T_total % window_size
            logger.warning(
                f"Discarded last {discarded} timesteps (< window_size={window_size})."
            )

        return windows

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        X: np.ndarray,
        M: np.ndarray,
        method: str = "minmax"
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize data attributes to [0, 1] or standard normal.

        Normalization statistics are computed only on observed values (M=1)
        to avoid data leakage from imputed positions.

        Parameters
        ----------
        X : np.ndarray, shape (J, T, D) or (N, D)
            Data tensor with NaN for missing values.
        M : np.ndarray, same shape as X
            Observation mask (1=observed, 0=missing).
        method : str
            "minmax" for Min-Max normalization to [0,1].
            "zscore" for Z-score normalization (mean=0, std=1).

        Returns
        -------
        X_norm : np.ndarray
            Normalized data (NaN positions preserved as NaN).
        stats : dict
            Normalization statistics for inverse-transform later.
            For minmax: {"min": array(D,), "max": array(D,)}
            For zscore: {"mean": array(D,), "std": array(D,)}
        """
        # Reshape to (N, D) for easier computation
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1]).copy()  # (N, D)
        M_flat = M.reshape(-1, M.shape[-1]).astype(bool)

        D = X_flat.shape[1]
        X_norm = X_flat.copy()

        if method == "minmax":
            x_min = np.full(D, np.nan)
            x_max = np.full(D, np.nan)

            for d in range(D):
                observed = X_flat[M_flat[:, d], d]
                if len(observed) > 0:
                    x_min[d] = observed.min()
                    x_max[d] = observed.max()
                    denom = x_max[d] - x_min[d]
                    if denom > 0:
                        X_norm[M_flat[:, d], d] = (observed - x_min[d]) / denom
                    else:
                        X_norm[M_flat[:, d], d] = 0.0

            stats = {"min": x_min, "max": x_max, "method": "minmax"}

        elif method == "zscore":
            x_mean = np.full(D, np.nan)
            x_std = np.full(D, np.nan)

            for d in range(D):
                observed = X_flat[M_flat[:, d], d]
                if len(observed) > 0:
                    x_mean[d] = observed.mean()
                    x_std[d] = observed.std() + 1e-8
                    X_norm[M_flat[:, d], d] = (observed - x_mean[d]) / x_std[d]

            stats = {"mean": x_mean, "std": x_std, "method": "zscore"}

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")

        X_norm = X_norm.reshape(original_shape)
        logger.info(f"Normalized data using {method}. Shape: {X_norm.shape}")
        return X_norm, stats

    def denormalize(self, X_norm: np.ndarray, stats: dict) -> np.ndarray:
        """
        Inverse normalization to recover original scale.

        Parameters
        ----------
        X_norm : np.ndarray
            Normalized data.
        stats : dict
            Statistics from normalize() call.

        Returns
        -------
        X_orig : np.ndarray
            Data in original scale.
        """
        method = stats.get("method", "minmax")
        X_orig = X_norm.copy()

        if method == "minmax":
            x_min = stats["min"]
            x_max = stats["max"]
            X_orig = X_norm * (x_max - x_min) + x_min
        elif method == "zscore":
            X_orig = X_norm * stats["std"] + stats["mean"]

        return X_orig

    def load_synthetic_dataset(
        self,
        num_streams: int = 10,
        num_timesteps: int = 48,
        num_attributes: int = 37,
        sparsity: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic multi-stream temporal dataset for quick testing and debugging.

        Parameters
        ----------
        num_streams : int
            Number of sensor streams (patients/stations).
        num_timesteps : int
            Number of time steps per stream.
        num_attributes : int
            Number of sensor features.
        sparsity : float
            Fraction of values to set as missing (NaN).

        Returns
        -------
        X : np.ndarray, shape (num_streams, num_timesteps, num_attributes)
            Data with NaN where missing.
        M : np.ndarray, shape (num_streams, num_timesteps, num_attributes), dtype bool
            Observation mask (True where observed).
        y : np.ndarray, shape (num_streams,), binary labels
        """
        logger.info(
            f"Generating synthetic dataset: streams={num_streams}, "
            f"timesteps={num_timesteps}, attributes={num_attributes}, sparsity={sparsity:.1%}"
        )
        np.random.seed(42)
        X = (np.random.randn(num_streams, num_timesteps, num_attributes).astype(np.float32) * 10) + 50.0
        M = (np.random.rand(num_streams, num_timesteps, num_attributes) > sparsity)

        # Ensure at least 1 observed value per instance
        for j in range(num_streams):
            for t in range(num_timesteps):
                if not M[j, t].any():
                    M[j, t, 0] = True

        X[~M] = np.nan
        y = np.random.randint(0, 2, size=num_streams).astype(np.int64)
        return X, M, y
