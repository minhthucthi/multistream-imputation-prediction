"""
src/models/predictor.py
========================
Downstream prediction models for Task 2.

After imputing missing values, we use the imputed data for downstream prediction:
- Task: ICU in-hospital mortality prediction (binary classification)
- Input: 48-hour ICU time series with 37 physiological attributes (imputed)
- Output: P(mortality = 1)

Two model types:
1. LogisticPredictor: Logistic Regression on aggregated features (simple, interpretable)
2. LSTMPredictor: LSTM on full time series (captures temporal patterns)

The goal is not to maximize prediction performance, but to show that
better imputation → better downstream task performance.

Usage:
------
    # Logistic
    clf = LogisticPredictor(config)
    clf.fit(X_imputed_flat, y_labels)
    metrics = clf.evaluate(X_test_flat, y_test)

    # LSTM
    lstm = LSTMPredictor(D=37, config=config)
    train_predictor(X_imputed, y_labels, config)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ===========================================================================
# Logistic Regression Predictor
# ===========================================================================

class LogisticPredictor:
    """
    Logistic Regression for ICU mortality prediction.

    Uses scikit-learn's LogisticRegression on handcrafted features
    aggregated from the imputed time series (mean, std, min, max per attribute).

    This is the primary classifier for Task 2: simple, interpretable,
    and isolates the effect of imputation quality on downstream performance.

    Parameters
    ----------
    config : dict
        Configuration dictionary with 'prediction.logistic' settings.
    """

    def __init__(self, config: dict):
        self.config = config
        pred_cfg = config.get("prediction", {}).get("logistic", {})

        self.C = pred_cfg.get("C", 1.0)
        self.max_iter = pred_cfg.get("max_iter", 1000)
        self.class_weight = pred_cfg.get("class_weight", "balanced")
        self.solver = pred_cfg.get("solver", "lbfgs")

        self.model = None
        self.scaler = None
        self._fitted = False

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from imputed time series.

        Parameters
        ----------
        X : np.ndarray
            Shape (J, T, D) or (J, T*D) or (N, D) where N = J (after mean-pooling).

        Returns
        -------
        features : np.ndarray, shape (J, D*4)
            Concatenated [mean, std, min, max] per attribute across time.
        """
        if X.ndim == 3:
            # X shape (J, T, D) → aggregate over T
            feat_mean = np.nanmean(X, axis=1)  # (J, D)
            feat_std  = np.nanstd(X, axis=1)   # (J, D)
            feat_min  = np.nanmin(X, axis=1)   # (J, D)
            feat_max  = np.nanmax(X, axis=1)   # (J, D)
            features = np.concatenate([feat_mean, feat_std, feat_min, feat_max], axis=1)
        elif X.ndim == 2:
            # Already 2D (N, D) – use as-is
            features = X
        else:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")

        # Replace any remaining NaN (from all-missing columns) with 0
        features = np.nan_to_num(features, nan=0.0)
        return features.astype(np.float32)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticPredictor":
        """
        Train Logistic Regression classifier.

        Parameters
        ----------
        X_train : np.ndarray, shape (J, T, D) or (J, D)
            Training imputed data.
        y_train : np.ndarray, shape (J,)
            Binary labels (0=survived, 1=mortality).

        Returns
        -------
        self
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("scikit-learn required. Install: pip install scikit-learn")

        features = self._extract_features(X_train)

        # Standardize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        logger.info(
            f"Training LogisticPredictor: n_samples={len(y_train)}, "
            f"n_features={features.shape[1]}, "
            f"pos_rate={y_train.mean():.3f}"
        )

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=42
        )
        self.model.fit(features_scaled, y_train)
        self._fitted = True

        train_acc = self.model.score(features_scaled, y_train)
        logger.info(f"LogisticPredictor training accuracy: {train_acc:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        features = self._extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class (mortality)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        features = self._extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Parameters
        ----------
        X_test : np.ndarray
        y_test : np.ndarray

        Returns
        -------
        metrics : dict
            {'accuracy': ..., 'auc_roc': ..., 'f1': ..., 'precision': ..., 'recall': ...}
        """
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            precision_score, recall_score
        )

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0))
        }

        logger.info(
            f"LogisticPredictor evaluation: "
            f"Acc={metrics['accuracy']:.4f}, AUC={metrics['auc_roc']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )
        return metrics


# ===========================================================================
# LSTM Predictor (PyTorch)
# ===========================================================================

class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor for ICU time series classification.

    Takes full 48-hour imputed time series and outputs mortality probability.
    Captures temporal patterns that LogisticRegression on aggregated features misses.

    Architecture:
        Input: (batch, T, D)
           ↓
        LSTM: num_layers=2, hidden_dim=64
           ↓
        Last hidden state: (batch, hidden_dim)
           ↓
        Linear → Sigmoid → P(mortality)

    Parameters
    ----------
    D : int
        Input feature dimension (number of attributes, e.g., 37 for ICU).
    hidden_dim : int
        LSTM hidden state size.
    num_layers : int
        Number of LSTM layers (stacked).
    dropout : float
        Dropout between LSTM layers.
    """

    def __init__(
        self,
        D: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.D = D
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        logger.info(
            f"LSTMPredictor: D={D}, hidden={hidden_dim}, "
            f"layers={num_layers}, dropout={dropout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, D)
            Imputed time series.

        Returns
        -------
        logits : torch.Tensor, shape (batch,)
            Raw logits (apply sigmoid for probability).
        """
        # LSTM: output (batch, T, hidden), h_n (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)

        # Take hidden state of last layer
        last_hidden = h_n[-1]  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(last_hidden).squeeze(-1)  # (batch,)
        return logits


# ===========================================================================
# Training & Evaluation Functions
# ===========================================================================

def train_predictor(
    X_imputed: np.ndarray,
    y_labels: np.ndarray,
    config: dict,
    predictor_type: str = "logistic"
) -> Any:
    """
    Train a predictor on imputed data.

    Parameters
    ----------
    X_imputed : np.ndarray, shape (J, T, D) or (J, D)
        Imputed training data.
    y_labels : np.ndarray, shape (J,)
        Binary labels.
    config : dict
        Configuration dictionary.
    predictor_type : str
        "logistic" or "lstm".

    Returns
    -------
    model : trained predictor
    """
    pred_cfg = config.get("prediction", {})

    if predictor_type == "logistic":
        model = LogisticPredictor(config)
        model.fit(X_imputed, y_labels)
        return model

    elif predictor_type == "lstm":
        lstm_cfg = pred_cfg.get("lstm", {})
        hidden_dim = lstm_cfg.get("hidden_dim", 64)
        num_layers_lstm = lstm_cfg.get("num_layers", 2)
        dropout = lstm_cfg.get("dropout", 0.3)
        epochs = lstm_cfg.get("epochs", 50)
        lr = lstm_cfg.get("lr", 0.001)
        batch_size = lstm_cfg.get("batch_size", 64)

        device_str = config.get("general", {}).get("device", "cpu")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        if X_imputed.ndim == 2:
            raise ValueError(
                "LSTM predictor requires 3D input (J, T, D). "
                "Got 2D input. Use logistic predictor for flat features."
            )

        J, T, D = X_imputed.shape

        # Normalize to [0, 1] per attribute
        X_norm = X_imputed.copy()
        for d in range(D):
            col = X_norm[:, :, d]
            vmin, vmax = np.nanmin(col), np.nanmax(col)
            if vmax > vmin:
                X_norm[:, :, d] = (col - vmin) / (vmax - vmin)

        X_tensor = torch.tensor(np.nan_to_num(X_norm, nan=0.0), dtype=torch.float32)
        y_tensor = torch.tensor(y_labels, dtype=torch.float32)

        # Train/val split (80/20)
        n_train = int(0.8 * J)
        idx = np.random.permutation(J)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        X_train_t, y_train_t = X_tensor[train_idx], y_tensor[train_idx]
        X_val_t, y_val_t = X_tensor[val_idx], y_tensor[val_idx]

        model = LSTMPredictor(D=D, hidden_dim=hidden_dim,
                               num_layers=num_layers_lstm, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(y_labels == 0).sum() / (y_labels == 1).sum() + 1e-6])
        )

        logger.info(f"Training LSTMPredictor: epochs={epochs}, batch_size={batch_size}")

        for epoch in range(epochs):
            model.train()
            # Mini-batch training
            perm = torch.randperm(len(X_train_t))
            epoch_loss = 0.0
            for start in range(0, len(X_train_t), batch_size):
                batch_idx = perm[start:start+batch_size]
                xb = X_train_t[batch_idx].to(device)
                yb = y_train_t[batch_idx].to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_t.to(device))
                    val_loss = criterion(val_logits, y_val_t.to(device)).item()
                logger.info(f"LSTM Epoch [{epoch}/{epochs}]: val_loss={val_loss:.4f}")

        return model

    else:
        raise ValueError(f"Unknown predictor_type: '{predictor_type}'. Use 'logistic' or 'lstm'.")


def evaluate_predictor(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predictor_type: str = "logistic"
) -> Dict[str, float]:
    """
    Evaluate a trained predictor on test data.

    Parameters
    ----------
    model : trained predictor (LogisticPredictor or LSTMPredictor)
    X_test : np.ndarray
    y_test : np.ndarray
    predictor_type : str
        "logistic" or "lstm"

    Returns
    -------
    metrics : dict
        {'accuracy': ..., 'auc_roc': ..., 'f1': ..., ...}
    """
    if predictor_type == "logistic":
        return model.evaluate(X_test, y_test)

    elif predictor_type == "lstm":
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        model.eval()
        device = next(model.parameters()).device

        X_norm = np.nan_to_num(X_test, nan=0.0).astype(np.float32)
        X_t = torch.tensor(X_norm).to(device)

        with torch.no_grad():
            logits = model(X_t)
            proba = torch.sigmoid(logits).cpu().numpy()

        y_pred = (proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, proba)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0))
        }
        logger.info(f"LSTM Predictor: {metrics}")
        return metrics

    else:
        raise ValueError(f"Unknown predictor_type: '{predictor_type}'.")
