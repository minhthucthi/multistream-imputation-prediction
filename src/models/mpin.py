"""
src/models/mpin.py
==================
MPIN: Message Propagation Imputation Network

Implementation of the core MPIN model from:
"Missing Value Imputation for Multi-attribute Sensor Data Streams
 via Message Propagation" (VLDB 2023, arXiv:2311.07344)

Architecture Overview:
----------------------
1. Input: Data chunk X ∈ R^{N×D} with mask M ∈ {0,1}^{N×D}
2. Similarity graph: KNN graph Adj ∈ {0,1}^{N×N}
3. MsgPropLayer × num_layers: GraphSAGE-style message propagation
   - For each node i: aggregate weighted features from neighbors N(i)
   - x̃_i = Σ_{k∈N(i)} c(x_i, x_k) * x_k * W
   - Bound condition: keep observed values, only update missing positions
4. Output: Imputed X̂ ∈ R^{N×D} (all missing values filled)

Key Design Choices:
-------------------
- Transductive learning: train and infer on the same chunk
- Validation split 5% of observed positions
- 2 MsgProp layers (K-hop receptive field = 2)
- GraphSAGE-style aggregation with learnable correlation weights
- Bound condition preserves observed values exactly

Usage:
------
    model = MPIN(D=37, num_layers=2, K=10)
    X_imputed = train_mpin(X_chunk, M_chunk, adj, config)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CorrelationAttention(nn.Module):
    """
    Learnable correlation weight c(x_i, x_k).

    Computes a scalar attention weight for each neighbor k of node i,
    based on the concatenated features of i and k.

    Parameters
    ----------
    in_dim : int
        Input feature dimension D.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        # Attention vector: maps [x_i || x_k] (size 2*D) to scalar
        self.attn_vec = nn.Linear(2 * in_dim, 1, bias=False)

    def forward(
        self,
        x_i: torch.Tensor,
        x_neighbors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correlation weights for node i over all neighbors.

        Parameters
        ----------
        x_i : torch.Tensor, shape (D,) or (1, D)
            Feature vector of the target node i.
        x_neighbors : torch.Tensor, shape (K, D)
            Feature vectors of K neighbors.

        Returns
        -------
        weights : torch.Tensor, shape (K,)
            Softmax-normalized attention weights. Sum to 1.
        """
        K, D = x_neighbors.shape
        # Expand x_i to match neighbors: (K, D)
        x_i_expanded = x_i.expand(K, -1)  # (K, D)
        # Concatenate [x_i || x_k] for each k: shape (K, 2D)
        concat = torch.cat([x_i_expanded, x_neighbors], dim=-1)
        # Attention logits: shape (K, 1) → (K,)
        logits = self.attn_vec(concat).squeeze(-1)
        # Softmax to get normalized weights
        weights = F.softmax(logits / (D ** 0.5), dim=0)
        return weights


class MsgPropLayer(nn.Module):
    """
    Message Propagation Layer (MsgProp) – GraphSAGE-style aggregation.

    For each node i, aggregates information from its KNN neighbors:
        x̃_i = Σ_{k ∈ N(i)} c(x_i, x_k) · x_k · W

    Then applies the bound condition:
        x_i^{out}[d] = x_i^{observed}[d]  if m_i[d] = 1  (preserve observed)
        x_i^{out}[d] = x̃_i[d]            if m_i[d] = 0  (fill missing)

    Parameters
    ----------
    in_dim : int
        Input feature dimension D.
    out_dim : int
        Output feature dimension (usually same as in_dim = D).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Learnable weight matrix W: maps aggregated neighbor features
        self.W = nn.Linear(in_dim, out_dim, bias=True)

        # Correlation attention
        self.attention = CorrelationAttention(in_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        x_observed: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: message propagation with bound condition.

        Parameters
        ----------
        x : torch.Tensor, shape (N, D)
            Current node features (may include previously imputed values).
        adj : torch.Tensor, shape (N, N)
            Binary adjacency matrix (1 = edge exists).
        x_observed : torch.Tensor, shape (N, D)
            Original observed values (NaN where missing, used for bound condition).
        mask : torch.Tensor, shape (N, D)
            Observation mask (1=observed, 0=missing).

        Returns
        -------
        x_out : torch.Tensor, shape (N, out_dim)
            Updated node features after message propagation and bound condition.
        """
        N, D = x.shape
        x_aggregated = torch.zeros(N, self.out_dim, device=x.device)

        # Message propagation: for each node i, aggregate from neighbors
        for i in range(N):
            # Find neighbors of node i
            neighbor_indices = adj[i].nonzero(as_tuple=True)[0]  # shape (K,)

            if len(neighbor_indices) == 0:
                # Isolated node: use self-features transformed by W
                x_aggregated[i] = self.W(x[i])
                continue

            # Get neighbor features: (K, D)
            x_neighbors = x[neighbor_indices]  # (K, D)

            # Compute correlation weights: (K,)
            with torch.no_grad():
                # Detach for weight computation (don't backprop through attention twice)
                weights = self.attention(x[i].detach(), x_neighbors.detach())

            # Weighted aggregation: Σ c(x_i, x_k) * x_k
            # weights: (K,) → (K, 1) for broadcasting
            weighted_sum = (weights.unsqueeze(-1) * x_neighbors).sum(dim=0)  # (D,)

            # Apply linear transformation W
            x_aggregated[i] = self.W(weighted_sum)  # (out_dim,)

        # Apply layer normalization
        x_aggregated = self.layer_norm(x_aggregated)
        x_aggregated = F.relu(x_aggregated)

        # Bound condition: preserve observed values, fill missing
        # mask: (N, D), 1=observed 0=missing
        x_out = x.clone()
        # Only update positions where mask = 0 (missing)
        missing_mask = (mask == 0).float()  # (N, D)
        x_out = x_observed * mask + x_aggregated * missing_mask

        return x_out


class MPIN(nn.Module):
    """
    Message Propagation Imputation Network (MPIN).

    Stacks multiple MsgPropLayer to progressively refine imputation estimates.
    With 2 layers, information propagates through 2-hop neighborhoods.

    Parameters
    ----------
    D : int
        Number of attributes (feature dimension).
    num_layers : int, default 2
        Number of MsgProp layers. MPIN paper uses 2.
    hidden_dim : int, default None
        Hidden dimension. If None, keeps D throughout.
    dropout : float, default 0.0
        Dropout rate between layers.

    Architecture:
    -------------
    Input X (N, D)
       ↓
    [MsgPropLayer 1] → X_1 (N, D)
       ↓
    [MsgPropLayer 2] → X_2 (N, D)
       ↓
    Output X_hat (N, D) — imputed data
    """

    def __init__(
        self,
        D: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.D = D
        self.num_layers = num_layers
        self.dropout = dropout

        if hidden_dim is None:
            hidden_dim = D  # Keep same dimension throughout

        # Build stack of MsgProp layers
        self.layers = nn.ModuleList()
        in_dim = D
        for l in range(num_layers):
            out_dim = hidden_dim if l < num_layers - 1 else D
            self.layers.append(MsgPropLayer(in_dim, out_dim))
            in_dim = out_dim

        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        logger.info(
            f"MPIN initialized: D={D}, num_layers={num_layers}, "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )

    def forward(
        self,
        x_init: torch.Tensor,
        adj: torch.Tensor,
        x_observed: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through all MsgProp layers.

        Parameters
        ----------
        x_init : torch.Tensor, shape (N, D)
            Initial node features (mean-filled: observed values + mean for missing).
        adj : torch.Tensor, shape (N, N)
            Binary adjacency matrix.
        x_observed : torch.Tensor, shape (N, D)
            Original observed values (preserved by bound condition).
            Missing positions should be 0 (not NaN).
        mask : torch.Tensor, shape (N, D)
            Observation mask (1=observed, 0=missing).

        Returns
        -------
        x_out : torch.Tensor, shape (N, D)
            Imputed data: observed values unchanged, missing values filled.
        """
        x = x_init

        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, adj, x_observed, mask)
            if layer_idx < self.num_layers - 1:
                x = self.dropout_layer(x)

        return x


def compute_correlation(
    xi: torch.Tensor,
    xk: torch.Tensor
) -> torch.Tensor:
    """
    Compute basic cosine correlation between two feature vectors.

    This is a simplified version of the learned correlation weight.
    The full MPIN uses a learned attention vector (see CorrelationAttention).

    Parameters
    ----------
    xi : torch.Tensor, shape (D,)
    xk : torch.Tensor, shape (D,)

    Returns
    -------
    corr : torch.Tensor, scalar
        Correlation score in [-1, 1].
    """
    return F.cosine_similarity(xi.unsqueeze(0), xk.unsqueeze(0)).squeeze()


def train_mpin(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    adj: np.ndarray,
    config: dict
) -> Tuple[MPIN, np.ndarray]:
    """
    Full MPIN training pipeline on a single data chunk (transductive).

    Steps:
    1. Prepare tensors and initial mean-filled features
    2. Create validation split (5% of observed positions)
    3. Train MPIN with MSE loss on training observed positions
    4. Evaluate on validation split, early stopping
    5. Return trained model and imputed X

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Data chunk with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    adj : np.ndarray, shape (N, N)
        Pre-built KNN adjacency matrix.
    config : dict
        Configuration dict with keys under 'mpin':
            - epochs: int (default 100)
            - lr: float (default 0.001)
            - val_split: float (default 0.05)
            - num_layers: int (default 2)
            - patience: int (default 20)

    Returns
    -------
    model : MPIN
        Trained MPIN model.
    X_imputed : np.ndarray, shape (N, D)
        Imputed data chunk. Observed values unchanged; missing filled.
    """
    mpin_cfg = config.get("mpin", {})
    epochs = mpin_cfg.get("epochs", 100)
    lr = mpin_cfg.get("lr", 0.001)
    val_split = mpin_cfg.get("val_split", 0.05)
    num_layers = mpin_cfg.get("num_layers", 2)
    patience = mpin_cfg.get("patience", 20)
    weight_decay = mpin_cfg.get("weight_decay", 1e-4)

    device_str = config.get("general", {}).get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Training MPIN on device: {device}")

    N, D = X_chunk.shape

    # --- Prepare tensors ---
    # Replace NaN with 0 for the mask-based computation
    X_no_nan = np.nan_to_num(X_chunk, nan=0.0)
    M_float = M_chunk.astype(np.float32)

    # Mean-fill for initial features
    X_mean_filled = X_no_nan.copy()
    for d in range(D):
        obs = M_float[:, d].astype(bool)
        if obs.sum() > 0:
            fill_val = X_no_nan[obs, d].mean()
        else:
            fill_val = 0.0
        X_mean_filled[~obs, d] = fill_val

    # Convert to tensors
    X_init = torch.tensor(X_mean_filled, dtype=torch.float32, device=device)
    X_obs = torch.tensor(X_no_nan, dtype=torch.float32, device=device)
    M = torch.tensor(M_float, dtype=torch.float32, device=device)
    Adj = torch.tensor(adj, dtype=torch.float32, device=device)

    # --- Create validation split ---
    # Find observed positions
    obs_positions = np.argwhere(M_chunk == 1)  # shape (num_obs, 2)
    num_val = max(1, int(len(obs_positions) * val_split))
    rng = np.random.RandomState(42)
    val_idx = rng.choice(len(obs_positions), size=num_val, replace=False)
    val_positions = obs_positions[val_idx]  # shape (num_val, 2)
    val_i = torch.tensor(val_positions[:, 0], dtype=torch.long, device=device)
    val_d = torch.tensor(val_positions[:, 1], dtype=torch.long, device=device)

    # Training mask: M_train = M but val positions set to 0
    M_train = M.clone()
    M_train[val_i, val_d] = 0.0

    # --- Initialize model ---
    model = MPIN(D=D, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    logger.info(f"Starting MPIN training: epochs={epochs}, lr={lr}, N={N}, D={D}")

    # --- Training loop ---
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        X_imputed = model(X_init, Adj, X_obs, M_train)

        # Training loss: MSE on training observed positions (not val, not missing)
        # Find positions: M_train = 1 (training observed)
        train_obs_mask = M_train  # (N, D)
        loss = F.mse_loss(
            X_imputed * train_obs_mask,
            X_obs * train_obs_mask,
            reduction="sum"
        ) / (train_obs_mask.sum() + 1e-8)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation loss: MSE on held-out positions
        with torch.no_grad():
            val_pred = X_imputed[val_i, val_d]
            val_true = X_obs[val_i, val_d]
            val_loss = F.mse_loss(val_pred, val_true).item()

        scheduler.step(val_loss)
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"train_loss={loss.item():.6f}, val_loss={val_loss:.6f}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final imputation on full chunk ---
    model.eval()
    with torch.no_grad():
        X_imputed_tensor = model(X_init, Adj, X_obs, M)

    X_imputed = X_imputed_tensor.cpu().numpy()
    logger.info(
        f"MPIN training complete. Best val_loss={best_val_loss:.6f}. "
        f"Imputed shape: {X_imputed.shape}"
    )

    return model, X_imputed, {"train_losses": train_losses, "val_losses": val_losses}
