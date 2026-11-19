"""
src/models/mpin.py
==================
MPIN: Message Propagation Imputation Network

Implementation of the core MPIN model from:
"Missing Value Imputation for Multi-attribute Sensor Data Streams
 via Message Propagation" (VLDB 2023/2024, arXiv:2311.07344)

Architecture Overview:
----------------------
1. Input: Data chunk X ∈ R^{N×D} with mask M ∈ {0,1}^{N×D}
2. Similarity graph: Sparse KNN graph in COO edge_index ∈ Z^{2×E}
3. MsgPropLayer × num_layers: Vectorized, mask-aware message propagation
   - Differentiable correlation attention c(h_i, h_k) over graph edges
   - Confidence/mask-aware neighbor aggregation (avoids mean-fill dilution)
   - GraphSAGE-style hidden aggregation + cross-attribute residual refinement
   - Bound condition: preserve observed values, only update missing positions
4. Training: Dynamic self-supervised reconstruction masking over observed positions
   so gradients flow cleanly without bound-condition zeroing or validation leakage.

Usage:
------
    model = MPIN(D=37, num_layers=2)
    model, X_imputed, history = train_mpin(X_chunk, M_chunk, edge_index, config)
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _edge_softmax(
    logits: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """Compute numerically stable softmax over incoming edges per target node."""
    if logits.numel() == 0:
        return logits

    with torch.no_grad():
        max_v = torch.full(
            (num_nodes,), -torch.inf, device=logits.device, dtype=logits.dtype
        )
        max_v.scatter_reduce_(0, dst, logits, reduce="amax", include_self=True)

    exp_l = torch.exp(logits - max_v[dst])
    sum_v = torch.zeros(num_nodes, device=logits.device, dtype=logits.dtype)
    sum_v.index_add_(0, dst, exp_l)
    return exp_l / (sum_v[dst] + 1e-12)


def _to_edge_index(
    graph: Optional[Union[np.ndarray, torch.Tensor]],
    num_nodes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Normalize a graph representation (sparse edge_index or dense adj) to (2, E) LongTensor.

    Convention: edge_index[0] = source neighbor nodes, edge_index[1] = target nodes.
    """
    if graph is None:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    graph_t = torch.as_tensor(graph) if not isinstance(graph, torch.Tensor) else graph

    if graph_t.is_sparse:
        return graph_t.coalesce().indices().to(device=device, dtype=torch.long)

    if graph_t.ndim != 2:
        raise ValueError(f"Expected 2D graph (edge_index or adj), got shape {tuple(graph_t.shape)}")

    int_dtypes = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
    is_int = graph_t.dtype in int_dtypes

    if graph_t.shape[0] == 2 and (
        graph_t.shape[1] != num_nodes
        or (is_int and graph_t.numel() > 0 and int(graph_t.max().item()) >= 2)
    ):
        return graph_t.to(device=device, dtype=torch.long)

    if graph_t.shape == (num_nodes, num_nodes):
        dst, src = (graph_t > 0).nonzero(as_tuple=True)
        return torch.stack([src, dst], dim=0).to(device=device, dtype=torch.long)

    raise ValueError(
        f"Graph shape {tuple(graph_t.shape)} incompatible with num_nodes={num_nodes}."
    )


class CorrelationAttention(nn.Module):
    """
    Learnable scaled dot-product correlation attention c(h_i, h_k) over graph edges.

    Parameters
    ----------
    in_dim : int
        Node representation dimension H.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.scale = in_dim ** 0.5

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized attention weights for each edge (src -> dst).

        Parameters
        ----------
        h : torch.Tensor, shape (N, H)
            Encoded node representations.
        edge_index : torch.Tensor, shape (2, E)
            Directed edges [src; dst].

        Returns
        -------
        weights : torch.Tensor, shape (E,)
            Softmax-normalized weights per destination node.
        """
        src, dst = edge_index[0], edge_index[1]
        q = self.q_proj(h)
        k = self.k_proj(h)
        logits = (q[dst] * k[src]).sum(dim=-1) / self.scale
        return _edge_softmax(logits, dst, h.shape[0])


class MsgPropLayer(nn.Module):
    """
    Vectorized, mask-aware Message Propagation Layer (MsgProp).

    Combines:
    1. Learnable correlation attention over sparse KNN edges.
    2. Confidence-weighted neighbor attribute propagation (avoids mean-fill dilution).
    3. GraphSAGE-style hidden aggregation + cross-attribute residual refinement.
    4. Bound condition preserving observed positions exactly.
    """

    def __init__(self, D: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.D = D
        self.hidden_dim = hidden_dim

        self.feat_enc = nn.Sequential(
            nn.Linear(2 * D, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention = CorrelationAttention(hidden_dim)
        self.log_prior = nn.Parameter(torch.full((D,), -2.0))

        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neigh = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.refine = nn.Sequential(
            nn.Linear(hidden_dim + D, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, D),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        x_observed: torch.Tensor,
        mask: torch.Tensor,
        conf: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: sparse message propagation with bound condition.

        Parameters
        ----------
        x : torch.Tensor, shape (N, D)
            Current node features.
        edge_index : torch.Tensor, shape (2, E)
            Directed edge list [src; dst].
        x_observed : torch.Tensor, shape (N, D)
            Observed values (0 at missing positions).
        mask : torch.Tensor, shape (N, D)
            Binary observation mask (1=observed, 0=missing).
        conf : torch.Tensor, shape (N, D), optional
            Soft per-attribute confidence in [0, 1]. Defaults to `mask`.

        Returns
        -------
        x_out : torch.Tensor, shape (N, D)
            Updated features with observed entries preserved.
        new_conf : torch.Tensor, shape (N, D)
            Updated confidence scores for subsequent propagation layers.
        """
        if conf is None:
            conf = mask

        h = self.feat_enc(torch.cat([x, conf], dim=-1))

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            w = self.attention(h, edge_index)  # (E,)

            # Mask/confidence-aware attribute propagation
            w_conf = w.unsqueeze(-1) * conf[src]  # (E, D)
            num = torch.zeros_like(x)
            den = torch.zeros_like(x)
            num.index_add_(0, dst, w_conf * x[src])
            den.index_add_(0, dst, w_conf)

            prior = F.softplus(self.log_prior).unsqueeze(0)  # (1, D)
            x_prop = (num + prior * x) / (den + prior)
            new_conf = torch.clamp(
                mask + (1.0 - mask) * (den / (den + prior)), 0.0, 1.0
            )

            h_neigh = torch.zeros_like(h)
            h_neigh.index_add_(0, dst, w.unsqueeze(-1) * h[src])
        else:
            x_prop = x
            new_conf = conf
            h_neigh = h

        h_upd = h + self.dropout(
            F.gelu(self.norm(self.w_self(h) + self.w_neigh(h_neigh)))
        )
        delta = self.refine(torch.cat([h_upd, x_prop], dim=-1))
        x_pred = x_prop + self.res_scale * delta

        # Bound condition: preserve observed positions exactly
        x_out = x_observed * mask + x_pred * (1.0 - mask)
        return x_out, new_conf


class MPIN(nn.Module):
    """
    Message Propagation Imputation Network (MPIN).

    Stacks multiple MsgPropLayer modules to propagate attribute values and
    cross-attribute representations across multi-hop KNN neighborhoods.

    Parameters
    ----------
    D : int
        Number of attributes (feature dimension).
    num_layers : int, default 2
        Number of stacked MsgProp layers.
    hidden_dim : int, optional, default 64
        Latent hidden dimension inside MsgPropLayer.
    dropout : float, default 0.0
        Dropout rate inside MsgPropLayer.
    """

    def __init__(
        self,
        D: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.D = D
        self.num_layers = num_layers
        if hidden_dim is None:
            hidden_dim = 64
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [MsgPropLayer(D=D, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )

        logger.info(
            f"MPIN initialized: D={D}, num_layers={num_layers}, "
            f"hidden_dim={hidden_dim}, dropout={dropout}"
        )

    def forward(
        self,
        x_init: torch.Tensor,
        adj: Optional[torch.Tensor],
        x_observed: torch.Tensor,
        mask: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all MsgProp layers.

        Parameters
        ----------
        x_init : torch.Tensor, shape (N, D)
            Initial mean-filled node features.
        adj : torch.Tensor, optional
            Dense adjacency matrix (N, N) or sparse edge_index (2, E) if `edge_index` is None.
        x_observed : torch.Tensor, shape (N, D)
            Observed values (0 at missing positions).
        mask : torch.Tensor, shape (N, D)
            Observation mask (1=observed, 0=missing).
        edge_index : torch.Tensor, shape (2, E), optional
            Pre-converted COO edge index.

        Returns
        -------
        x_out : torch.Tensor, shape (N, D)
            Imputed matrix with observed values preserved.
        """
        ei = (
            edge_index
            if edge_index is not None
            else _to_edge_index(adj, x_init.shape[0], x_init.device)
        )

        x = x_init
        conf = mask
        for layer in self.layers:
            x, conf = layer(x, ei, x_observed, mask, conf)
        return x


def compute_correlation(
    xi: torch.Tensor,
    xk: torch.Tensor
) -> torch.Tensor:
    """Compute basic cosine correlation between two feature vectors."""
    return F.cosine_similarity(xi.unsqueeze(0), xk.unsqueeze(0)).squeeze()


def _column_means(X_no_nan: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """Compute per-attribute mean strictly over positions where mask_bool is True."""
    D = X_no_nan.shape[1]
    means = np.zeros(D, dtype=np.float32)
    for d in range(D):
        obs = mask_bool[:, d]
        if obs.any():
            means[d] = float(X_no_nan[obs, d].mean())
    return means


def train_mpin(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    adj: Optional[Union[np.ndarray, torch.Tensor]],
    config: dict,
    init_model: Optional[MPIN] = None,
    edge_index: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[MPIN, np.ndarray, Dict[str, list]]:
    """
    Full MPIN transductive training pipeline on a single data chunk.

    Uses dynamic self-supervised reconstruction masking over training-observed
    entries so that the bound condition never zeroes out training gradients,
    and avoids leaking held-out validation targets into initial features.

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Data chunk with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    adj : np.ndarray or torch.Tensor, optional
        Graph structure: either sparse edge_index (2, E) or dense adj (N, N).
    config : dict
        Configuration dict with settings under 'mpin' and 'general'.
    init_model : MPIN, optional
        Existing model to warm-start / fine-tune (used by MPIN-DM).
    edge_index : np.ndarray or torch.Tensor, optional
        Explicit sparse COO edge_index (2, E), overrides `adj` if provided.

    Returns
    -------
    model : MPIN
        Trained MPIN model.
    X_imputed : np.ndarray, shape (N, D)
        Imputed data chunk (observed values preserved, missing filled).
    history : dict
        {"train_losses": [...], "val_losses": [...]}.
    """
    mpin_cfg = config.get("mpin", {})
    epochs = int(mpin_cfg.get("epochs", 100))
    lr = float(mpin_cfg.get("lr", 0.001))
    weight_decay = float(mpin_cfg.get("weight_decay", 1e-4))
    val_split = float(mpin_cfg.get("val_split", 0.05))
    mask_ratio = float(mpin_cfg.get("mask_ratio", 0.2))
    num_layers = int(mpin_cfg.get("num_layers", 2))
    hidden_dim = int(mpin_cfg.get("hidden_dim", 64) or 64)
    dropout = float(mpin_cfg.get("dropout", 0.0))
    patience = int(mpin_cfg.get("patience", 20))
    loss_name = str(mpin_cfg.get("loss", "mse")).lower()
    use_scheduler = bool(mpin_cfg.get("use_scheduler", True))
    sched_patience = int(mpin_cfg.get("scheduler_patience", 10))
    sched_factor = float(mpin_cfg.get("scheduler_factor", 0.5))
    min_lr = float(mpin_cfg.get("min_lr", 1e-5))

    seed = int(config.get("general", {}).get("seed", 42))
    device_str = config.get("general", {}).get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    N, D = X_chunk.shape
    X_no_nan = np.nan_to_num(X_chunk, nan=0.0).astype(np.float32)
    M_float = M_chunk.astype(np.float32)

    graph_input = edge_index if edge_index is not None else adj
    edge_index_t = _to_edge_index(graph_input, N, device)

    model = (
        init_model.to(device)
        if init_model is not None
        else MPIN(D=D, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout).to(device)
    )

    obs_positions = np.argwhere(M_float == 1)
    if len(obs_positions) < 2:
        logger.warning("Too few observed entries (<2) to train MPIN; returning column-mean fill.")
        means_full = _column_means(X_no_nan, M_float == 1)
        X_fallback = np.where(M_float == 1, X_no_nan, means_full[None, :]).astype(np.float32)
        return model, X_fallback, {"train_losses": [], "val_losses": []}

    # --- Train / Validation split on observed entries ---
    num_val = max(1, min(len(obs_positions) - 1, int(len(obs_positions) * val_split)))
    rng = np.random.RandomState(seed)
    val_idx = rng.choice(len(obs_positions), size=num_val, replace=False)
    val_positions = obs_positions[val_idx]

    M_train = M_float.copy()
    M_train[val_positions[:, 0], val_positions[:, 1]] = 0.0

    # Compute initial fill means strictly from M_train (no validation leakage)
    col_means_tr = _column_means(X_no_nan, M_train == 1)
    col_means_full = _column_means(X_no_nan, M_float == 1)

    X_obs_t = torch.tensor(X_no_nan, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_float, dtype=torch.float32, device=device)
    M_train_t = torch.tensor(M_train, dtype=torch.float32, device=device)
    means_tr_t = torch.tensor(col_means_tr, dtype=torch.float32, device=device).unsqueeze(0)
    means_full_t = torch.tensor(col_means_full, dtype=torch.float32, device=device).unsqueeze(0)

    val_i = torch.tensor(val_positions[:, 0], dtype=torch.long, device=device)
    val_d = torch.tensor(val_positions[:, 1], dtype=torch.long, device=device)
    train_obs_flat = (M_train_t == 1).nonzero(as_tuple=False)

    loss_fn = F.l1_loss if loss_name in ("mae", "l1") else F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=sched_patience,
            factor=sched_factor, min_lr=min_lr
        )
        if use_scheduler
        else None
    )

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    train_losses = []
    val_losses = []

    logger.info(
        f"Training MPIN on {device}: epochs={epochs}, lr={lr}, N={N}, D={D}, "
        f"E={edge_index_t.shape[1]}, loss={loss_name}"
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Dynamic self-supervised masking over M_train
        m_loss = (M_train_t == 1) & (torch.rand_like(M_train_t) < mask_ratio)
        if m_loss.sum() == 0:
            pick = train_obs_flat[torch.randint(len(train_obs_flat), (1,), device=device).item()]
            m_loss[pick[0], pick[1]] = True

        m_in = M_train_t * (~m_loss).float()
        x_in = X_obs_t * m_in + means_tr_t * (1.0 - m_in)

        out = model(x_in, None, X_obs_t, m_in, edge_index=edge_index_t)
        loss = loss_fn(out[m_loss], X_obs_t[m_loss])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation: hide val_positions in both mask and input features
        model.eval()
        with torch.no_grad():
            x_val_in = X_obs_t * M_train_t + means_tr_t * (1.0 - M_train_t)
            val_out = model(x_val_in, None, X_obs_t, M_train_t, edge_index=edge_index_t)
            val_loss = loss_fn(val_out[val_i, val_d], X_obs_t[val_i, val_d]).item()

        if scheduler is not None:
            scheduler.step(val_loss)

        train_losses.append(float(loss.item()))
        val_losses.append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] train_loss={loss.item():.6f}, val_loss={val_loss:.6f}"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x_full_in = X_obs_t * M_t + means_full_t * (1.0 - M_t)
        X_imputed = model(
            x_full_in, None, X_obs_t, M_t, edge_index=edge_index_t
        ).cpu().numpy()

    logger.info(
        f"MPIN training complete. Best val_loss={best_val_loss:.6f}, "
        f"imputed shape={X_imputed.shape}"
    )
    return model, X_imputed, {"train_losses": train_losses, "val_losses": val_losses}
