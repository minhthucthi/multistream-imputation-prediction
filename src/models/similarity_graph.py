"""
src/models/similarity_graph.py
==============================
Similarity graph construction for MPIN.

The similarity graph is a K-Nearest Neighbor (KNN) graph where:
- Each node represents one instance (j, t) = one stream at one timestep
- Two nodes are connected if they are among each other's K nearest neighbors
- Similarity is measured by Euclidean distance on mean-filled attribute vectors

This graph captures "which instances are similar to each other" and guides
the message propagation in MPIN.

Reference: Section 3.2 of arXiv:2311.07344
"""

import numpy as np
from typing import Tuple, Optional
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def fill_initial(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray
) -> np.ndarray:
    """
    Perform mean-fill on a data chunk for initial graph construction.

    Missing values are replaced by the column-wise mean computed from
    observed values. This is used only to enable distance computation
    before MPIN imputation.

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Data chunk with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).

    Returns
    -------
    X_filled : np.ndarray, shape (N, D)
        Data with NaN replaced by per-attribute mean.
        No NaN values remain.
    """
    X_filled = X_chunk.copy()
    N, D = X_chunk.shape

    for d in range(D):
        obs_mask = M_chunk[:, d].astype(bool)
        if obs_mask.sum() > 0:
            col_mean = np.nanmean(X_chunk[obs_mask, d])
        else:
            col_mean = 0.0
            logger.warning(f"Attribute {d} is entirely missing. Using 0.0 as fill value.")
        # Fill NaN positions
        X_filled[~obs_mask, d] = col_mean
        # Also fill any remaining NaN (safety)
        nan_mask = np.isnan(X_filled[:, d])
        if nan_mask.any():
            X_filled[nan_mask, d] = col_mean

    assert not np.isnan(X_filled).any(), "NaN values remain after mean-fill!"
    return X_filled


def compute_pairwise_distance(
    X_filled: np.ndarray,
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute pairwise distance matrix between all instances.

    Parameters
    ----------
    X_filled : np.ndarray, shape (N, D)
        Mean-filled data chunk (no NaN).
    metric : str, default "euclidean"
        Distance metric. Options: "euclidean", "cosine", "manhattan".
        MPIN paper uses Euclidean distance.

    Returns
    -------
    dist_matrix : np.ndarray, shape (N, N)
        Pairwise distance matrix. dist_matrix[i, k] = distance(i, k).
        Symmetric: dist_matrix[i, k] == dist_matrix[k, i].
        Diagonal is 0: dist_matrix[i, i] = 0.

    Notes
    -----
    For large N (N = J * T), this is O(N^2 * D) in memory and time.
    For N > 10000, consider approximate NN methods (e.g., FAISS).
    """
    supported = {"euclidean", "cosine", "manhattan", "cityblock"}
    if metric == "manhattan":
        metric = "cityblock"  # scipy alias
    if metric not in supported and metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {supported}")

    logger.info(f"Computing pairwise {metric} distances for N={X_filled.shape[0]} instances...")
    dist_matrix = cdist(X_filled, X_filled, metric=metric)  # shape (N, N)
    logger.info(f"Distance matrix computed: shape={dist_matrix.shape}")
    return dist_matrix.astype(np.float32)


def knn_graph(
    dist_matrix: np.ndarray,
    k: int = 10,
    symmetric: bool = True
) -> np.ndarray:
    """
    Build a K-Nearest Neighbor adjacency matrix from a distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray, shape (N, N)
        Pairwise distance matrix.
    k : int, default 10
        Number of nearest neighbors per node.
        MPIN paper uses K=10.
    symmetric : bool, default True
        If True, make the graph undirected:
            if k is in N(i), add i in N(k) as well.
        This ensures the graph is symmetric (Adj = Adj^T).

    Returns
    -------
    adj : np.ndarray, shape (N, N), dtype int
        Binary adjacency matrix.
        adj[i, k] = 1 if there is an edge between nodes i and k.
        adj[i, i] = 0 (no self-loops).

    Notes
    -----
    Self-connections are excluded: each node's nearest neighbors do not
    include itself (diagonal is effectively masked out).
    """
    N = dist_matrix.shape[0]
    adj = np.zeros((N, N), dtype=np.int32)
    k_eff = min(k, max(N - 1, 0))
    if k_eff == 0:
        return adj

    # Set diagonal to infinity so nodes don't select themselves
    dist_matrix_no_self = dist_matrix.copy()
    np.fill_diagonal(dist_matrix_no_self, np.inf)

    # For each node, find K nearest neighbors
    for i in range(N):
        # argsort ascending: closest first
        nn_indices = np.argsort(dist_matrix_no_self[i])[:k_eff]
        adj[i, nn_indices] = 1

    # Make symmetric (undirected graph)
    if symmetric:
        adj = np.maximum(adj, adj.T)

    # Remove self-loops (should already be 0, but safety check)
    np.fill_diagonal(adj, 0)

    num_edges = adj.sum() // 2 if symmetric else adj.sum()
    logger.info(
        f"KNN graph (K={k}, symmetric={symmetric}): "
        f"{N} nodes, {num_edges} edges, avg_degree={adj.sum(1).mean():.2f}"
    )
    return adj


def build_similarity_graph(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    symmetric: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: build KNN similarity graph from a data chunk.

    Steps:
    1. Mean-fill missing values (for distance computation only)
    2. Compute pairwise Euclidean distance
    3. Build K-nearest neighbor adjacency matrix

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Data chunk with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    k : int, default 10
        Number of KNN neighbors (from MPIN paper).
    metric : str, default "euclidean"
        Distance metric for KNN graph construction.
    symmetric : bool, default True
        Whether to make the graph undirected.

    Returns
    -------
    adj : np.ndarray, shape (N, N)
        Binary adjacency matrix.
    X_filled : np.ndarray, shape (N, D)
        Mean-filled version of X_chunk (used for initial node features).

    Example
    -------
    >>> X_chunk.shape = (600, 37)  # 100 streams * 6 timesteps
    >>> adj, X_filled = build_similarity_graph(X_chunk, M_chunk, k=10)
    >>> adj.shape  # (600, 600)
    """
    logger.info(
        f"Building similarity graph: N={X_chunk.shape[0]}, D={X_chunk.shape[1]}, "
        f"K={k}, metric={metric}"
    )

    # Step 1: Mean-fill for distance computation
    X_filled = fill_initial(X_chunk, M_chunk)

    # Step 2: Pairwise distance
    dist_matrix = compute_pairwise_distance(X_filled, metric=metric)

    # Step 3: KNN adjacency
    adj = knn_graph(dist_matrix, k=k, symmetric=symmetric)

    return adj, X_filled


def build_knn_edge_index(
    X_chunk: np.ndarray,
    M_chunk: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    symmetric: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scalably build a sparse KNN similarity graph in COO edge_index format.

    Avoids constructing an O(N^2) dense distance or adjacency matrix by using
    scikit-learn NearestNeighbors directly.

    Parameters
    ----------
    X_chunk : np.ndarray, shape (N, D)
        Data chunk with NaN at missing positions.
    M_chunk : np.ndarray, shape (N, D)
        Observation mask (1=observed, 0=missing).
    k : int, default 10
        Number of nearest neighbors per node.
    metric : str, default "euclidean"
        Distance metric ("euclidean", "cosine", "manhattan", "cityblock").
    symmetric : bool, default True
        Whether to make the graph undirected (if src->dst exists, add dst->src).

    Returns
    -------
    edge_index : np.ndarray, shape (2, E), dtype int64
        Directed edge list where edge_index[0] = source neighbor nodes and
        edge_index[1] = target nodes.
    X_filled : np.ndarray, shape (N, D)
        Mean-filled version of X_chunk.
    """
    from sklearn.neighbors import NearestNeighbors

    X_filled = fill_initial(X_chunk, M_chunk)
    N = X_filled.shape[0]
    k_eff = min(int(k), max(N - 1, 0))

    if N <= 1 or k_eff <= 0:
        return np.zeros((2, 0), dtype=np.int64), X_filled

    metric_map = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "manhattan": "manhattan",
        "cityblock": "manhattan",
    }
    if metric not in metric_map:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: {set(metric_map.keys())}")
    sklearn_metric = metric_map[metric]

    logger.info(
        f"Building sparse KNN edge_index: N={N}, D={X_chunk.shape[1]}, "
        f"K={k_eff}, metric={sklearn_metric}, symmetric={symmetric}"
    )

    nn_model = NearestNeighbors(
        n_neighbors=k_eff + 1, metric=sklearn_metric, n_jobs=-1
    )
    nn_model.fit(X_filled)
    _, indices = nn_model.kneighbors(X_filled)

    # Strip self-loops robustly even when duplicate all-missing rows exist
    is_self = (indices == np.arange(N)[:, None])
    no_self_row = ~is_self.any(axis=1)
    is_self[no_self_row, -1] = True
    keep = ~is_self

    src = indices[keep].astype(np.int64)
    dst = np.repeat(np.arange(N, dtype=np.int64), k_eff)

    if symmetric:
        src_all = np.concatenate([src, dst])
        dst_all = np.concatenate([dst, src])
        keys = src_all * N + dst_all
        _, uniq_idx = np.unique(keys, return_index=True)
        src = src_all[uniq_idx]
        dst = dst_all[uniq_idx]

    edge_index = np.stack([src, dst], axis=0)
    logger.info(f"Sparse KNN graph built: {N} nodes, {edge_index.shape[1]} directed edges.")
    return edge_index, X_filled


def adj_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Convert adjacency matrix to edge_index format (for PyTorch Geometric).

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Binary adjacency matrix.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E)
        Edge index in COO format where E = number of directed edges.
        edge_index[0] = source nodes
        edge_index[1] = target nodes
    """
    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0)  # shape (2, E)
    logger.debug(f"Edge index: {edge_index.shape[1]} directed edges.")
    return edge_index


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalize the adjacency matrix: D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Binary adjacency matrix (without self-loops).

    Returns
    -------
    adj_norm : np.ndarray, shape (N, N)
        Normalized adjacency.
    """
    # Add self-loops
    adj_sl = adj + np.eye(adj.shape[0])
    # Degree matrix
    d = adj_sl.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_sl @ D_inv_sqrt
    return adj_norm.astype(np.float32)
