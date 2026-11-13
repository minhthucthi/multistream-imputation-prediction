"""
src/visualization/plots.py
===========================
Visualization utilities for MPIN project.

Provides functions to visualize:
1. Missing value patterns (heatmap)
2. Method comparison (bar charts)
3. Imputed vs. true time series (line plots)
4. Training loss curves
5. ROC curves for prediction task

Usage:
------
    from src.visualization.plots import plot_missing_pattern, plot_comparison_bar
    plot_missing_pattern(M_chunk, title="ICU Missing Pattern")
    plot_comparison_bar(results_dict, metric="mae", save_path="results/plots/mae_bar.png")
"""

import numpy as np
import os
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Lazy import matplotlib/seaborn to avoid import errors if not installed
def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

def _get_sns():
    try:
        import seaborn as sns
        return sns
    except ImportError:
        raise ImportError("seaborn required. Install: pip install seaborn")


def _setup_style(config: Optional[dict] = None) -> None:
    """Apply matplotlib style settings."""
    plt = _get_plt()
    style = "seaborn-v0_8-whitegrid"
    if config:
        style = config.get("visualization", {}).get("style", style)
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("ggplot")  # Fallback


def _save_or_show(
    plt_module,
    save_path: Optional[str] = None,
    show: bool = False,
    dpi: int = 150
) -> None:
    """Save and/or show figure."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt_module.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    if show:
        plt_module.show()
    plt_module.close()


def plot_missing_pattern(
    M: np.ndarray,
    title: str = "Missing Value Pattern",
    save_path: Optional[str] = None,
    show: bool = False,
    max_instances: int = 200,
    max_attrs: int = 37
) -> None:
    """
    Visualize missing value pattern as a heatmap.

    Shows which (instance, attribute) pairs are observed (white) or missing (black).

    Parameters
    ----------
    M : np.ndarray, shape (N, D) or (J, T, D)
        Observation mask. 1=observed, 0=missing.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure (e.g., "results/plots/missing_pattern.png").
    show : bool
        Whether to display the plot interactively.
    max_instances : int
        Maximum number of instances to show (for readability).
    max_attrs : int
        Maximum number of attributes to show.

    Example
    -------
    >>> plot_missing_pattern(M_chunk, title="ICU Window Missing Pattern",
    ...                      save_path="results/plots/icu_missing.png")
    """
    plt = _get_plt()
    sns = _get_sns()
    _setup_style()

    # Flatten to 2D if needed
    if M.ndim == 3:
        J, T, D = M.shape
        M_2d = M.reshape(-1, D)
    else:
        M_2d = M

    N, D = M_2d.shape

    # Subsample for readability
    if N > max_instances:
        idx = np.linspace(0, N-1, max_instances, dtype=int)
        M_2d = M_2d[idx]
    if D > max_attrs:
        M_2d = M_2d[:, :max_attrs]

    fig, ax = plt.subplots(figsize=(min(D/2, 15), min(N/10, 8)))

    # Heatmap: 1=white (observed), 0=black (missing)
    sns.heatmap(
        M_2d.T,  # Transpose: rows=attributes, cols=instances
        ax=ax,
        cmap="binary_r",  # 0=black (missing), 1=white (observed)
        cbar=True,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Instances (j×t)", fontsize=11)
    ax.set_ylabel("Attributes (d)", fontsize=11)

    missing_rate = 1.0 - M_2d.mean()
    ax.text(
        0.01, 0.01,
        f"Missing rate: {100*missing_rate:.1f}%",
        transform=ax.transAxes,
        fontsize=9, color="gray"
    )

    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)


def plot_comparison_bar(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = "mae",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot a bar chart comparing imputation methods on a given metric.

    Parameters
    ----------
    results_dict : dict
        {method_name: {metric_name: value, ...}}
        Example: {"MEAN": {"mae": 0.12, ...}, "MPIN": {"mae": 0.05, ...}}
    metric : str
        Metric to plot. Options: "mae", "mre", "rmse".
    title : str, optional
        Plot title. Auto-generated if None.
    save_path : str, optional
        File path to save.
    show : bool
        Whether to display interactively.

    Example
    -------
    >>> plot_comparison_bar(results, metric="mae",
    ...                     save_path="results/plots/icu_mae_bar.png")
    """
    plt = _get_plt()
    _setup_style()

    methods = list(results_dict.keys())
    values = []
    for m in methods:
        val = results_dict[m].get(metric, np.nan)
        values.append(float(val) if not isinstance(val, str) else np.nan)

    # Color: highlight MPIN in orange, others in steelblue
    colors = []
    for m in methods:
        if "mpin" in m.lower():
            colors.append("#FF8C00")  # Orange for MPIN
        else:
            colors.append("#4682B4")  # Steelblue for baselines

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    bars = ax.bar(methods, values, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9
            )

    metric_labels = {"mae": "MAE", "mre": "MRE", "rmse": "RMSE"}
    metric_label = metric_labels.get(metric.lower(), metric.upper())

    ax.set_title(title or f"Imputation Comparison – {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric_label + " ↓ (lower is better)", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Find best and annotate
    valid = [(m, v) for m, v in zip(methods, values) if not np.isnan(v)]
    if valid:
        best_method = min(valid, key=lambda x: x[1])[0]
        ax.text(
            0.99, 0.99,
            f"Best: {best_method}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#FF8C00",
            fontweight="bold"
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)


def plot_imputed_vs_true(
    X_true: np.ndarray,
    X_imputed: np.ndarray,
    M: np.ndarray,
    feature_idx: int = 0,
    stream_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot imputed vs. true time series for a specific attribute and stream.

    Shows:
    - True values (all timesteps as dashed gray line)
    - Observed values (green circles)
    - Imputed values at missing positions (red crosses)

    Parameters
    ----------
    X_true : np.ndarray, shape (J, T, D) or (N, D)
        Ground truth data.
    X_imputed : np.ndarray, shape same as X_true
        Imputed data.
    M : np.ndarray, same shape as X_true
        Observation mask (1=observed, 0=missing).
    feature_idx : int
        Which attribute (d) to plot.
    stream_idx : int
        Which stream (j) to plot. Ignored if 2D.
    title : str, optional
    save_path : str, optional
    show : bool
    """
    plt = _get_plt()
    _setup_style()

    if X_true.ndim == 3:
        x_true = X_true[stream_idx, :, feature_idx]
        x_imp = X_imputed[stream_idx, :, feature_idx]
        m = M[stream_idx, :, feature_idx]
        T = x_true.shape[0]
    else:
        # 2D: use directly
        x_true = X_true[:, feature_idx]
        x_imp = X_imputed[:, feature_idx]
        m = M[:, feature_idx]
        T = len(x_true)

    timesteps = np.arange(T)
    obs_mask = m.astype(bool)
    miss_mask = ~obs_mask

    fig, ax = plt.subplots(figsize=(12, 4))

    # True values (ground truth line)
    ax.plot(timesteps, x_true, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7, label="True (ground truth)", zorder=1)

    # Observed values
    ax.scatter(
        timesteps[obs_mask], x_true[obs_mask],
        color="#2ECC71", s=30, zorder=3, label="Observed"
    )

    # Imputed values at missing positions
    ax.scatter(
        timesteps[miss_mask], x_imp[miss_mask],
        color="#E74C3C", marker="x", s=50, linewidths=1.5,
        zorder=3, label="Imputed (missing)"
    )

    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Value (normalized)", fontsize=11)
    ax.set_title(
        title or f"Imputed vs. True | Stream {stream_idx}, Attribute {feature_idx}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # MAE at missing positions
    if miss_mask.sum() > 0:
        mae_val = np.abs(x_imp[miss_mask] - x_true[miss_mask]).mean()
        ax.text(
            0.01, 0.95,
            f"MAE at missing = {mae_val:.4f}",
            transform=ax.transAxes,
            fontsize=9, color="#E74C3C", va="top"
        )

    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)


def plot_training_loss(
    losses: Dict[str, List[float]],
    title: str = "MPIN Training Loss",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    losses : dict
        {"train_losses": [...], "val_losses": [...]}
    title : str
    save_path : str, optional
    show : bool
    """
    plt = _get_plt()
    _setup_style()

    fig, ax = plt.subplots(figsize=(10, 4))

    train_losses = losses.get("train_losses", [])
    val_losses = losses.get("val_losses", [])
    epochs = range(1, len(train_losses) + 1)

    if train_losses:
        ax.plot(epochs, train_losses, label="Train MSE", color="#3498DB", linewidth=1.5)
    if val_losses:
        ax.plot(epochs, val_losses, label="Val MSE", color="#E74C3C",
                linewidth=1.5, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val = min(val_losses)
        ax.axvline(x=best_epoch, color="green", linestyle=":", alpha=0.7)
        ax.text(
            best_epoch, best_val,
            f" Best (epoch {best_epoch})\n val={best_val:.4f}",
            fontsize=8, color="green"
        )

    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    title: str = "ROC Curves – ICU Mortality Prediction",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot ROC curves for prediction task (if proba data available).

    Parameters
    ----------
    results : dict
        {method_name: {"auc_roc": ..., "y_true": ..., "y_proba": ...}}
    title : str
    save_path : str
    show : bool
    """
    plt = _get_plt()
    _setup_style()

    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.error("scikit-learn required for ROC curves.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    for (method, result), color in zip(results.items(), colors):
        y_true = result.get("y_true")
        y_proba = result.get("y_proba")
        auc = result.get("auc_roc", 0)

        if y_true is not None and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{method} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)


def plot_attribute_mae(
    mae_per_attr: np.ndarray,
    attribute_names: Optional[List[str]] = None,
    title: str = "MAE per Attribute",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot per-attribute MAE as a horizontal bar chart.

    Parameters
    ----------
    mae_per_attr : np.ndarray, shape (D,)
        MAE for each attribute.
    attribute_names : list, optional
        Names of attributes. If None, uses "Attr 0", "Attr 1", ...
    title : str
    save_path : str
    show : bool
    """
    plt = _get_plt()
    _setup_style()

    D = len(mae_per_attr)
    if attribute_names is None:
        attribute_names = [f"Attr {d}" for d in range(D)]

    # Sort by MAE
    sorted_idx = np.argsort(mae_per_attr)
    sorted_mae = mae_per_attr[sorted_idx]
    sorted_names = [attribute_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, D * 0.3)))
    colors = ["#E74C3C" if v > np.nanmean(mae_per_attr) else "#3498DB"
              for v in sorted_mae]

    ax.barh(sorted_names, sorted_mae, color=colors, edgecolor="white")
    ax.set_xlabel("MAE ↓", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(np.nanmean(mae_per_attr), color="gray", linestyle="--",
               alpha=0.7, label=f"Mean={np.nanmean(mae_per_attr):.4f}")
    ax.legend()
    plt.tight_layout()
    _save_or_show(plt, save_path=save_path, show=show)
