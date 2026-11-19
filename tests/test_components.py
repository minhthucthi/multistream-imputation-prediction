"""
Unit tests for data loader, similarity graph, metrics, and MPIN layer forward pass.
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add project root and src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.models.similarity_graph import build_similarity_graph, build_knn_edge_index
from src.models.mpin import MPIN, train_mpin
from src.models.baselines import mean_imputation
from src.models.continuous_framework import MPINcontinuous
from src.evaluation.metrics import mae, mre, create_evaluation_mask


class TestMPINPipeline(unittest.TestCase):
    def setUp(self):
        # Create a tiny synthetic data chunk: 4 instances, 3 attributes
        # Instance 0 has missing values
        self.X = np.array([
            [1.0, np.nan, 3.0],
            [1.2, 2.1, 2.9],
            [5.0, 5.2, 5.1],
            [np.nan, 2.0, 3.2]
        ], dtype=np.float32)

        self.M = (~np.isnan(self.X)).astype(np.float32)

    def test_metrics(self):
        X_true = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        X_imputed = np.array([[1.0, 2.5, 3.0]], dtype=np.float32)
        # Position (0, 1) is missing
        M_missing = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        val_mae = mae(X_true, X_imputed, M_missing)
        self.assertAlmostEqual(val_mae, 0.5, places=4)

        val_mre = mre(X_true, X_imputed, M_missing)
        self.assertAlmostEqual(val_mre, 0.25, places=4)

    def test_similarity_graph(self):
        adj, _ = build_similarity_graph(self.X, self.M, k=2, metric="euclidean")
        self.assertEqual(adj.shape, (4, 4))
        # Self-connections or neighbors should exist
        self.assertTrue(np.all(adj >= 0))

    def test_mpin_forward(self):
        model = MPIN(D=3, num_layers=2)
        adj = torch.eye(4)
        x_tensor = torch.tensor(np.nan_to_num(self.X, nan=0.0), dtype=torch.float32)
        m_tensor = torch.tensor(self.M, dtype=torch.float32)

        out = model(x_tensor, adj, x_tensor, m_tensor)
        self.assertEqual(out.shape, (4, 3))
        # Verify bound condition: observed values must remain intact in output
        observed_mask = (m_tensor == 1.0)
        self.assertTrue(torch.allclose(out[observed_mask], x_tensor[observed_mask], atol=1e-5))

    def test_mre_with_zeros(self):
        X_true = np.array([[0.0, 2.0, 0.0, 2.0]], dtype=np.float32)
        X_imputed = np.array([[0.5, 2.5, 0.0, 2.0]], dtype=np.float32)
        M_missing = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

        val_mre = mre(X_true, X_imputed, M_missing)
        # Sum of errors = 0.5 + 0.5 + 0.0 = 1.0; sum of |true| = 0 + 2 + 0 = 2.0 -> 0.5
        self.assertAlmostEqual(val_mre, 0.5, places=4)
        self.assertTrue(np.isfinite(val_mre))

    def test_build_knn_edge_index_and_duplicates(self):
        # Include 3 identical all-missing rows alongside observed rows
        X_dup = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 2.9],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 5.1, 5.2],
        ], dtype=np.float32)
        M_dup = (~np.isnan(X_dup)).astype(np.float32)

        edge_index, X_filled = build_knn_edge_index(X_dup, M_dup, k=3, symmetric=True)
        self.assertEqual(edge_index.ndim, 2)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertFalse(np.isnan(X_filled).any())
        # No self-loops
        self.assertFalse(np.any(edge_index[0] == edge_index[1]))
        # Symmetric: every (u, v) has reverse (v, u)
        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for u, v in edge_set:
            self.assertIn((v, u), edge_set)

    def test_mpin_forward_edge_index_and_empty(self):
        model = MPIN(D=3, num_layers=2, hidden_dim=16)
        x_tensor = torch.tensor(np.nan_to_num(self.X, nan=0.0), dtype=torch.float32)
        m_tensor = torch.tensor(self.M, dtype=torch.float32)

        edge_index, _ = build_knn_edge_index(self.X, self.M, k=2, symmetric=True)
        ei_t = torch.tensor(edge_index, dtype=torch.long)
        out = model(x_tensor, None, x_tensor, m_tensor, edge_index=ei_t)
        self.assertEqual(out.shape, (4, 3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out[m_tensor == 1.0], x_tensor[m_tensor == 1.0], atol=1e-5))

        empty_ei = torch.zeros((2, 0), dtype=torch.long)
        out_empty = model(x_tensor, None, x_tensor, m_tensor, edge_index=empty_ei)
        self.assertEqual(out_empty.shape, (4, 3))
        self.assertTrue(torch.isfinite(out_empty).all())

    def test_mpin_gradients_flow(self):
        torch.manual_seed(42)
        model = MPIN(D=3, num_layers=2, hidden_dim=16)
        x_obs = torch.tensor(np.nan_to_num(self.X, nan=0.0), dtype=torch.float32)
        mask = torch.tensor(self.M, dtype=torch.float32)
        edge_index, _ = build_knn_edge_index(self.X, self.M, k=2, symmetric=True)
        ei_t = torch.tensor(edge_index, dtype=torch.long)

        # Mask out one observed position for reconstruction loss
        m_in = mask.clone()
        m_in[1, 1] = 0.0
        x_in = x_obs * m_in

        out = model(x_in, None, x_obs, m_in, edge_index=ei_t)
        loss = torch.nn.functional.mse_loss(out[1, 1], x_obs[1, 1])
        loss.backward()

        for layer in model.layers:
            self.assertIsNotNone(layer.attention.q_proj.weight.grad)
            self.assertIsNotNone(layer.attention.k_proj.weight.grad)
            self.assertTrue(torch.isfinite(layer.attention.q_proj.weight.grad).all())
            self.assertTrue(torch.isfinite(layer.attention.k_proj.weight.grad).all())
            self.assertGreater(layer.attention.q_proj.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(layer.attention.k_proj.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(layer.w_neigh.weight.grad.abs().sum().item(), 0.0)

    def test_mpin_training_beats_mean(self):
        rng = np.random.RandomState(42)
        N, D = 80, 8
        # Build strongly clustered/correlated synthetic data
        centers = rng.randn(4, D).astype(np.float32)
        labels = rng.randint(0, 4, size=N)
        X_full = centers[labels] + 0.08 * rng.randn(N, D).astype(np.float32)
        M = (rng.rand(N, D) > 0.5).astype(np.float32)
        for i in range(N):
            if M[i].sum() < 2:
                M[i, :2] = 1.0
        X_incomplete = X_full.copy()
        X_incomplete[M == 0] = np.nan

        X_eval, M_obs, M_missing = create_evaluation_mask(M, X_incomplete, 0.15, seed=42)
        edge_index, _ = build_knn_edge_index(X_eval, M_obs, k=8, symmetric=True)
        cfg = {
            "general": {"seed": 42, "device": "cpu"},
            "mpin": {
                "epochs": 80,
                "lr": 0.005,
                "val_split": 0.1,
                "mask_ratio": 0.2,
                "num_layers": 2,
                "hidden_dim": 32,
                "patience": 20,
                "loss": "mse",
            },
        }
        _, X_mpin, history = train_mpin(X_eval, M_obs, edge_index, cfg)
        X_mean = mean_imputation(X_eval, M_obs)

        self.assertFalse(np.isnan(X_mpin).any())
        self.assertGreater(len(history["train_losses"]), 0)
        obs_bool = M_obs.astype(bool)
        self.assertTrue(np.allclose(X_mpin[obs_bool], X_eval[obs_bool], atol=1e-5))

        mae_mpin = mae(X_full, X_mpin, M_missing)
        mae_mean = mae(X_full, X_mean, M_missing)
        self.assertLess(mae_mpin, mae_mean)

    def test_mpin_continuous_two_windows(self):
        rng = np.random.RandomState(7)
        N, D = 24, 6
        cfg = {
            "general": {"seed": 42, "device": "cpu"},
            "mpin": {
                "K": 5,
                "epochs": 15,
                "lr": 0.005,
                "hidden_dim": 16,
                "num_layers": 2,
                "val_split": 0.1,
                "patience": 10,
            },
            "mpin_dm": {
                "eta": 0.5,
                "max_keep": 10,
                "fine_tune_epochs": 8,
                "fine_tune_lr": 0.002,
            },
        }
        dm = MPINcontinuous(D=D, config=cfg)
        for _ in range(2):
            X_w = rng.randn(N, D).astype(np.float32)
            M_w = (rng.rand(N, D) > 0.4).astype(np.float32)
            M_w[:, 0] = 1.0  # guarantee observed positions and high-phi instances
            X_w[M_w == 0] = np.nan

            X_imp = dm.process_window(X_w, M_w)
            self.assertEqual(X_imp.shape, (N, D))
            self.assertFalse(np.isnan(X_imp).any())
            obs_bool = M_w.astype(bool)
            self.assertTrue(np.allclose(X_imp[obs_bool], X_w[obs_bool], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
