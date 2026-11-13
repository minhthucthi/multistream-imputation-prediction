"""
Unit tests for data loader, similarity graph, metrics, and MPIN layer forward pass.
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data.preprocessor import create_data_chunk, compute_importance_score
from models.similarity_graph import build_similarity_graph
from models.mpin import MPIN
from evaluation.metrics import mae, mre


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
        adj = build_similarity_graph(self.X, self.M, k=2, metric="euclidean")
        self.assertEqual(adj.shape, (4, 4))
        # Self-connections or neighbors should exist
        self.assertTrue(np.all(adj >= 0))

    def test_mpin_forward(self):
        model = MPIN(num_attributes=3, num_layers=2)
        adj = torch.eye(4)
        x_tensor = torch.tensor(np.nan_to_num(self.X, nan=0.0), dtype=torch.float32)
        m_tensor = torch.tensor(self.M, dtype=torch.float32)

        out = model(x_tensor, adj, m_tensor)
        self.assertEqual(out.shape, (4, 3))
        # Verify bound condition: observed values must remain intact in output
        observed_mask = (m_tensor == 1.0)
        self.assertTrue(torch.allclose(out[observed_mask], x_tensor[observed_mask], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
