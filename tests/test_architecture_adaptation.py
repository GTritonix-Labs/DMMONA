import os
import sys
import unittest
import torch

# Add the project root to sys.path so that 'src' modules can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.architecture_adaptation import adapt_architecture

class TestArchitectureAdaptation(unittest.TestCase):
    def test_adapt_string_expansion(self):
        """
        Test that a meta signal with a mean >= expand threshold
        causes a string model to be marked as 'expanded'.
        """
        current_model = "baseline_cnn"
        # Meta signal: average = (0.3 + 0.1 + 0.2) / 3 = 0.2 (inclusive threshold triggers expansion)
        meta_signal = torch.tensor([[0.3, 0.1, 0.2]])
        adapted_model = adapt_architecture(current_model, meta_signal)
        self.assertEqual(adapted_model, "baseline_cnn_expanded")

    def test_adapt_string_pruning(self):
        """
        Test that a meta signal with a mean <= prune threshold
        causes a string model to be marked as 'pruned'.
        """
        current_model = "baseline_cnn"
        # Meta signal: average = (-0.3 + -0.1 + -0.2) / 3 = -0.2 (inclusive threshold triggers pruning)
        meta_signal = torch.tensor([[-0.3, -0.1, -0.2]])
        adapted_model = adapt_architecture(current_model, meta_signal)
        self.assertEqual(adapted_model, "baseline_cnn_pruned")

    def test_adapt_string_unchanged(self):
        """
        Test that a meta signal within the thresholds leaves the model unchanged.
        """
        current_model = "baseline_cnn"
        # Meta signal: average = (0.0 + 0.1 + 0.0) / 3 = 0.033..., within (-0.2, 0.2)
        meta_signal = torch.tensor([[0.0, 0.1, 0.0]])
        adapted_model = adapt_architecture(current_model, meta_signal)
        self.assertEqual(adapted_model, "baseline_cnn")

    def test_adapt_dict(self):
        """
        Test that a dictionary model representation is updated correctly.
        """
        current_model = {"name": "baseline_cnn", "layers": 5}
        # Meta signal triggering expansion (average = 0.5)
        meta_signal = torch.tensor([[0.5, 0.5, 0.5]])
        adapted_model = adapt_architecture(current_model, meta_signal)
        self.assertEqual(adapted_model["adaptation"], "expanded")
        self.assertIn("expanded", adapted_model["name"])

    def test_invalid_meta_signal(self):
        """
        Test that providing an invalid meta signal (e.g., None) raises a ValueError.
        """
        current_model = "baseline_cnn"
        with self.assertRaises(ValueError):
            adapt_architecture(current_model, None)

if __name__ == "__main__":
    unittest.main()
