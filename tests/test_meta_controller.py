import os
import sys

# Add the project root to sys.path so that 'src' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import torch
from src.meta_controller import MetaController

class TestMetaController(unittest.TestCase):
    def setUp(self):
        # Create a MetaController instance with specified dimensions.
        self.meta_ctrl = MetaController(input_dim=2, hidden_dim=16, output_dim=3, lr=0.001)
    
    def test_forward_output_shape(self):
        """
        Test that the forward pass produces a tensor with the correct shape.
        For a batch of size 2, the output should have shape (2, 3).
        """
        input_tensor = torch.tensor([[50.0, 13.0], [60.0, 12.0]])
        output = self.meta_ctrl(input_tensor)
        self.assertEqual(output.shape, (2, 3))
    
    def test_forward_with_single_sample(self):
        """
        Test that the forward pass handles a single-sample input (batch size 1)
        without triggering normalization errors.
        """
        input_tensor = torch.tensor([[50.0, 13.0]])
        output = self.meta_ctrl(input_tensor)
        self.assertEqual(output.shape, (1, 3))
    
    def test_update_policy(self):
        """
        Test that update_policy changes the network parameters.
        We compare a copy of the parameters before and after a dummy update.
        """
        input_tensor = torch.tensor([[50.0, 13.0]])
        output_before = self.meta_ctrl(input_tensor).clone()
        dummy_loss = torch.mean(output_before ** 2)
        
        # Save a copy of parameters before update.
        params_before = [p.clone() for p in self.meta_ctrl.parameters()]
        self.meta_ctrl.update_policy(dummy_loss)
        params_after = list(self.meta_ctrl.parameters())
        
        # Check that at least one parameter has changed.
        changed = any(torch.any(torch.ne(pb, pa)) for pb, pa in zip(params_before, params_after))
        self.assertTrue(changed)

if __name__ == "__main__":
    unittest.main()
