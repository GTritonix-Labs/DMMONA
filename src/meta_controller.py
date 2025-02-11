import torch
import torch.nn as nn
import torch.optim as optim

class MetaController(nn.Module):
    """
    MetaController is a reinforcement learning agent that adjusts
    training hyperparameters and guides architecture adaptation based on
    resource metrics.
    
    It receives a tensor of resource metrics (e.g., [cpu, memory])
    and outputs an adjustment signal vector.
    """
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=3, lr=0.001, use_normalization=True, dropout_p=0.1):
        """
        Initializes the MetaController.
        
        Args:
            input_dim (int): Dimensionality of the input metrics (default 2: CPU, memory).
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Dimensionality of the output adjustment signal.
            lr (float): Learning rate for the meta controller.
            use_normalization (bool): Whether to apply batch normalization to inputs.
            dropout_p (float): Dropout probability after the hidden layer.
        """
        super(MetaController, self).__init__()
        self.use_normalization = use_normalization
        if self.use_normalization:
            # Using BatchNorm1d for normalization.
            self.bn = nn.BatchNorm1d(input_dim)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Outputs values in the range (-1, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, resource_metrics):
        """
        Forward pass of the meta controller.
        
        Args:
            resource_metrics (torch.Tensor): Tensor of shape (batch_size, input_dim)
                representing current resource metrics (e.g., [cpu, memory]). 
                Values should be pre-scaled appropriately.
                
        Returns:
            torch.Tensor: Adjustment signals of shape (batch_size, output_dim).
        """
        # If normalization is enabled and batch size is > 1, apply batch normalization.
        # Otherwise, bypass normalization to avoid errors.
        if self.use_normalization and resource_metrics.size(0) > 1:
            x = self.bn(resource_metrics)
        else:
            x = resource_metrics
        return self.net(x)
    
    def update_policy(self, loss):
        """
        Updates the meta controller's policy using the provided loss (reward signal).
        
        Args:
            loss (torch.Tensor): A scalar loss value for backpropagation.
        
        Note: In a full RL setup, the loss here would be computed from a reward signal that
              balances performance improvements and resource efficiency.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, file_path):
        """
        Saves the meta controller's state_dict to the given file path.
        
        Args:
            file_path (str): Destination file path.
        """
        torch.save(self.state_dict(), file_path)
    
    def load_model(self, file_path):
        """
        Loads the meta controller's state_dict from the given file path.
        
        Args:
            file_path (str): Source file path.
        """
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set to evaluation mode after loading

# For testing purposes when running this file directly.
if __name__ == "__main__":
    # Create an instance of the MetaController.
    meta_ctrl = MetaController(input_dim=2, hidden_dim=16, output_dim=3, lr=0.001, use_normalization=True, dropout_p=0.1)
    
    # Simulate resource metrics input (batch size 1, e.g., 50% CPU and 13GB memory).
    resource_input = torch.tensor([[50.0, 13.0]])
    
    # Perform a forward pass.
    adjustment_signal = meta_ctrl(resource_input)
    print("Meta Controller adjustment signal (batch size 1):", adjustment_signal.detach().numpy())
    
    # Now simulate a batch of 3 examples.
    resource_input_batch = torch.tensor([[50.0, 13.0],
                                           [60.0, 12.5],
                                           [55.0, 13.2]])
    adjustment_signal_batch = meta_ctrl(resource_input_batch)
    print("Meta Controller adjustment signal (batch size 3):", adjustment_signal_batch.detach().numpy())
    
    # Demonstrate a dummy policy update.
    dummy_loss = torch.mean(adjustment_signal_batch ** 2)
    meta_ctrl.update_policy(dummy_loss)
    print("Policy updated successfully.")
    
    # Test save and load functionality.
    save_path = "meta_controller_state.pt"
    meta_ctrl.save_model(save_path)
    print(f"Model saved to {save_path}")
    
    new_meta_ctrl = MetaController(input_dim=2, hidden_dim=16, output_dim=3, lr=0.001, use_normalization=True, dropout_p=0.1)
    new_meta_ctrl.load_model(save_path)
    loaded_signal = new_meta_ctrl(resource_input_batch)
    print("Loaded model adjustment signal:", loaded_signal.detach().numpy())
