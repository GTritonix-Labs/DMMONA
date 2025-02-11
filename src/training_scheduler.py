import os
import sys
import logging
import time
import torch
import yaml

# Ensure the project root is added to sys.path so that 'src' modules are importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.resource_monitor import ResourceMonitor
from src.meta_controller import MetaController
from src.architecture_adaptation import adapt_architecture
from src.adaptive_precision import select_precision_mode

def training_loop(config: dict):
    """
    Main training loop for DMMONA.

    Integrates resource monitoring, meta controller, architecture adaptation,
    and precision mode selection to simulate a training process.

    Args:
        config (dict): Configuration parameters loaded from config/config.yaml.

    Returns:
        The final adapted model (for demonstration, a string or dict).
    """
    logger = logging.getLogger(__name__)

    # Retrieve training parameters.
    training_params = config.get("training", {})
    epochs = training_params.get("epochs", 10)
    batch_size = training_params.get("batch_size", 32)
    learning_rate = training_params.get("learning_rate", 0.001)
    log_interval = training_params.get("log_interval", 3)  # Used as moving_avg_window.
    adapt_interval = training_params.get("adapt_interval", 5)  # Trigger architecture adaptation only every N epochs.

    # Initialize the model using the base name from configuration.
    base_model_name = config.get("architecture", {}).get("initial_model", "baseline_cnn")
    current_model = base_model_name
    last_adaptation_action = None  # To track the last adaptation decision.
    logger.info("Initial model: %s", current_model)

    # Initialize modules.
    monitor = ResourceMonitor(interval=1, moving_avg_window=log_interval)
    meta_ctrl = MetaController(input_dim=2, hidden_dim=16, output_dim=3, lr=learning_rate)

    logger.info("Starting training loop for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)

        # Log resource metrics.
        resource_state = monitor.log_metrics()
        forecast = monitor.forecast_resources()
        logger.info("Resource metrics - CPU: %.2f%%, Memory: %.2f GB", resource_state["cpu"], resource_state["memory"])

        # Prepare input for meta controller.
        resource_tensor = torch.tensor([[forecast["cpu"], forecast["memory"]]])
        meta_signal = meta_ctrl(resource_tensor)
        logger.info("Meta signal: %s", meta_signal.detach().numpy())

        # Only adapt architecture every 'adapt_interval' epochs.
        if epoch % adapt_interval == 0:
            adapted_model = adapt_architecture(current_model, meta_signal)
            # For string models, extract the last action from the name.
            if isinstance(current_model, str) and "_" in current_model:
                last_action = current_model.split("_")[-1]
            else:
                last_action = None

            # Update only if the adaptation action is different from the last one.
            if adapted_model != current_model and (last_action is None or adapted_model.split("_")[-1] != last_action):
                logger.info("Model adapted from %s to %s", current_model, adapted_model)
                current_model = adapted_model
                last_adaptation_action = current_model.split("_")[-1]
            else:
                logger.info("Skipping architecture adaptation: no significant change or same as previous action.")
        else:
            logger.info("Skipping architecture adaptation this epoch (adapt_interval not reached).")

        # Prevent the model name from growing indefinitely.
        if isinstance(current_model, str) and (len(current_model) > 80 or current_model.count("expanded") > 3 or current_model.count("pruned") > 3):
            logger.warning("Model name is too long (%s). Resetting to base model name with suffix.", current_model)
            current_model = base_model_name + "_adapted"
            logger.info("Reset model name to: %s", current_model)

        # Select precision mode based on current resource state.
        precision_mode = select_precision_mode(resource_state)
        logger.info("Selected precision mode: %s", precision_mode)

        # Simulate a training step (replace with actual training code later).
        logger.info("Executing training step with batch size %d...", batch_size)
        time.sleep(1)  # Simulate training duration.

        # (Optional) Update meta controller based on a reward signal here.

    logger.info("Training loop complete. Final model: %s", current_model)
    return current_model

def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error("Error loading configuration: %s", e)
        raise

if __name__ == "__main__":
    # Set up basic logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Determine the configuration file path.
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    # Load configuration.
    config = load_config(config_path)
    
    # Start the training loop.
    final_model = training_loop(config)
    print("Final model:", final_model)
