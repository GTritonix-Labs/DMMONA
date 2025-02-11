import logging
from typing import Union, List, Dict
import torch

# Set up a logger for this module if not already configured by the main logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def adapt_architecture(
    current_model: Union[str, Dict],
    meta_signal: Union[torch.Tensor, List[float], float],
    prune_threshold: float = -0.2,
    expand_threshold: float = 0.2
) -> Union[str, Dict]:
    """
    Adjust the network architecture based on the meta signal.
    
    This placeholder function simulates architecture adaptation by:
      - Pruning: if the mean meta signal is less than or equal to `prune_threshold`.
      - Expanding: if the mean meta signal is greater than or equal to `expand_threshold`.
      - No change: if the meta signal falls between these thresholds.
      
    Args:
        current_model (Union[str, Dict]): The current model representation. This can be a string
            identifier or a dictionary with model details.
        meta_signal (Union[torch.Tensor, List[float], float]): The meta signal output from the meta controller.
        prune_threshold (float, optional): Threshold below which the model is "pruned". Defaults to -0.2.
        expand_threshold (float, optional): Threshold above which the model is "expanded". Defaults to 0.2.
        
    Returns:
        Union[str, Dict]: The adapted model. For a string, an appropriate suffix is appended;
                          for a dict, an 'adaptation' key is added (and the 'name' field updated, if present).
    """
    
    # Aggregate the meta signal into a single float value.
    if isinstance(meta_signal, list):
        signal_value = sum(meta_signal) / len(meta_signal)
    elif isinstance(meta_signal, (float, int)):
        signal_value = float(meta_signal)
    elif isinstance(meta_signal, torch.Tensor):
        signal_value = meta_signal.mean().item()
    else:
        raise ValueError("Unsupported type for meta_signal. Must be tensor, list, or float.")
    
    logger.debug("Aggregated meta_signal value: %f", signal_value)
    logger.debug("Prune threshold: %f, Expand threshold: %f", prune_threshold, expand_threshold)
    
    # Determine adaptation action based on inclusive threshold comparisons.
    if signal_value <= prune_threshold:
        action = "pruned"
        logger.info("Architecture adaptation: Pruning layers (meta_signal: %f).", signal_value)
    elif signal_value >= expand_threshold:
        action = "expanded"
        logger.info("Architecture adaptation: Expanding network capacity (meta_signal: %f).", signal_value)
    else:
        action = "unchanged"
        logger.info("Architecture adaptation: No change needed (meta_signal: %f).", signal_value)
    
    # Apply adaptation to the current_model
    if isinstance(current_model, dict):
        # If current_model is a dict, update a 'name' key and add an 'adaptation' key.
        adapted_model = current_model.copy()
        adapted_model["adaptation"] = action
        if "name" in adapted_model:
            adapted_model["name"] += f"_{action}"
        else:
            adapted_model["name"] = f"model_{action}"
    elif isinstance(current_model, str):
        # Append the action to the model name if an adaptation is needed.
        adapted_model = f"{current_model}_{action}" if action != "unchanged" else current_model
    else:
        raise ValueError("Unsupported type for current_model. Must be str or dict.")
    
    return adapted_model

# For testing purposes when running this file directly.
if __name__ == "__main__":
    # Example 1: Using a simple string as the model.
    current_model_str = "baseline_cnn"
    meta_signal_tensor = torch.tensor([[0.3, 0.1, 0.2]])  # This should trigger "expanded".
    adapted_model_str = adapt_architecture(current_model_str, meta_signal_tensor)
    print("Adapted Model (string):", adapted_model_str)
    
    # Example 2: Using a dictionary to represent the model.
    current_model_dict = {"name": "baseline_cnn", "layers": 5}
    meta_signal_list = [-0.3, -0.1, -0.2]  # This should trigger "pruned".
    adapted_model_dict = adapt_architecture(current_model_dict, meta_signal_list)
    print("Adapted Model (dict):", adapted_model_dict)
