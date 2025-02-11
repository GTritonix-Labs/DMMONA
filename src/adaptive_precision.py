import logging

def select_precision_mode(
    resource_state: dict,
    available_modes: list = ["fp32", "mixed", "quantized"],
    memory_thresholds: tuple = (10, 6)
) -> str:
    """
    Selects the computation precision mode based on the current resource state.

    Args:
        resource_state (dict): A dictionary with keys:
            - 'memory': The current memory usage in GB (float).
              (Optional: can include other keys like 'cpu' for future extension.)
        available_modes (list): List of available precision modes.
            Default is ["fp32", "mixed", "quantized"].
        memory_thresholds (tuple): A tuple (high_threshold, low_threshold) in GB.
            - If memory >= high_threshold, return "fp32".
            - If low_threshold <= memory < high_threshold, return "mixed".
            - If memory < low_threshold, return "quantized".
            Default thresholds are (10, 6) GB.
            
    Returns:
        str: The selected precision mode.
        
    Raises:
        ValueError: If 'memory' is not present in resource_state.
    """
    memory = resource_state.get("memory")
    if memory is None:
        raise ValueError("Resource state must include a 'memory' key (in GB).")
    
    high_threshold, low_threshold = memory_thresholds
    if memory >= high_threshold:
        selected_mode = "fp32"
    elif low_threshold <= memory < high_threshold:
        selected_mode = "mixed"
    else:
        selected_mode = "quantized"
    
    logging.info("Resource Monitor: Memory usage is %.2f GB. Selected precision mode: %s", memory, selected_mode)
    return selected_mode

# For testing purposes when running this file directly.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define test resource states.
    test_states = [
        {"memory": 13.45},  # Expected "fp32"
        {"memory": 9.0},    # Expected "mixed"
        {"memory": 5.5}     # Expected "quantized"
    ]
    
    for idx, state in enumerate(test_states, 1):
        mode = select_precision_mode(state)
        print(f"Test {idx} - Memory: {state['memory']} GB => Selected Precision Mode: {mode}")
