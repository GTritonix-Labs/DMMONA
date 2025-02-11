import os
import sys

# Ensure the project root is in sys.path so that the 'src' package is importable.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import argparse
import logging
from src.training_scheduler import training_loop
from src.logger import setup_logger

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file.
    
    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error("Error loading configuration file: %s", e)
        sys.exit(1)

def parse_arguments():
    """
    Parse command-line arguments for the configuration file path.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="DMMONA: Dynamic Multi-Objective Meta-Optimizer")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="Path to the configuration YAML file")
    return parser.parse_args()

def print_banner():
    """
    Print a startup banner with project name and version.
    """
    banner = r"""
============================================
  DMMONA – Dynamic Multi-Objective Meta-Optimizer
           Version: 1.0.0
============================================
    """
    print(banner)

def override_config_with_env(config: dict) -> dict:
    """
    Override configuration parameters using environment variables, if set.
    
    For example, the number of epochs can be overridden by DMMONA_EPOCHS.
    
    Args:
        config (dict): Original configuration dictionary.
    
    Returns:
        dict: Updated configuration dictionary.
    """
    if os.getenv("DMMONA_EPOCHS"):
        try:
            config["training"]["epochs"] = int(os.getenv("DMMONA_EPOCHS"))
        except ValueError:
            logging.warning("Invalid value for DMMONA_EPOCHS; using the value from the config file.")
    # Additional overrides can be added here.
    return config

def main():
    print_banner()
    
    # Parse command-line arguments.
    args = parse_arguments()
    config_path = args.config if args.config else os.path.join(project_root, "config", "config.yaml")
    
    # Load configuration and override with environment variables if provided.
    config = load_config(config_path)
    config = override_config_with_env(config)
    
    # Set up logging.
    logger = setup_logger()
    logger.info("Configuration loaded successfully from %s", config_path)
    logger.info("Starting training loop...")
    
    # Start the training loop.
    try:
        training_loop(config)
    except Exception as e:
        logger.exception("An error occurred during training: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
