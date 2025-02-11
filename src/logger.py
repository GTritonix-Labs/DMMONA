import logging

def setup_logger(log_file: str = "dmmona.log", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger for the DMMONA project.
    
    Logs messages to both the console and a specified file.
    
    Args:
        log_file (str): The file path to store log messages (default: "dmmona.log").
        level (int): Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger with a specific name.
    logger = logging.getLogger("DMMONA")
    logger.setLevel(level)
    
    # Create a formatter that includes the timestamp, log level, and message.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and configure the console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Create and configure the file handler.
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Clear any existing handlers to avoid duplicate logging.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Add both handlers to the logger.
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    # For testing purposes, set up the logger and log a test message.
    logger = setup_logger()
    logger.info("Logger is set up successfully.")
