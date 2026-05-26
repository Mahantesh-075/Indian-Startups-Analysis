import os
import logging
import yaml

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def setup_logging():
    """Sets up standard logger for the application."""
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(PROJECT_ROOT, "execution.log"), mode="w", encoding="utf-8")
        ]
    )
    return logging.getLogger("StartupAnalysis")

logger = setup_logging()

def load_config():
    """Loads configuration yaml and creates necessary directories."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Resolve all paths relative to the project root and create directories
    resolved_paths = {}
    for key, value in config["paths"].items():
        full_path = os.path.join(PROJECT_ROOT, value)
        resolved_paths[key] = full_path
        
        # Automatically create directory
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
            
    config["resolved_paths"] = resolved_paths
    return config

if __name__ == "__main__":
    cfg = load_config()
    print("Configuration loaded successfully!")
    print("Resolved Paths:")
    for k, v in cfg["resolved_paths"].items():
        print(f"  {k}: {v}")
