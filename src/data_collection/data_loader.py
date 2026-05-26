import os
import shutil
import pandas as pd
from src.utils.config_loader import load_config, logger, PROJECT_ROOT

def copy_raw_files(config):
    """Copies the raw CSV files from project root to data/raw directory."""
    raw_dir = config["resolved_paths"]["raw_data_dir"]
    
    # Files to copy
    files_to_copy = [config["files"]["funding_raw"], config["files"]["gov_raw"]]
    
    for filename in files_to_copy:
        src_path = os.path.join(PROJECT_ROOT, filename)
        dest_path = os.path.join(raw_dir, filename)
        
        if os.path.exists(src_path):
            # Only copy if destination doesn't exist or size is different
            if not os.path.exists(dest_path) or os.path.getsize(src_path) != os.path.getsize(dest_path):
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied {filename} to raw data folder: {dest_path}")
            else:
                logger.info(f"File {filename} already exists in raw data folder.")
        else:
            if not os.path.exists(dest_path):
                logger.error(f"Source file {filename} not found at {src_path} and does not exist in destination.")
                raise FileNotFoundError(f"Source file {filename} not found.")
            else:
                logger.warning(f"Source file {filename} not found at root, but already exists in raw folder.")

def load_raw_funding(config):
    """Loads raw startup funding data into DataFrame."""
    raw_dir = config["resolved_paths"]["raw_data_dir"]
    file_path = os.path.join(raw_dir, config["files"]["funding_raw"])
    
    logger.info(f"Loading raw funding data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded funding data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def load_raw_gov(config):
    """Loads raw government registered startup data.
    
    Note: gov-data-final.csv is actually an Excel spreadsheet (.xlsx format).
    We load it using pandas read_excel with openpyxl engine.
    """
    raw_dir = config["resolved_paths"]["raw_data_dir"]
    file_path = os.path.join(raw_dir, config["files"]["gov_raw"])
    
    logger.info(f"Loading raw government data from {file_path}")
    try:
        # Load as Excel workbook since it is openxml
        df = pd.read_excel(file_path, engine="openpyxl")
        logger.info(f"Loaded government Excel data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading government data as Excel: {e}")
        raise e

def run_data_collection():
    """Main function to run collection and copy operations."""
    config = load_config()
    copy_raw_files(config)
    df_funding = load_raw_funding(config)
    df_gov = load_raw_gov(config)
    return df_funding, df_gov

if __name__ == "__main__":
    df_f, df_g = run_data_collection()
    print("\nSample Funding Data:")
    print(df_f.head(2))
    print("\nSample Government Data:")
    print(df_g.head(2))
