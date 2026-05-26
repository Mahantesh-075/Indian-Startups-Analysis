import os
import pandas as pd
import numpy as np
from src.utils.config_loader import load_config, logger
from src.data_collection.data_loader import run_data_collection

def clean_funding_data(df, config):
    """Cleans and standardizes the private funding dataset."""
    logger.info("Cleaning private funding dataset...")
    df_clean = df.copy()
    
    # 1. Strip whitespace from text columns
    text_cols = ["Company", "Industry", "Investor", "City"]
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
    # 2. Normalize City names
    city_mapping = {
        "bengaluru": "Bangalore",
        "bangalore": "Bangalore",
        "new delhi": "Delhi",
        "delhi": "Delhi",
        "noida": "Noida",
        "gurugram": "Gurgaon",
        "gurgaon": "Gurgaon",
        "mumbai": "Mumbai",
        "pune": "Pune",
        "chennai": "Chennai",
        "hyderabad": "Hyderabad"
    }
    
    def normalize_city(city_val):
        city_lower = str(city_val).lower().strip()
        return city_mapping.get(city_lower, city_val)
        
    df_clean["City"] = df_clean["City"].apply(normalize_city)
    
    # 3. Map Industries to Standard Sectors
    ind_mapping = config.get("industry_mapping", {})
    
    def map_industry(ind_val):
        ind_str = str(ind_val).strip()
        # Direct lookup
        if ind_str in ind_mapping:
            return ind_mapping[ind_str]
        # Case insensitive lookup
        for k, v in ind_mapping.items():
            if k.lower() == ind_str.lower():
                return v
        return "Others"
        
    df_clean["Standard_Industry"] = df_clean["Industry"].apply(map_industry)
    
    # 4. Handle Funding outliers & null values (if any)
    # Funding_Amount_USD: check for non-positive or null
    df_clean["Funding_Amount_USD"] = pd.to_numeric(df_clean["Funding_Amount_USD"], errors="coerce")
    median_funding = df_clean["Funding_Amount_USD"].median()
    df_clean["Funding_Amount_USD"] = df_clean["Funding_Amount_USD"].fillna(median_funding)
    
    # Remove duplicates
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Dropped {initial_rows - df_clean.shape[0]} duplicate rows from funding dataset.")
    
    logger.info(f"Funding dataset cleaned: {df_clean.shape[0]} rows remaining.")
    return df_clean

def clean_gov_data(df, config):
    """Cleans and standardizes the government registered startups dataset."""
    logger.info("Cleaning government startup dataset...")
    df_clean = df.copy()
    
    # Drop "S No." since it's just a serial number
    if "S No." in df_clean.columns:
        df_clean = df_clean.drop(columns=["S No."])
        
    # 1. Clean string columns
    df_clean["State"] = df_clean["State"].astype(str).str.strip()
    df_clean["Industry"] = df_clean["Industry"].astype(str).str.strip()
    
    # Standardize State capitalization and names
    # State names mapping for consistency
    state_aliases = {
        "andaman & nicobar islands": "Andaman and Nicobar Islands",
        "andhra pradesh": "Andhra Pradesh",
        "arunachal pradesh": "Arunachal Pradesh",
        "assam": "Assam",
        "bihar": "Bihar",
        "chandigarh": "Chandigarh",
        "chhattisgarh": "Chhattisgarh",
        "delhi": "Delhi",
        "goa": "Goa",
        "gujarat": "Gujarat",
        "haryana": "Haryana",
        "himachal pradesh": "Himachal Pradesh",
        "jammu & kashmir": "Jammu and Kashmir",
        "jammu and kashmir": "Jammu and Kashmir",
        "jharkhand": "Jharkhand",
        "karnataka": "Karnataka",
        "kerala": "Kerala",
        "ladakh": "Ladakh",
        "lakshadweep": "Lakshadweep",
        "madhya pradesh": "Madhya Pradesh",
        "maharashtra": "Maharashtra",
        "manipur": "Manipur",
        "meghalaya": "Meghalaya",
        "mizoram": "Mizoram",
        "nagaland": "Nagaland",
        "odisha": "Odisha",
        "puducherry": "Puducherry",
        "punjab": "Punjab",
        "rajasthan": "Rajasthan",
        "sikkim": "Sikkim",
        "tamil nadu": "Tamil Nadu",
        "telangana": "Telangana",
        "tripura": "Tripura",
        "uttar pradesh": "Uttar Pradesh",
        "uttarakhand": "Uttarakhand",
        "west bengal": "West Bengal"
    }
    
    def normalize_state(state_val):
        state_lower = str(state_val).lower().strip()
        return state_aliases.get(state_lower, state_val.title())
        
    df_clean["State"] = df_clean["State"].apply(normalize_state)
    
    # 2. Map Industry to Standard Sector
    ind_mapping = config.get("industry_mapping", {})
    
    def map_industry(ind_val):
        ind_str = str(ind_val).strip()
        if ind_str in ind_mapping:
            return ind_mapping[ind_str]
        for k, v in ind_mapping.items():
            if k.lower() == ind_str.lower():
                return v
        return "Others"
        
    df_clean["Standard_Industry"] = df_clean["Industry"].apply(map_industry)
    
    # 3. Year column validation
    df_clean["Year"] = pd.to_numeric(df_clean["Year"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Year"])
    df_clean["Year"] = df_clean["Year"].astype(int)
    
    # 4. Count column clean
    df_clean["Count"] = pd.to_numeric(df_clean["Count"], errors="coerce")
    df_clean["Count"] = df_clean["Count"].fillna(1).astype(int)
    
    # 5. Last Update column formatting
    if "Last Update" in df_clean.columns:
        df_clean["Last Update"] = pd.to_datetime(df_clean["Last Update"], errors="coerce")
        
    # Deduplicate
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    logger.info(f"Dropped {initial_rows - df_clean.shape[0]} duplicate rows from government dataset.")
    
    logger.info(f"Government dataset cleaned: {df_clean.shape[0]} rows remaining.")
    return df_clean

def run_data_cleaning():
    """Main pipeline function to load and clean all datasets, saving to data/cleaned/."""
    config = load_config()
    df_funding, df_gov = run_data_collection()
    
    cleaned_funding = clean_funding_data(df_funding, config)
    cleaned_gov = clean_gov_data(df_gov, config)
    
    # Save cleaned files
    cleaned_dir = config["resolved_paths"]["cleaned_data_dir"]
    
    funding_out = os.path.join(cleaned_dir, config["files"]["funding_cleaned"])
    gov_out = os.path.join(cleaned_dir, config["files"]["gov_cleaned"])
    
    cleaned_funding.to_csv(funding_out, index=False)
    cleaned_gov.to_csv(gov_out, index=False)
    
    logger.info(f"Saved cleaned datasets:\n  - Funding: {funding_out}\n  - Government: {gov_out}")
    return cleaned_funding, cleaned_gov

if __name__ == "__main__":
    df_cf, df_cg = run_data_cleaning()
    print("\nCleaned Funding shape:", df_cf.shape)
    print("Cleaned Government shape:", df_cg.shape)
    print("\nCleaned standard industry value counts in Funding:\n", df_cf["Standard_Industry"].value_counts())
    print("\nCleaned standard industry value counts in Gov:\n", df_cg["Standard_Industry"].value_counts())
