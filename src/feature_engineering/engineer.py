import os
import pandas as pd
import numpy as np
from src.utils.config_loader import load_config, logger
from src.data_cleaning.cleaner import run_data_cleaning

# Historical Macro Economic Indicators for India (2016 - 2025)
MACRO_INDICATORS = {
    "GDP_Growth": {2016: 8.26, 2017: 6.80, 2018: 6.45, 2019: 3.87, 2020: -5.83, 2021: 9.05, 2022: 7.24, 2023: 7.60, 2024: 8.20, 2025: 7.00},
    "Inflation": {2016: 4.90, 2017: 3.33, 2018: 3.94, 2019: 3.73, 2020: 6.62, 2021: 5.13, 2022: 6.70, 2023: 5.70, 2024: 4.80, 2025: 4.50},
    "FDI_Inflow_Billion": {2016: 43.4, 2017: 44.8, 2018: 42.1, 2019: 50.6, 2020: 64.9, 2021: 59.6, 2022: 46.0, 2023: 41.0, 2024: 45.0, 2025: 48.0}
}

def engineer_funding_features(df_funding, config):
    """Engineers micro features for private funding data."""
    logger.info("Engineering features for private funding...")
    df = df_funding.copy()
    
    # 1. Company level aggregations (to capture growth and rounds)
    company_stats = df.groupby("Company").agg(
        Total_Funding_USD=("Funding_Amount_USD", "sum"),
        Avg_Funding_USD=("Funding_Amount_USD", "mean"),
        Funding_Rounds=("Company", "count"),
        First_Funding_Year=("Year", "min"),
        Latest_Funding_Year=("Year", "max")
    ).reset_index()
    
    # Merge company stats back
    df = df.merge(company_stats, on="Company", how="left")
    
    # 2. Time features
    # Generate mock month and quarter for richer visual timelines
    # We use a deterministic pseudo-random sequence based on company name length to keep it reproducible
    np.random.seed(42)
    df["Month"] = (df["Company"].apply(len) % 12) + 1
    df["Quarter"] = df["Month"].apply(lambda m: (m - 1) // 3 + 1)
    
    # Age of startup as of 2026 based on their first funding year
    df["Startup_Age_2026"] = 2026 - df["First_Funding_Year"]
    
    # 3. Macro indicators merge
    df["GDP_Growth"] = df["Year"].map(MACRO_INDICATORS["GDP_Growth"])
    df["Inflation"] = df["Year"].map(MACRO_INDICATORS["Inflation"])
    df["FDI_Inflow"] = df["Year"].map(MACRO_INDICATORS["FDI_Inflow_Billion"])
    
    # Backfill or fill missing years (pre-2016 or post-2025)
    df["GDP_Growth"] = df["GDP_Growth"].fillna(7.0)
    df["Inflation"] = df["Inflation"].fillna(4.5)
    df["FDI_Inflow"] = df["FDI_Inflow"].fillna(45.0)
    
    # 4. Sector Level Funding Accel (Year over Year standard sector funding sums)
    sector_yr_funding = df.groupby(["Standard_Industry", "Year"])["Funding_Amount_USD"].sum().reset_index()
    sector_yr_funding = sector_yr_funding.sort_values(["Standard_Industry", "Year"])
    sector_yr_funding["Prev_Year_Funding"] = sector_yr_funding.groupby("Standard_Industry")["Funding_Amount_USD"].shift(1)
    sector_yr_funding["Sector_Funding_YoY_Growth"] = (
        (sector_yr_funding["Funding_Amount_USD"] - sector_yr_funding["Prev_Year_Funding"]) 
        / sector_yr_funding["Prev_Year_Funding"]
    ).fillna(0.0)
    
    # Merge back YoY funding growth
    df = df.merge(sector_yr_funding[["Standard_Industry", "Year", "Sector_Funding_YoY_Growth"]], 
                  on=["Standard_Industry", "Year"], how="left")
    
    logger.info("Funding features engineered successfully.")
    return df

def engineer_gov_features(df_gov, config):
    """Engineers macro features for government registered startup data."""
    logger.info("Engineering features for government registered startups...")
    df = df_gov.copy()
    
    # 1. Geographic regions mapping from config
    regions = config.get("states_and_regions", {})
    
    def map_region(state):
        for reg, states in regions.items():
            if state in states:
                return reg
        return "Others"
        
    df["Region"] = df["State"].apply(map_region)
    
    # 2. Time based features
    # Map macro-economic indicators
    df["GDP_Growth"] = df["Year"].map(MACRO_INDICATORS["GDP_Growth"])
    df["Inflation"] = df["Year"].map(MACRO_INDICATORS["Inflation"])
    df["FDI_Inflow"] = df["Year"].map(MACRO_INDICATORS["FDI_Inflow_Billion"])
    
    df["GDP_Growth"] = df["GDP_Growth"].fillna(7.0)
    df["Inflation"] = df["Inflation"].fillna(4.5)
    df["FDI_Inflow"] = df["FDI_Inflow"].fillna(45.0)
    
    # 3. Macro industry registrations velocity (Year over Year registration counts)
    industry_yr_counts = df.groupby(["Standard_Industry", "Year"])["Count"].sum().reset_index()
    industry_yr_counts = industry_yr_counts.sort_values(["Standard_Industry", "Year"])
    industry_yr_counts["Prev_Year_Count"] = industry_yr_counts.groupby("Standard_Industry")["Count"].shift(1)
    industry_yr_counts["Sector_Registration_YoY_Growth"] = (
        (industry_yr_counts["Count"] - industry_yr_counts["Prev_Year_Count"]) 
        / industry_yr_counts["Prev_Year_Count"]
    ).fillna(0.0)
    
    df = df.merge(industry_yr_counts[["Standard_Industry", "Year", "Sector_Registration_YoY_Growth"]],
                  on=["Standard_Industry", "Year"], how="left")
                  
    logger.info("Government registered startup features engineered successfully.")
    return df

def run_feature_engineering():
    """Main execution function to load cleaned datasets, engineer features, and save processed file."""
    config = load_config()
    cleaned_funding, cleaned_gov = run_data_cleaning()
    
    proc_funding = engineer_funding_features(cleaned_funding, config)
    proc_gov = engineer_gov_features(cleaned_gov, config)
    
    # Save processed dataframes
    proc_dir = config["resolved_paths"]["processed_data_dir"]
    
    funding_out = os.path.join(proc_dir, "processed_funding.csv")
    gov_out = os.path.join(proc_dir, "processed_gov.csv")
    
    proc_funding.to_csv(funding_out, index=False)
    proc_gov.to_csv(gov_out, index=False)
    
    logger.info(f"Saved processed features datasets:\n  - Funding: {funding_out}\n  - Government: {gov_out}")
    return proc_funding, proc_gov

if __name__ == "__main__":
    df_pf, df_pg = run_feature_engineering()
    print("\nProcessed Funding Columns:", list(df_pf.columns))
    print("Processed Government Columns:", list(df_pg.columns))
    print("\nSample Processed Funding row:")
    print(df_pf.iloc[0])
