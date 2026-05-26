import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config_loader import load_config, logger
from src.feature_engineering.engineer import run_feature_engineering

def generate_static_plots(df_funding, df_gov, plots_dir):
    """Generates beautiful static visualizations for reports."""
    logger.info("Generating static plots...")
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    
    # 1. Year-wise Startup Funding (Private Ecosystem)
    plt.figure()
    funding_by_yr = df_funding.groupby("Year")["Funding_Amount_USD"].sum() / 1e6 # in millions
    sns.lineplot(x=funding_by_yr.index, y=funding_by_yr.values, marker="o", color="#4F46E5", linewidth=2.5)
    plt.title("Total Startup Funding in India (2010 - 2025)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Year")
    plt.ylabel("Funding Amount ($ Millions)")
    plt.tight_layout()
    plt_path = os.path.join(plots_dir, "funding_trends_yoy.png")
    plt.savefig(plt_path, dpi=150)
    plt.close()
    
    # 2. State-wise Government Registered Startups (Top 10 States)
    plt.figure()
    state_counts = df_gov.groupby("State")["Count"].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=state_counts.values, y=state_counts.index, hue=state_counts.index, palette="viridis", legend=False)
    plt.title("Top 10 Indian States by Registered Startups", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Total Registered Startups")
    plt.ylabel("State")
    plt.tight_layout()
    plt_path = os.path.join(plots_dir, "state_wise_distribution.png")
    plt.savefig(plt_path, dpi=150)
    plt.close()
    
    # 3. Sector-wise Market Share (Registered vs Private Funding)
    plt.figure()
    sector_funding = df_funding.groupby("Standard_Industry")["Funding_Amount_USD"].sum().sort_values(ascending=False)
    plt.pie(sector_funding.values, labels=sector_funding.index, autopct="%1.1f%%", 
            colors=sns.color_palette("Spectral", len(sector_funding)))
    plt.title("Private Startup Funding Share by Sector", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt_path = os.path.join(plots_dir, "sector_market_share.png")
    plt.savefig(plt_path, dpi=150)
    plt.close()

    # 4. Top Investors by Funding Count
    plt.figure()
    top_investors = df_funding["Investor"].value_counts().head(10)
    sns.barplot(x=top_investors.values, y=top_investors.index, hue=top_investors.index, palette="magma", legend=False)
    plt.title("Top 10 VCs by Number of Startup Investments", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Deal Count")
    plt.ylabel("Investor / VC")
    plt.tight_layout()
    plt_path = os.path.join(plots_dir, "investor_contributions.png")
    plt.savefig(plt_path, dpi=150)
    plt.close()
    
    logger.info("Static plots generated successfully.")

def calculate_state_development_index(df_gov):
    """Calculates custom State Development Index based on startup counts and YoY growth."""
    logger.info("Calculating State Development Index...")
    
    # Aggregations by State
    state_agg = df_gov.groupby("State").agg(
        Total_Startups=("Count", "sum"),
        Growth_Velocity=("Sector_Registration_YoY_Growth", "mean"),
    ).reset_index()
    
    # Standardize indicators to a 0-100 scale
    max_startups = state_agg["Total_Startups"].max()
    min_startups = state_agg["Total_Startups"].min()
    
    # Scale counts (logarithmic scale since Karnataka/Maharashtra have massive counts)
    state_agg["Count_Score"] = np.interp(
        np.log1p(state_agg["Total_Startups"]), 
        [np.log1p(min_startups), np.log1p(max_startups)], 
        [10, 100]
    )
    
    # Scale growth
    max_growth = state_agg["Growth_Velocity"].max()
    min_growth = state_agg["Growth_Velocity"].min()
    state_agg["Growth_Score"] = np.interp(
        state_agg["Growth_Velocity"], 
        [min_growth, max_growth], 
        [20, 100]
    )
    
    # Weighted Index: 65% Count, 35% Growth
    state_agg["State_Startup_Index"] = (state_agg["Count_Score"] * 0.65) + (state_agg["Growth_Score"] * 0.35)
    state_agg = state_agg.sort_values(by="State_Startup_Index", ascending=False).reset_index(drop=True)
    
    logger.info("State Development Index calculated successfully.")
    return state_agg

def run_analysis():
    """Main execution function to load processed data, run EDA and calculate indices."""
    config = load_config()
    proc_funding, proc_gov = run_feature_engineering()
    
    plots_dir = config["resolved_paths"]["static_plots_dir"]
    generate_static_plots(proc_funding, proc_gov, plots_dir)
    
    state_index = calculate_state_development_index(proc_gov)
    
    # Save State Index report to reports directory
    reports_dir = config["resolved_paths"]["reports_dir"]
    state_index_path = os.path.join(reports_dir, "state_development_index.csv")
    state_index.to_csv(state_index_path, index=False)
    logger.info(f"State Development Index report saved to {state_index_path}")
    
    return state_index

if __name__ == "__main__":
    s_idx = run_analysis()
    print("\nTop 5 States by State Development Index:")
    print(s_idx.head(5))
