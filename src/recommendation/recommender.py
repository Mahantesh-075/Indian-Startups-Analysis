import os
import pandas as pd
import numpy as np
from src.utils.config_loader import load_config, logger
from src.sentiment.analyzer import get_sector_sentiment_stats

# Real-world Government Schemes Database
GOV_SCHEMES = [
    {
        "name": "Startup India Seed Fund Scheme (SISFS)",
        "description": "Provides financial assistance to early-stage startups for proof of concept, prototype development, product trials, market entry, and commercialization.",
        "benefits": "Grants up to INR 20 Lakhs for prototype and up to INR 50 Lakhs for market entry/scaling through incubators.",
        "sectors": ["AI & DeepTech", "DeepTech & Aerospace", "HealthTech & BioTech", "CleanTech & Green Energy", "FoodTech & AgriTech"],
        "states": "All"
    },
    {
        "name": "SAMRIDH Scheme (MeitY)",
        "description": "Aims to support startup accelerators to facilitate funding, mentoring, and scale-up support for software startups in India.",
        "benefits": "Matching funding up to INR 40 Lakhs from the government, international accelerator linkages, and custom corporate mentorship.",
        "sectors": ["SaaS & Enterprise", "AI & DeepTech", "FinTech", "EdTech", "Media & Entertainment"],
        "states": "All"
    },
    {
        "name": "ASPIRE (Scheme for Promotion of Innovation, Rural Industries & Entrepreneurship)",
        "description": "Focuses on setting up a network of technology centers and incubation centers to accelerate entrepreneurship in rural areas.",
        "benefits": "Grants ranging from INR 10 Lakhs to 1 Crore for setting up livelihood business incubators and agro-processing clusters.",
        "sectors": ["FoodTech & AgriTech", "Industrial & Chemicals", "Logistics & Mobility"],
        "states": "All"
    },
    {
        "name": "Atal New India Challenges (ANIC)",
        "description": "Grants to support technology-driven innovations that solve critical problems of national importance and societal relevance.",
        "benefits": "Funding grants of up to INR 1 Crore for selected technology prototypes and commercialization support.",
        "sectors": ["CleanTech & Green Energy", "Logistics & Mobility", "DeepTech & Aerospace", "Real Estate & Construction"],
        "states": "All"
    },
    {
        "name": "Pradhan Mantri Mudra Yojana (Mudra Loans)",
        "description": "Offers institutional credit to micro/small business enterprises and early non-corporate start-ups.",
        "benefits": "Collateral-free working capital or term loans up to INR 10 Lakhs under Shishu, Kishor, and Tarun categories.",
        "sectors": ["E-Commerce & Retail", "Media & Entertainment", "SaaS & Enterprise"],
        "states": "All"
    },
    # State Specific Policies
    {
        "name": "Karnataka Startup Policy - Elevate Idea2PoC",
        "description": "A marquee grant-in-aid scheme by the Government of Karnataka to identify and fund innovative tech start-ups.",
        "benefits": "Non-dilutive grant up to INR 50 Lakhs, free access to state-sponsored incubators, and state-backed networking events.",
        "sectors": "All",
        "states": ["Karnataka"]
    },
    {
        "name": "Maharashtra Startup Week & Innovation Grants",
        "description": "Initiative by MSInS (Maharashtra State Innovation Society) to match innovative startups with state government projects.",
        "benefits": "Work orders up to INR 15 Lakhs for pilot projects with government departments and fast-track regulatory clearances.",
        "sectors": "All",
        "states": ["Maharashtra"]
    },
    {
        "name": "UP Startup Policy - Seed Grant & Incubator Support",
        "description": "Financial assistance by the Government of Uttar Pradesh to foster entrepreneurship and grassroots innovations.",
        "benefits": "Seed grant of INR 5 Lakhs (non-dilutive), marketing assistance up to INR 7.5 Lakhs, and monthly stipend of INR 17,500.",
        "sectors": "All",
        "states": ["Uttar Pradesh"]
    },
    {
        "name": "Telangana State T-HUB Incubation and Grants",
        "description": "Access to the world's largest startup incubator T-Hub in Hyderabad, alongside innovation grants.",
        "benefits": "Seed funding grants up to INR 15 Lakhs, state co-working access, and fast-tracked access to local angel networks.",
        "sectors": "All",
        "states": ["Telangana"]
    }
]

def calculate_opportunity_score(sector, df_funding, df_gov, config):
    """Calculates custom Opportunity Score (0-100) based on multiple parameters."""
    logger.info(f"Calculating opportunity score for sector: {sector}")
    
    # 1. Registration growth rate (Velocity)
    sector_gov = df_gov[df_gov["Standard_Industry"] == sector]
    reg_growth = sector_gov["Sector_Registration_YoY_Growth"].mean() if not sector_gov.empty else 0.0
    
    # 2. Funding growth velocity
    sector_fund = df_funding[df_funding["Standard_Industry"] == sector]
    fund_growth = sector_fund["Sector_Funding_YoY_Growth"].mean() if not sector_fund.empty else 0.0
    
    # 3. Market size / Registered startup count density
    total_reg = df_gov["Count"].sum()
    sector_reg = sector_gov["Count"].sum() if not sector_gov.empty else 1
    reg_density_score = (sector_reg / total_reg) * 100 if total_reg > 0 else 0.0
    
    # 4. Sentiment score (NLTK compound)
    sent_stats, _ = get_sector_sentiment_stats(sector)
    sent_score = sent_stats["compound"]
    
    # Standardize indicators into 0-100 scales
    s_growth = np.interp(reg_growth, [-0.5, 1.5], [10, 100])
    s_funding = np.interp(fund_growth, [-0.5, 2.0], [20, 100])
    s_density = np.interp(np.log1p(reg_density_score), [0, np.log1p(50)], [30, 100])
    s_sentiment = np.interp(sent_score, [-1.0, 1.0], [0, 100])
    
    weights = config.get("opportunity_weights", {
        "growth_rate": 0.35,
        "funding_velocity": 0.30,
        "registered_density": 0.15,
        "investor_interest": 0.20
    })
    
    score = (
        s_growth * weights["growth_rate"] +
        s_funding * weights["funding_velocity"] +
        s_density * weights["registered_density"] +
        s_sentiment * weights["investor_interest"]
    )
    
    # Ensure range 0-100
    score = min(max(score, 0.0), 100.0)
    
    logger.info(f"Opportunity score for {sector} is {score:.2f}")
    return {
        "opportunity_score": round(score, 1),
        "metrics": {
            "registration_growth": round(reg_growth * 100, 1),
            "funding_growth": round(fund_growth * 100, 1),
            "sentiment": sent_stats["sentiment"],
            "sentiment_score": round(sent_score, 2),
            "startup_density_pct": round(reg_density_score, 2)
        }
    }

def get_recommended_schemes(sector, state):
    """Recommends specific government schemes based on sector and state inputs."""
    recommended = []
    
    for scheme in GOV_SCHEMES:
        match_sector = False
        match_state = False
        
        # Check Sector
        if scheme["sectors"] == "All" or sector in scheme["sectors"]:
            match_sector = True
            
        # Check State
        if scheme["states"] == "All" or state in scheme["states"]:
            match_state = True
            
        if match_sector and match_state:
            recommended.append(scheme)
            
    return recommended

def get_investor_matches(sector, df_funding):
    """Matches top VCs and check sizes for a given sector."""
    sector_deals = df_funding[df_funding["Standard_Industry"] == sector]
    
    if sector_deals.empty:
        # Return fallback general top investors
        top_investors = df_funding.groupby("Investor").agg(
            deals=("Company", "count"),
            avg_check_usd=("Funding_Amount_USD", "mean")
        ).sort_values("deals", ascending=False).head(5).reset_index()
    else:
        top_investors = sector_deals.groupby("Investor").agg(
            deals=("Company", "count"),
            avg_check_usd=("Funding_Amount_USD", "mean")
        ).sort_values("deals", ascending=False).head(5).reset_index()
        
    top_investors["avg_check_usd"] = top_investors["avg_check_usd"].round(0).astype(int)
    return top_investors.to_dict("records")

def get_startup_similarity_matches(sector, city, budget, df_funding):
    """Finds top 3 similar startups based on sector, city, and check sizes."""
    df_sec = df_funding[df_funding["Standard_Industry"] == sector].copy()
    
    if df_sec.empty:
        df_sec = df_funding.copy()
        
    # Calculate similarity metrics
    # City matching gives higher score
    df_sec["city_match"] = (df_sec["City"] == city).astype(int)
    
    # Distance in budget (closer is more similar)
    budget_diff = np.abs(df_sec["Funding_Amount_USD"] - budget)
    max_diff = budget_diff.max() if budget_diff.max() > 0 else 1.0
    df_sec["budget_score"] = 1.0 - (budget_diff / max_diff)
    
    # Final Score: 60% budget, 40% city match
    df_sec["similarity_score"] = (df_sec["budget_score"] * 0.6) + (df_sec["city_match"] * 0.4)
    similar = df_sec.sort_values("similarity_score", ascending=False).head(3)
    
    return similar[["Company", "City", "Standard_Industry", "Funding_Amount_USD", "Investor", "Year"]].to_dict("records")

if __name__ == "__main__":
    cfg = load_config()
    proc_funding = pd.read_csv(os.path.join(cfg["resolved_paths"]["processed_data_dir"], "processed_funding.csv"))
    proc_gov = pd.read_csv(os.path.join(cfg["resolved_paths"]["processed_data_dir"], "processed_gov.csv"))
    
    opp = calculate_opportunity_score("AI & DeepTech", proc_funding, proc_gov, cfg)
    print("Opportunity Score (AI & DeepTech):", opp)
    
    print("\nRecommended Schemes (AI & DeepTech in Karnataka):")
    for s in get_recommended_schemes("AI & DeepTech", "Karnataka"):
        print(f" - {s['name']}")
        
    print("\nInvestor Matches for AI & DeepTech:")
    print(get_investor_matches("AI & DeepTech", proc_funding)[:2])
    
    print("\nSimilarity Matches for AI & DeepTech in Bangalore (Budget 5M USD):")
    print(get_startup_similarity_matches("AI & DeepTech", "Bangalore", 5000000, proc_funding))
