import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure src modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config_loader import load_config, PROJECT_ROOT
from src.sentiment.analyzer import get_sector_sentiment_stats, analyze_text
from src.recommendation.recommender import (
    calculate_opportunity_score,
    get_recommended_schemes,
    get_investor_matches,
    get_startup_similarity_matches
)
from src.visualization.charts import (
    plot_startup_growth_yoy,
    plot_funding_trends,
    plot_state_distribution,
    plot_sector_treemap,
    plot_radar_metrics,
    plot_investor_network,
    plot_forecast_chart,
    COLOR_PALETTE
)

# ----------------- STREAMLIT PAGE SETUP -----------------
st.set_page_config(
    page_title="Indian Startup Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration and datasets
@st.cache_resource
def get_resources():
    config = load_config()
    proc_dir = config["resolved_paths"]["processed_data_dir"]
    models_dir = config["resolved_paths"]["models_dir"]
    
    df_funding = pd.read_csv(os.path.join(proc_dir, "processed_funding.csv"))
    df_gov = pd.read_csv(os.path.join(proc_dir, "processed_gov.csv"))
    state_index = pd.read_csv(os.path.join(config["resolved_paths"]["reports_dir"], "state_development_index.csv"))
    
    # Load Models
    success_data = joblib.load(os.path.join(models_dir, config["files"]["success_model"]))
    funding_data = joblib.load(os.path.join(models_dir, config["files"]["funding_model"]))
    forecast_data = joblib.load(os.path.join(models_dir, config["files"]["forecast_model"]))
    
    return config, df_funding, df_gov, state_index, success_data, funding_data, forecast_data

try:
    config, df_funding, df_gov, state_index, success_data, funding_data, forecast_data = get_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}. Please run 'python main.py' first to train models.")
    st.stop()

# Custom CSS for high-end Premium Dark Glassmorphism Styling
st.markdown(f"""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    h1, h2, h3 {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }}
    
    /* Background override */
    .stApp {{
        background-color: {COLOR_PALETTE["background"]};
        color: {COLOR_PALETTE["text"]};
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: #0B0F19;
        border-right: 1px solid #1E293B;
    }}
    
    /* Custom metric cards */
    .metric-card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }}
    .metric-label {{
        font-size: 14px;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }}
    .metric-value {{
        font-size: 32px;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(to right, #FFFFFF, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-sub {{
        font-size: 12px;
        color: {COLOR_PALETTE["secondary"]};
        font-weight: 500;
        margin-top: 6px;
    }}
    
    /* SWOT Box Styles */
    .swot-box {{
        padding: 16px;
        border-radius: 12px;
        height: 100%;
        color: #F8FAFC;
    }}
    .swot-s {{ background-color: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); }}
    .swot-w {{ background-color: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); }}
    .swot-o {{ background-color: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); }}
    .swot-t {{ background-color: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); }}
    
    .swot-title {{
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }}
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: white; margin: 0; font-family: Outfit;'>Ecosystem IQ ⚡</h2>
        <p style='color: #64748B; font-size: 13px; font-weight: 500;'>Indian Startup Intelligence</p>
    </div>
""", unsafe_allow_html=True)

tabs = ["Ecosystem Explorer", "AI Pitch Evaluator & Simulator", "Entrepreneur Co-Pilot", "Market Projections & Signals", "Sentiment & News Analyzer"]
selected_tab = st.sidebar.radio("Navigate Dashboard", tabs)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
    <div style='font-size: 12px; color: #64748B;'>
        <b>Tech Stack:</b> Python, Streamlit, Scikit-learn, NLTK, Plotly<br>
        <b>Last Data Sync:</b> May 2026<br>
        <b>OS Target:</b> Windows Intel
    </div>
""", unsafe_allow_html=True)

# ----------------- TAB 1: ECOSYSTEM EXPLORER -----------------
if selected_tab == "Ecosystem Explorer":
    st.markdown("<h1 style='margin-bottom: 10px;'>Indian Startup Ecosystem Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; margin-bottom: 30px;'>Empirical intelligence mapping macro government registries and private venture deployments.</p>", unsafe_allow_html=True)
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_reg = df_gov["Count"].sum()
    total_funding = df_funding["Funding_Amount_USD"].sum()
    top_state = state_index.iloc[0]["State"]
    top_sector = df_gov.groupby("Standard_Industry")["Count"].sum().idxmax()
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Registered DPIIT Startups</div>
                <div class="metric-value">{total_reg:,}</div>
                <div class="metric-sub">⚡ 2016-2025 Registries</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Private Funding Deployed</div>
                <div class="metric-value">${total_funding / 1e9:.2f}B</div>
                <div class="metric-sub" style="color: {COLOR_PALETTE['secondary']};">▲ High Growth VC</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Ecosystem State Leader</div>
                <div class="metric-value">{top_state}</div>
                <div class="metric-sub">🏆 State Index Rank #1</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dominant Sector</div>
                <div class="metric-value">{top_sector}</div>
                <div class="metric-sub" style="color: {COLOR_PALETTE['primary']};">⚙️ Highest density</div>
            </div>
        """, unsafe_allow_html=True)

    # Charts Grid
    st.markdown("### Market Momentum & State Densities")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.plotly_chart(plot_startup_growth_yoy(df_gov), use_container_width=True)
        st.plotly_chart(plot_sector_treemap(df_gov), use_container_width=True)
        
    with g_col2:
        st.plotly_chart(plot_funding_trends(df_funding), use_container_width=True)
        st.plotly_chart(plot_state_distribution(df_gov), use_container_width=True)
        
    # Investor Network Map
    st.markdown("### Investor Syndications & Portfolio Clusters")
    st.plotly_chart(plot_investor_network(df_funding), use_container_width=True)

# ----------------- TAB 2: AI PITCH EVALUATOR & SIMULATOR -----------------
elif selected_tab == "AI Pitch Evaluator & Simulator":
    st.markdown("<h1>AI Pitch Evaluator & Startup Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; margin-bottom: 30px;'>Input pitch dimensions to classify success probability, forecast funding rounds, and simulate risk scores.</p>", unsafe_allow_html=True)
    
    # Input panel
    in_col1, in_col2 = st.columns([1, 2])
    
    with in_col1:
        st.markdown("<h3 style='color: white;'>Configure Startup Profile</h3>", unsafe_allow_html=True)
        
        inp_sector = st.selectbox("Target Sector / Industry", df_funding["Standard_Industry"].unique())
        inp_city = st.selectbox("Ecosystem Hub / City", df_funding["City"].unique())
        inp_budget = st.slider("Target Capital Requirement ($ USD Millions)", 1.0, 100.0, 10.0)
        inp_rounds = st.number_input("Prior Funding Rounds Raised", min_value=0, max_value=8, value=1)
        inp_diversity = st.radio("Gender Diverse Co-Founding Team?", ["Yes", "No"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Simulate Pitch Evaluation 🚀", use_container_width=True)
        
    with in_col2:
        st.markdown("<h3 style='color: white;'>Simulation Matrix Outputs</h3>", unsafe_allow_html=True)
        
        # We calculate the encoded indicators using pre-loaded label encoders from main classifier
        le_ind = success_data["le_industry"]
        le_ct = success_data["le_city"]
        clf_model = success_data["model"]
        
        # Encodings
        try:
            sec_enc = le_ind.transform([inp_sector])[0]
        except Exception:
            sec_enc = 0
            
        try:
            city_enc = le_ct.transform([inp_city])[0]
        except Exception:
            city_enc = 0
            
        # Target years macro parameters (Latest 2025 macro: GDP 7.0, Inflation 4.5, FDI 48.0)
        pred_features = [[sec_enc, city_enc, 7.0, 4.5, 48.0]]
        
        # Run classification
        prob_success = clf_model.predict_proba(pred_features)[0][1] * 100
        
        # Run regressor
        reg_model = funding_data["model"]
        estimated_funding = reg_model.predict(pred_features)[0] / 1e6 # in millions
        
        # Let's adjust values based on round count and budget to represent standard logic
        success_score = min(max(prob_success + (inp_rounds * 5.0) + (10.0 if inp_diversity == "Yes" else 0.0), 10.0), 98.0)
        
        # Success probability dial
        dial_color = COLOR_PALETTE["secondary"] if success_score >= 70 else (COLOR_PALETTE["accent"] if success_score >= 45 else COLOR_PALETTE["danger"])
        
        st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {dial_color};">
                <div class="metric-label">AI Evaluated Success Probability Index</div>
                <div class="metric-value" style="color: {dial_color}; font-size: 40px; -webkit-text-fill-color: initial;">{success_score:.1f}%</div>
                <div class="metric-sub" style="color: #94A3B8;">Confidence based on historical {inp_sector} outcomes in {inp_city}.</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Custom metrics and details
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            st.metric("Model Estimated Check Size", f"${estimated_funding:.2f}M USD", delta="Based on city comps")
        with o_col2:
            st.metric("Estimated Startup Age (Stable Phase)", f"{2026 - 2018} Years", delta="VC expansion threshold")
            
        # Risk assessment heatmap
        st.markdown("#### Dynamic Multi-Dimensional Risk Engine")
        
        # Dynamic risk assignments
        risk_map = {
            "Market Risk": "Low" if success_score > 70 else "Medium",
            "Financial Burn Risk": "High" if inp_budget > 30 else "Medium",
            "Regulatory Risk": "Low" if inp_sector in ["SaaS & Enterprise", "EdTech"] else "Medium",
            "Competition Risk": "High" if inp_city in ["Bangalore", "Mumbai"] else "Medium"
        }
        
        # Render clean risk block table
        r_cols = st.columns(4)
        for idx, (rk, rv) in enumerate(risk_map.items()):
            bg = COLOR_PALETTE["danger"] if rv == "High" else (COLOR_PALETTE["accent"] if rv == "Medium" else COLOR_PALETTE["secondary"])
            with r_cols[idx]:
                st.markdown(f"""
                    <div style="background-color: {bg}22; border: 1.5px solid {bg}; border-radius: 12px; padding: 12px; text-align: center;">
                        <div style="font-size: 11px; text-transform: uppercase; color: #94A3B8; font-weight:600;">{rk}</div>
                        <div style="font-size: 20px; font-weight:700; color: {bg}; margin-top:4px;">{rv}</div>
                    </div>
                """, unsafe_allow_html=True)
                
        # Simulated SWOT Generator
        st.markdown("<br>#### Generative SWOT Briefing", unsafe_allow_html=True)
        sw_col1, sw_col2 = st.columns(2)
        
        with sw_col1:
            st.markdown(f"""
                <div class="swot-box swot-s">
                    <div class="swot-title">💡 Strengths</div>
                    <ul>
                        <li>Strong localized cluster dynamics in <b>{inp_city}</b>.</li>
                        <li>High structural scalability within <b>{inp_sector}</b>.</li>
                        <li>{"Diversified founder equity background reduces standard risk." if inp_diversity == "Yes" else "Ecosystem entry validated by target VC checks."}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="swot-box swot-o">
                    <div class="swot-title">🚀 Opportunities</div>
                    <ul>
                        <li>Favorable macro-economic FDI inflows support scaling series.</li>
                        <li>Possibility of leveraging seed grant matching schemes.</li>
                        <li>Integration of rural digital interfaces to scale market.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with sw_col2:
            st.markdown(f"""
                <div class="swot-box swot-w">
                    <div class="swot-title">⚠️ Weaknesses</div>
                    <ul>
                        <li>{"High initial budget request increases capital burn pressures." if inp_budget > 25 else "Moderate check sizes limit competitive expansion velocity."}</li>
                        <li>Lack of established cash flow history for B2B models.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="swot-box swot-t">
                    <div class="swot-title">⚡ Threats</div>
                    <ul>
                        <li>Aggressive talent acquisition cost inflation in key hubs.</li>
                        <li>Regulatory credit lending limits in transaction portals.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 3: ENTREPRENEUR CO-PILOT -----------------
elif selected_tab == "Entrepreneur Co-Pilot":
    st.markdown("<h1>Entrepreneur Advisor & Co-Pilot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; margin-bottom: 30px;'>Select your industry and state to analyze opportunity ratings, match government seed schemes, and find VC matches.</p>", unsafe_allow_html=True)
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        cop_sector = st.selectbox("Select Target Sector", df_gov["Standard_Industry"].unique())
    with col_sel2:
        cop_state = st.selectbox("Select Registration State", state_index["State"].tolist())
    with col_sel3:
        cop_budget = st.number_input("Target Startup Capital ($ USD)", min_value=10000, max_value=250000000, value=2500000)
        
    st.markdown("---")
    
    # Recommender outputs
    opp_data = calculate_opportunity_score(cop_sector, df_funding, df_gov, config)
    opp_score = opp_data["opportunity_score"]
    metrics = opp_data["metrics"]
    
    res_col1, res_col2 = st.columns([1, 1.5])
    
    with res_col1:
        st.markdown(f"""
            <div class="metric-card" style="text-align: center; border: 1.5px solid {COLOR_PALETTE['secondary']};">
                <div class="metric-label">Sector Opportunity Score</div>
                <div class="metric-value" style="font-size: 55px; color: {COLOR_PALETTE['secondary']}; -webkit-text-fill-color: initial;">{opp_score}/100</div>
                <div class="metric-sub">Sentiment Signal: <b>{metrics['sentiment']}</b></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(plot_radar_metrics(metrics), use_container_width=True)
        
    with res_col2:
        # Government Schemes Matcher
        st.markdown("### Eligible Government Schemes & Seed Grants")
        schemes = get_recommended_schemes(cop_sector, cop_state)
        
        if not schemes:
            st.info("No state-specific schemes matched. Showing central schemes.")
            schemes = get_recommended_schemes(cop_sector, "Delhi")
            
        for sch in schemes[:3]:
            st.markdown(f"""
                <div style="background-color: #1E293B44; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <h5 style="color: {COLOR_PALETTE['secondary']}; margin: 0 0 6px 0; font-family: Outfit;">🛡️ {sch['name']}</h5>
                    <p style="font-size: 13px; color: #94A3B8; margin: 0 0 8px 0;">{sch['description']}</p>
                    <div style="font-size: 12px; color: #F8FAFC;"><b>Financial Benefits:</b> {sch['benefits']}</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Top matched investors
        st.markdown("### Top Venture Capitalists Matching Sector")
        vcs = get_investor_matches(cop_sector, df_funding)
        
        vc_cols = st.columns(len(vcs[:3]))
        for idx, vc in enumerate(vcs[:3]):
            with vc_cols[idx]:
                st.markdown(f"""
                    <div style="background-color: #1E293B77; border-top: 3.5px solid {COLOR_PALETTE['primary']}; border-radius: 8px; padding: 12px; text-align: center;">
                        <div style="font-size: 12px; font-weight: 600; color: #94A3B8;">{vc['Investor']}</div>
                        <div style="font-size: 16px; font-weight: 700; color: white; margin-top: 4px;">Avg check: ${vc['avg_check_usd']/1e6:.1f}M</div>
                        <div style="font-size: 10px; color: {COLOR_PALETTE['secondary']}; margin-top: 2px;">Deals: {vc['deals']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
    # Benchmark similar startups
    st.markdown("<br>### Benchmark Successful Comps in Sector", unsafe_allow_html=True)
    comps = get_startup_similarity_matches(cop_sector, "Bangalore", cop_budget, df_funding)
    
    comp_cols = st.columns(3)
    for idx, cp in enumerate(comps):
        with comp_cols[idx]:
            st.markdown(f"""
                <div style="background-color: #0F172A; border: 1.5px dashed #334155; border-radius: 12px; padding: 16px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748B; font-weight:600;">Ecosystem Benchmark {idx+1}</div>
                    <h4 style="margin: 4px 0; color: white; font-family: Outfit;">🏢 {cp['Company']}</h4>
                    <div style="font-size: 13px; color: #94A3B8;">City: {cp['City']} | Year Funded: {cp['Year']}</div>
                    <div style="font-size: 13px; color: {COLOR_PALETTE['secondary']}; margin-top: 6px;"><b>Round Size Raised:</b> ${cp['Funding_Amount_USD']/1e6:.1f}M USD</div>
                    <div style="font-size: 12px; color: #64748B;">Lead VC: {cp['Investor']}</div>
                </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 4: MARKET PROJECTIONS & SIGNALS -----------------
elif selected_tab == "Market Projections & Signals":
    st.markdown("<h1>Macro Ecosystem Forecasting & Signals</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; margin-bottom: 30px;'>Interact with macro-economic indicators and explore projected registries through 2030.</p>", unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([1.8, 1])
    
    with f_col1:
        # Load macro forecaster weights and generate forecast data
        coef = forecast_data["coefficients"]
        intercept = forecast_data["intercept"]
        
        hist_years = df_gov.groupby("Year")["Count"].sum().reset_index()
        hist_years = hist_years[hist_years["Year"] <= 2025]
        
        # Forecast years
        fc_years = list(range(2025, 2031))
        fc_counts = []
        for yr in fc_years:
            val = coef[0]*yr + coef[1]*(yr**2) + intercept
            fc_counts.append(int(val))
            
        df_forecast = pd.DataFrame({"Year": fc_years, "Count": fc_counts})
        
        st.plotly_chart(plot_forecast_chart(hist_years, df_forecast), use_container_width=True)
        
    with f_col2:
        st.markdown("### Interactive Macroeconomic Correlations")
        st.markdown("<p style='font-size:13px; color: #94A3B8;'>Vary macro parameters to see historical correlations on sector growth velocity.</p>", unsafe_allow_html=True)
        
        gdp_slider = st.slider("Projected Indian GDP Growth (%)", 4.0, 10.0, 7.5)
        fdi_slider = st.slider("Projected FDI Inflows ($ USD Billions)", 25.0, 90.0, 50.0)
        inf_slider = st.slider("Projected Inflation Rate (%)", 2.0, 8.0, 4.5)
        
        # Compute dynamic sensitivity score
        sensitivity = (gdp_slider * 6.5) + (fdi_slider * 0.45) - (inf_slider * 2.5)
        sens_pct = min(max(sensitivity, 0.0), 100.0)
        
        s_color = COLOR_PALETTE["secondary"] if sens_pct >= 60 else (COLOR_PALETTE["accent"] if sens_pct >= 40 else COLOR_PALETTE["danger"])
        
        st.markdown(f"""
            <div class="metric-card" style="margin-top: 25px; border-left: 5px solid {s_color};">
                <div class="metric-label">Macro Expansion Climate Index</div>
                <div class="metric-value" style="font-size: 35px; color: {s_color}; -webkit-text-fill-color: initial;">{sens_pct:.1f}%</div>
                <div class="metric-sub">Higher scores imply rapid venture funding acceleration and lower operational risks.</div>
            </div>
        """, unsafe_allow_html=True)

# ----------------- TAB 5: SENTIMENT & NEWS ANALYZER -----------------
elif selected_tab == "Sentiment & News Analyzer":
    st.markdown("<h1>NLP Sentiment & Live News Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; margin-bottom: 30px;'>Explore public sector sentiment gauges, read industry streams, or run custom pitch evaluations.</p>", unsafe_allow_html=True)
    
    s_col1, s_col2 = st.columns([1.2, 2])
    
    with s_col1:
        st.markdown("### Public Sentiment by Sector")
        target_sec = st.selectbox("Select Industry to Gauge", df_funding["Standard_Industry"].unique())
        
        avg_sent, feeds = get_sector_sentiment_stats(target_sec)
        
        dial_color = COLOR_PALETTE["secondary"] if avg_sent["compound"] >= 0.2 else (COLOR_PALETTE["accent"] if avg_sent["compound"] >= 0.0 else COLOR_PALETTE["danger"])
        
        st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-bottom: 5px solid {dial_color};">
                <div class="metric-label">Average Public Sentiment Rating</div>
                <div class="metric-value" style="font-size: 40px; color: {dial_color}; -webkit-text-fill-color: initial;">{avg_sent['sentiment']}</div>
                <div class="metric-sub">Compound VADER Score: <b>{avg_sent['compound']:.2f}</b></div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Sentiment Polarities")
        st.markdown(f"🟢 **Positive:** {avg_sent['pos']*100:.1f}%")
        st.markdown(f"⚪ **Neutral:** {avg_sent['neu']*100:.1f}%")
        st.markdown(f"🔴 **Negative:** {avg_sent['neg']*100:.1f}%")
        
    with s_col2:
        st.markdown("### Sector Live-News Stream")
        for fd in feeds:
            badge_color = "#10B981" if fd["sentiment"] == "Positive" else ("#F59E0B" if fd["sentiment"] == "Neutral" else "#EF4444")
            st.markdown(f"""
                <div style="background-color: #1E293B44; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 10px; color: #64748B; font-weight: 600;">NEWS ALERT</span>
                        <span style="font-size: 9px; font-weight:700; color: white; background-color: {badge_color}; border-radius: 4px; padding: 2px 6px;">{fd['sentiment']}</span>
                    </div>
                    <div style="font-size: 13.5px; color: #E2E8F0;">{fd['text']}</div>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("### Dynamic Text Sentiment Evaluator")
    st.markdown("<p style='font-size:13px; color: #94A3B8;'>Type a pitch deck summary, co-founder agreement details, or news headline below to run a real-time NLTK VADER sentiment calculation.</p>", unsafe_allow_html=True)
    
    custom_txt = st.text_area("Input Text for NLP Sentiment Evaluation", "Our startup is pioneering carbon capture materials in Gujarat, securing key state pilot testing rights and backed by early angel funding.")
    
    if custom_txt:
        custom_score = analyze_text(custom_txt)
        c_badge = "#10B981" if custom_score["sentiment"] == "Positive" else ("#F59E0B" if custom_score["sentiment"] == "Neutral" else "#EF4444")
        
        st.markdown(f"""
            <div style="background-color: {c_badge}15; border: 1.5px solid {c_badge}; border-radius: 12px; padding: 16px;">
                <div style="font-size: 13px; color: #94A3B8;">NLP Polarity Classification</div>
                <h4 style="margin: 4px 0 8px 0; color: white; font-family: Outfit;">Ecosystem Sentiment Label: <span style="color: {c_badge};">{custom_score['sentiment']}</span></h4>
                <div style="font-size: 13px; color: #E2E8F0;"><b>Compound Intensity:</b> {custom_score['compound']:.4f} | <b>Pos:</b> {custom_score['pos']:.3f} | <b>Neu:</b> {custom_score['neu']:.3f} | <b>Neg:</b> {custom_score['neg']:.3f}</div>
            </div>
        """, unsafe_allow_html=True)
