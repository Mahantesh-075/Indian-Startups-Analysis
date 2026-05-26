import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Harmonious Sleek HSL tailor color palette for premium design
COLOR_PALETTE = {
    "primary": "#6366F1",       # Indigo
    "secondary": "#10B981",     # Emerald
    "accent": "#F59E0B",        # Amber
    "background": "#0F172A",    # Dark Slate
    "text": "#E2E8F0",          # Off-white
    "danger": "#EF4444",        # Rose/Red
    "chart_colors": ["#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4", "#F43F5E", "#14B8A6"]
}

def plot_startup_growth_yoy(df_gov):
    """Renders a beautiful multi-bar or line chart of YoY registration growth."""
    growth = df_gov.groupby("Year")["Count"].sum().reset_index()
    
    fig = px.bar(
        growth, x="Year", y="Count",
        title="Registered Startup Growth in India",
        color_discrete_sequence=[COLOR_PALETTE["primary"]],
        text_auto=".2s"
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        title_font_family="Outfit, Inter, sans-serif",
        xaxis=dict(showgrid=False, title="Registration Year"),
        yaxis=dict(showgrid=True, gridcolor="#334155", title="Number of Startups"),
        hovermode="x unified"
    )
    return fig

def plot_funding_trends(df_funding):
    """Renders an area and scatter chart of private VC funding trends (Year vs Amount)."""
    funding = df_funding.groupby("Year")["Funding_Amount_USD"].agg(["sum", "count"]).reset_index()
    funding["sum_millions"] = funding["sum"] / 1e6
    
    fig = go.Figure()
    
    # Area for amount
    fig.add_trace(go.Scatter(
        x=funding["Year"], y=funding["sum_millions"],
        mode="lines+markers",
        name="Funding Amount ($M)",
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.2)",
        line=dict(color=COLOR_PALETTE["primary"], width=3)
    ))
    
    fig.update_layout(
        title="VC Funding Velocity & Trends (2010 - 2025)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        xaxis=dict(showgrid=False, title="Year"),
        yaxis=dict(showgrid=True, gridcolor="#334155", title="Total Funding ($ Millions)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_state_distribution(df_gov):
    """Renders a horizontal ranking chart of registrations by State."""
    state_counts = df_gov.groupby("State")["Count"].sum().sort_values(ascending=False).head(12).reset_index()
    
    fig = px.bar(
        state_counts, x="Count", y="State",
        orientation="h",
        title="Top 12 Indian States by Startup Densities",
        color="Count",
        color_continuous_scale=px.colors.sequential.Sunsetdark
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        xaxis=dict(showgrid=True, gridcolor="#334155", title="Registered Startups"),
        yaxis=dict(showgrid=False, categoryorder="total ascending", title=""),
        coloraxis_showscale=False
    )
    return fig

def plot_sector_treemap(df_gov):
    """Renders an interactive treemap of registered startups by standard industry."""
    sector_counts = df_gov.groupby("Standard_Industry")["Count"].sum().reset_index()
    
    fig = px.treemap(
        sector_counts, path=["Standard_Industry"], values="Count",
        title="Ecosystem Sector Market Shares",
        color="Count",
        color_continuous_scale=px.colors.sequential.Emrld
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        coloraxis_showscale=False
    )
    return fig

def plot_radar_metrics(metrics):
    """Renders a gorgeous radar chart for Opportunity Score metrics."""
    categories = ["Reg Growth YoY", "Funding Growth YoY", "Market Density Score", "Sentiment Score"]
    
    # Scale metrics to 0-100 for radar uniformity
    reg_growth = min(max(metrics["registration_growth"], -50.0), 150.0)
    fund_growth = min(max(metrics["funding_growth"], -50.0), 200.0)
    
    s_growth = np.interp(reg_growth, [-50.0, 150.0], [10, 100])
    s_funding = np.interp(fund_growth, [-50.0, 200.0], [10, 100])
    s_density = min(max(metrics["startup_density_pct"] * 8, 10), 100)
    s_sentiment = np.interp(metrics["sentiment_score"], [-1.0, 1.0], [10, 100])
    
    values = [s_growth, s_funding, s_density, s_sentiment]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.3)",
        line=dict(color=COLOR_PALETTE["secondary"], width=2.5),
        name="Sector Opportunity"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color=COLOR_PALETTE["text"],
                gridcolor="#334155",
                linecolor="#334155"
            ),
            angularaxis=dict(
                gridcolor="#334155",
                linecolor="#334155",
                color=COLOR_PALETTE["text"]
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title="Opportunity Dimensions Radar",
        showlegend=False
    )
    return fig

def plot_investor_network(df_funding):
    """Simulates an interactive 2D node-edge scatter plot representing top VCs and startup connections."""
    # Find top 8 investors and their top funded companies
    top_vcs = df_funding["Investor"].value_counts().head(8).index.tolist()
    sub_df = df_funding[df_funding["Investor"].isin(top_vcs)].copy()
    
    # We assign fixed angle coordinates on a circle to the VCs, and place startups in between
    nodes = []
    edges_x = []
    edges_y = []
    
    vc_coords = {}
    for idx, vc in enumerate(top_vcs):
        angle = 2 * np.pi * idx / len(top_vcs)
        x = np.cos(angle) * 4
        y = np.sin(angle) * 4
        vc_coords[vc] = (x, y)
        nodes.append({
            "name": vc, "x": x, "y": y, "type": "VC", "size": 35, 
            "hover": f"VC Investor: {vc}<br>Deals: {sub_df[sub_df['Investor'] == vc].shape[0]}"
        })
        
    # Place startups clustered around their investors
    np.random.seed(42)
    startups_placed = set()
    
    for _, row in sub_df.iterrows():
        startup = row["Company"]
        vc = row["Investor"]
        vc_x, vc_y = vc_coords[vc]
        
        if startup not in startups_placed:
            # Place startup slightly shifted inwards
            offset_x = (np.random.rand() - 0.5) * 1.5
            offset_y = (np.random.rand() - 0.5) * 1.5
            st_x = vc_x * 0.5 + offset_x
            st_y = vc_y * 0.5 + offset_y
            
            nodes.append({
                "name": startup, "x": st_x, "y": st_y, "type": "Startup", "size": 18,
                "hover": f"Startup: {startup}<br>Sector: {row['Standard_Industry']}<br>Funding: ${row['Funding_Amount_USD'] / 1e6:.1f}M"
            })
            startups_placed.add(startup)
            
            # Edges
            edges_x.extend([vc_x, st_x, None])
            edges_y.extend([vc_y, st_y, None])
            
    df_nodes = pd.DataFrame(nodes)
    
    fig = go.Figure()
    
    # Draw Edges
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        line=dict(color="#334155", width=1),
        hoverinfo="none",
        mode="lines"
    ))
    
    # Draw Nodes
    for n_type, color, size in [("VC", COLOR_PALETTE["primary"], 25), ("Startup", COLOR_PALETTE["secondary"], 12)]:
        df_sub = df_nodes[df_nodes["type"] == n_type]
        fig.add_trace(go.Scatter(
            x=df_sub["x"], y=df_sub["y"],
            mode="markers+text",
            name=f"{n_type}s",
            marker=dict(symbol="circle", size=size, color=color, line=dict(color="#1E293B", width=1.5)),
            text=df_sub["name"],
            textposition="top center",
            textfont=dict(color=COLOR_PALETTE["text"], size=10),
            hovertext=df_sub["hover"],
            hoverinfo="text"
        ))
        
    fig.update_layout(
        title="Active VC-Startup Investment Network Map",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_forecast_chart(historical, forecast):
    """Plots registered startup historical counts and quadratic regression predictions up to 2030."""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=historical["Year"], y=historical["Count"],
        mode="lines+markers",
        name="Historical (Gov Data)",
        line=dict(color=COLOR_PALETTE["secondary"], width=3)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast["Year"], y=forecast["Count"],
        mode="lines+markers",
        name="AI Forecast Trend",
        line=dict(color=COLOR_PALETTE["accent"], width=3, dash="dash")
    ))
    
    fig.update_layout(
        title="India Macro Startup Count Projection (Through 2030)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=COLOR_PALETTE["text"],
        title_font_size=18,
        xaxis=dict(showgrid=False, title="Year"),
        yaxis=dict(showgrid=True, gridcolor="#334155", title="Total Registered Startups"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
