
import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
import model_prediction as backend

# --- Configuration ---
# No backend URL needed

# --- Page Config ---
st.set_page_config(
    page_title="PriceSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Dark Theme & Styling) ---
st.markdown("""
<style>
    /* Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #0d1117; /* Very Dark Blue/Black */
        color: #e6edf3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    section[data-testid="stSidebar"] {
        width: 20vw;
    }

    /* --- COMPONENT STYLES --- */

    /* Card Container */
    .metric-card {
        background-color: #1f242d; /* #161b22 */
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.4);
        border-color: #58a6ff;
    }
    
    .metric-title {
        color: #8b949e;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .metric-delta {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-pos { background-color: rgba(46, 160, 67, 0.15); color: #3fb950; }
    .delta-neg { background-color: rgba(248, 81, 73, 0.15); color: #f85149; }
    .delta-neu { background-color: rgba(56, 139, 253, 0.15); color: #58a6ff; }
    
    /* Header Styles */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid #30363d;
    }
    
    .header-left {
        display: flex;
        align-items: flex-start;
        gap: 20px;
    }
    
    .logo-container {
        width: 54px;
        height: 54px;
        background: linear-gradient(135deg, #0070f3 0%, #00a6f3 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 112, 243, 0.3);
    }

    .title-stack h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .live-badge {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 100px;
        border: 1px solid rgba(46, 160, 67, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        vertical-align: middle;
        position: relative;
        top: -2px;
    }

    .title-stack p {
        margin: 4px 0 0 0;
        color: #8b949e;
        font-size: 1rem;
    }

    /* Plotly Chart Container */
    .chart-box {
        background-color: #1f242d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px; /* Reduced padding */
        margin-top: 30px;
    }
    
    /* Custom Button override */
    button[kind="primary"] {
        background-color: #238636 !important;
        border: none !important;
        transition: all 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: #2ea043 !important;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }
    
    /* Export Button Styling (Targeting the stDownloadButton) */
    [data-testid="stDownloadButton"] button {
        background-color: #006400;
        border: 1px solid #30363d;
        color: #e6edf3;
        transition: all 0.2s;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: #8b949e;
        background-color: #161b22;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# --- Header Section ---
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("""
    <div class="app-header" style="border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
        <div class="header-left">
            <div class="logo-container">
               <i style="font-style: normal;">⚡</i>
            </div>
            <div class="title-stack">
                <h1 style="margin-bottom: 0px;">PriceSense AI <span class="live-badge">● Live</span></h1>
                <p style="margin-top: -5px; opacity: 0.8;">Intelligent Price Optimization & Strategy Analytics</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    # Export Button (Functional)
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True) # Spacer
    # We create a dummy CSV for download
    dummy_data = pd.DataFrame({"Metric": ["Demand", "Revenue", "Profit"], "Value": [100, 2000, 500]})
    csv = dummy_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Export Report",
        data=csv,
        file_name="pricesense_report.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")

# Quick Stats row removed as per feedback

st.markdown("<br>", unsafe_allow_html=True)

# --- Sidebar: Simulation Parameters ---
with st.sidebar:
    st.header("Simulation Parameters")
    
    with st.form("simulation_form"):
        price = st.number_input("Target Price ($)", min_value=0.0, value=249.0, step=1.0)
        cost = st.number_input("Unit Cost ($)", min_value=0.0, value=85.0, step=1.0)
        competitor_price = st.number_input("Competitor Average ($)", min_value=0.0, value=265.0, step=1.0)
        
        st.markdown("---")
        elasticity_index = st.slider("Price Elasticity", 0.0, 5.0, 1.4, 0.1)
        discount = st.slider("Planned Discount (%)", 0.0, 100.0, 15.0, 1.0)
        return_rate = st.number_input("Return Rate (%)", 0.0, 100.0, 4.2, 0.1)
        reviews = st.slider("Avg Review Score", 0.0, 5.0, 4.5, 0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("✨ Run AI Prediction", type="primary")

# --- Logic & Display ---
# Prepare Inputs
payload = {
    "price": price,
    "cost": cost,
    "competitor_price": competitor_price,
    "discount": discount,
    "elasticity_index": elasticity_index,
    "return_rate": return_rate,
    "reviews": reviews
}

# Load Model (Cached)
@st.cache_resource
def load_backend_resources():
    backend.load_resources()

# Load resources silently at startup
try:
    load_backend_resources()
except Exception as e:
    st.error(f"Failed to load AI model: {e}")
    st.stop()

# --- 1. Get Predictions ---
try:
    # Direct Function Call instead of API Request
    data = backend.predict_metrics(payload)
    
    pred_demand = data.get("predicted_demand", 0)
    pred_revenue = data.get("predicted_revenue", 0)
    pred_profit = data.get("predicted_profit", 0)
    
    margin = (pred_profit / pred_revenue * 100) if pred_revenue > 0 else 0
    
    # Custom Cards Layout - 2 columns (Revenue & Profit only)
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
<div class="metric-card">
    <div>
        <div class="metric-title">Est. Revenue</div>
        <div class="metric-value">${pred_revenue:,.0f}</div>
    </div>
    <div>
            <div class="metric-delta delta-pos">
            <span>+ $42.8k</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    with c2:
        profit_delta_class = "delta-pos" if pred_profit >= 0 else "delta-neg"
        profit_arrow = "↑" if pred_profit >= 0 else "↓"
        # Using a mock delta for visual consistency as in the screenshot
        delta_val = f"{profit_arrow} ${abs(pred_profit * 0.08):,.0f}" if pred_profit >= 0 else f"{profit_arrow} ${abs(pred_profit * 0.12):,.0f}"
        
        st.markdown(f"""
<div class="metric-card" style="border-color: #1f6feb;">
    <div style="display: flex; justify-content: space-between;">
        <div class="metric-title" style="color: #58a6ff;">Predicted Net Profit</div>
        <span class="live-badge" style="background: #1f6feb; color: white; border: none;">OPTIMIZED</span>
    </div>
    <div class="metric-value" style="color: #58a6ff;">${pred_profit:,.0f}</div>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="metric-delta {profit_delta_class}">
            <span>{delta_val}</span>
        </div>
        <div style="color: #58a6ff; font-size: 0.8rem; font-weight: 600;">{margin:.1f}% Margin</div>
    </div>
    <div style="background: #21262d; height: 6px; border-radius: 3px; margin-top: 10px; overflow: hidden;">
        <div style="width: {min(margin, 100)}%; background: #58a6ff; height: 100%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()


# --- 2. Sensitivity Analysis ---
st.markdown("<br>", unsafe_allow_html=True)

# Container for Chart
st.markdown("""
<div class="chart-box">
    <h3 style="margin-top: 0; margin-bottom: 25px;">📊 Price Sensitivity Analysis</h3>
    <p style='color: #8b949e; margin-bottom: 20px;'>Exploring Revenue vs Profit curves from ±50% of target price.</p>
""", unsafe_allow_html=True)

try:
    # Direct Function Call
    sens_data = backend.get_sensitivity_analysis(payload)
    
    if "plot_json" in sens_data:
        fig_json = sens_data["plot_json"]
        fig = go.Figure(json.loads(fig_json))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#8b949e'),
            xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.warning("No plot data received.")

except Exception as e:
    st.error(f"Analysis Error: {e}")

st.markdown('</div>', unsafe_allow_html=True) # End chart-box

# --- 3. AI Recommendations ---
st.markdown("<br>", unsafe_allow_html=True)
col_rec, _ = st.columns([1, 1])

with col_rec:
    st.markdown("### 💡 AI Recommendations")
    st.markdown("""
    <div style="background: #1f242d; border: 1px dashed #30363d; border-radius: 12px; padding: 24px; display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 2rem;">🚀</div>
        <div>
            <div style="color: #e6edf3; font-weight: 600;">Advanced Strategy Module</div>
             <div style="color: #8b949e; font-size: 0.9rem;">Competitive gap analysis and dynamic pricing suggestions coming soon.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
