import streamlit as st
import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import Optional
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# 1. APP CONFIGURATION & SETUP
# --------------------------------------------------
st.set_page_config(
    page_title="PriceSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = "models"
MODEL_NAME = "rf_model.joblib"
REPO_ID = "mr-checker/static-price-optimizer-model"
FILENAME = "rf_model.joblib"

# 🛡️ CRITICAL: The exact column order the model expects
MODEL_COLUMNS = [
    "Price",
    "Competitor Price",
    "Discount",
    "Elasticity Index",
    "Storage Cost",
    "Return Rate (%)",
    "Customer Reviews"
]

DEFAULT_MEDIANS = {
    "Price": 1000.0,
    "Competitor Price": 1000.0,
    "Discount": 0.0,
    "Elasticity Index": 1.5,
    "Storage Cost": 500.0,
    "Return Rate (%)": 5.0,
    "Customer Reviews": 4.5
}

# --------------------------------------------------
# 2. CACHED RESOURCE LOADING (Runs once on startup)
# --------------------------------------------------

@st.cache_resource(show_spinner="⬇️ Downloading model...")
def download_model():
    """Downloads the model from Hugging Face if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    # Check if file exists and is valid size (e.g. > 10KB to avoid corrupt files)
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 10000:
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
                # token=st.secrets["HF_TOKEN"] # Uncomment if repo is private
            )
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None
    return model_path

@st.cache_resource(show_spinner="🧠 Loading AI Model...")
def load_ai_model(path):
    """Loads the model into memory."""
    try:
        if path and os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model file: {e}")
    return None

# Execute Setup
model_file_path = download_model()
loaded_model = load_ai_model(model_file_path)

# --------------------------------------------------
# 3. DATA CLASSES & LOGIC
# --------------------------------------------------

class PriceInput(BaseModel):
    # Inputs from App (snake_case)
    price: float = Field(..., gt=0)
    cost: Optional[float] = None             
    elasticity_index: Optional[float] = None 
    competitor_price: Optional[float] = None 
    discount: Optional[float] = None         
    return_rate: Optional[float] = None      
    reviews: Optional[float] = None          

    def get_filled_dict(self) -> dict:
        """Fills missing inputs with defaults/medians."""
        # Use defaults since we are single-file and don't rely on external JSON
        loaded_medians = DEFAULT_MEDIANS

        def get_val(user_val, json_key, default_val):
            if json_key == "Discount":
                if user_val is not None: return user_val
            else:
                if user_val is not None and user_val > 0: return user_val
            return loaded_medians.get(json_key, default_val)

        return {
            "Price": self.price,
            "Competitor Price": get_val(self.competitor_price, "Competitor Price", 1000.0),
            "Discount": get_val(self.discount, "Discount", 0.0),
            "Elasticity Index": get_val(self.elasticity_index, "Elasticity Index", 1.5),
            "Storage Cost": get_val(self.cost, "Storage Cost", 450.0),
            "Return Rate (%)": get_val(self.return_rate, "Return Rate (%)", 5.0),
            "Customer Reviews": get_val(self.reviews, "Customer Reviews", 4.0)
        }

    def to_dataframe(self) -> pd.DataFrame:
        data_dict = self.get_filled_dict()
        df = pd.DataFrame([data_dict])
        # 🛡️ FORCE EXACT COLUMN ORDER
        df = df[MODEL_COLUMNS]
        return df

class PriceSense:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data: PriceInput):
        df = input_data.to_dataframe()
        
        if self.model:
            try:
                demand = float(self.model.predict(df)[0])
            except Exception as e:
                st.error(f"⚠️ Prediction logic error: {e}")
                st.stop()
        else:
            # Fallback math if model completely fails to load
            filled = input_data.get_filled_dict()
            p = filled["Price"]
            e = filled["Elasticity Index"]
            demand = max(0, 1000 * (1 - p/2000) * e)

        # Business Metrics
        filled_vals = input_data.get_filled_dict()
        price_val = filled_vals["Price"]
        cost_val = filled_vals["Storage Cost"] 

        revenue = demand * price_val
        profit = demand * (price_val - cost_val)

        return demand, revenue, profit

    def plot_profit_curve(self, input_data: PriceInput):
        filled_original = input_data.get_filled_dict()
        base_price = filled_original["Price"]
        
        # Create price range (X-axis)
        prices = np.linspace(base_price * 0.5, base_price * 1.5, 20)
        
        profits = []
        revenues = []
        
        for p in prices:
            # 🛡️ CRITICAL FIX: Ensure the loop uses the exact same keys
            temp_input = PriceInput(
                price=p,
                cost=filled_original.get("Storage Cost"),
                elasticity_index=filled_original.get("Elasticity Index"),
                competitor_price=filled_original.get("Competitor Price"),
                discount=filled_original.get("Discount"),
                return_rate=filled_original.get("Return Rate (%)"),
                reviews=filled_original.get("Customer Reviews")
            )
            
            _, rev, prof = self.predict(temp_input)
            revenues.append(rev)
            profits.append(prof)
            
        # Plotting
        fig = go.Figure()

        # Revenue Line
        fig.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode='lines',
            name='Revenue',
            line=dict(color='#3b82f6', width=2, dash='dot')
        ))

        # Profit Line
        fig.add_trace(go.Scatter(
            x=prices, y=profits,
            mode='lines+markers',
            name='Profit',
            line=dict(color='#10b981', width=3)
        ))
        
        # Current Selection Dot
        _, _, curr_prof = self.predict(input_data)
        fig.add_trace(go.Scatter(
            x=[base_price], y=[curr_prof],
            mode='markers',
            name='Current Selection',
            marker=dict(color='white', size=12, line=dict(width=2, color='red'))
        ))

        fig.update_layout(
            title="Price Sensitivity Analysis",
            xaxis_title="Price ($)",
            yaxis_title="Value ($)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        return fig

# --------------------------------------------------
# 4. UI IMPLEMENTATION
# --------------------------------------------------

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    section[data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    .stNumberInput label { color: #cbd5e1 !important; font-weight: 500; }
    div[data-testid="metric-container"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
    h1, h2, h3 { color: #f8fafc !important; }
    div.stButton > button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Model Parameters")
with st.sidebar.form(key="prediction_form"):
    st.markdown("### 🏷️ Product Details")
    price = st.number_input("Price ($)", min_value=1.0, value=1044.62, step=10.0)
    competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=1045.0, step=10.0)
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 📊 Market Factors")
    elasticity = st.number_input("Elasticity Index", value=1.5, step=0.1)
    cost = st.number_input("Storage Cost ($)", min_value=0.0, value=450.0, step=10.0)
    return_rate = st.number_input("Return Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    reviews = st.number_input("Customer Reviews (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

    st.markdown("---")
    analyze_btn = st.form_submit_button("⚡ Analyze Impact")

# Main Page
st.title("📊 Price Impact Analysis")
st.markdown("Visualize how your pricing strategy affects **Profit** and **Revenue**.")
st.markdown("---")

# Instantiate Logic Class
price_sense = PriceSense(loaded_model)

if analyze_btn:
    if loaded_model is None:
        st.error("❌ Model is not loaded. Please check your internet connection or repository settings.")
    else:
        with st.spinner("🤖 Crunching numbers..."):
            try:
                # 1. Inputs
                input_data = PriceInput(
                    price=price, cost=cost, elasticity_index=elasticity,
                    competitor_price=competitor_price, discount=discount,
                    return_rate=return_rate, reviews=reviews
                )

                # 2. Predictions
                demand, revenue, profit = price_sense.predict(input_data)
                
                # 3. Metrics
                m1, m2, m3 = st.columns(3)
                with m1: st.metric("📦 Expected Demand", f"{int(demand):,} Units")
                with m2: st.metric("💰 Total Revenue", f"${revenue:,.2f}")
                with m3: 
                    margin = f"{(profit/revenue)*100:.1f}% Margin" if revenue > 0 else "0%"
                    st.metric("📈 Net Profit", f"${profit:,.2f}", delta=margin)

                # 4. Graph
                st.markdown("### 🔍 Sensitivity Analysis")
                fig = price_sense.plot_profit_curve(input_data)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("👈 Adjust inputs in the sidebar and click **'Analyze Impact'**.")