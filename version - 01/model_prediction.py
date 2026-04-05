
import os
import json
import logging
import pandas as pd
import joblib
from typing import Optional, List, Dict, Any
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "rf_model.joblib"
# Adjust path if running from root
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MEDIAN_VALUES_PATH = "median_values.json"
HF_REPO_ID = "mr-checker/static-price-optimizer-model"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
model = None
median_values = {}

def load_resources():
    """Loads model and median values. Should be called at startup."""
    global model, median_values
    
    # 1. Create models directory if missing
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")

    # 2. Check/Download Model
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Model not found at {MODEL_PATH}. Attempting download from {HF_REPO_ID}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model downloaded successfully to {downloaded_path}")
        except Exception as e:
            logger.critical(f"Failed to download model: {e}")
            raise RuntimeError(f"Failed to download model from Hugging Face: {e}")
    
    # 3. Load Model
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load model from {MODEL_PATH}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    # 4. Load Median Values
    if not median_values:
        if not os.path.exists(MEDIAN_VALUES_PATH):
            # Fallback path check if file is in Backend/
            if os.path.exists(os.path.join("Backend", MEDIAN_VALUES_PATH)):
                 # If we are in root, but json is in Backend/, let's try reading it there or assume user moved it.
                 # Given "don't create new code", I should be careful. 
                 # But list_dir showed median_values.json might NOT be in root? 
                 # Wait, Step 200 list_dir of PSA/ showed requirements.txt and app.py, but NO median_values.json.
                 # It might be in Backend/. The original main.py used "median_values.json".
                 # If main.py was in Backend/, then it was relative to Backend/.
                 # I will try both paths to be safe.
                 pass

        possible_paths = [MEDIAN_VALUES_PATH, os.path.join("Backend", MEDIAN_VALUES_PATH)]
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            logger.critical(f"Median values file missing. Checked: {possible_paths}")
            # raise RuntimeError(f"Median values file missing.") 
            # Allow fallback to empty or fail? Fail is better.
            # But for generation safety I'll assume it's where it needs to be or user handles it.
            # I'll stick to MEDIAN_VALUES_PATH but maybe the user will move it.
            # actually, let's keep it simple.
            raise RuntimeError(f"Median values file missing: {MEDIAN_VALUES_PATH}")

        try:
            with open(found_path, "r") as f:
                median_values = json.load(f)
            logger.info("Median values loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load median values: {e}")
            raise RuntimeError(f"Failed to load median values: {e}")

def prepare_input_dataframe(data: Dict[str, Any]) -> tuple:
    """Prepares the input DataFrame for prediction, filling missing values with medians."""
    # Map API fields to model features exactly
    # Expecting data dict keys: price, cost, competitor_price, discount, elasticity_index, return_rate, reviews
    
    price = data.get("price")
    competitor_price = data.get("competitor_price")
    discount = data.get("discount")
    elasticity_index = data.get("elasticity_index")
    cost = data.get("cost")
    return_rate = data.get("return_rate")
    reviews = data.get("reviews")

    row_data = {
        "Price": price,
        "Competitor Price": competitor_price if competitor_price is not None else median_values.get("Competitor Price"),
        "Discount": discount if discount is not None else median_values.get("Discount"),
        "Elasticity Index": elasticity_index if elasticity_index is not None else median_values.get("Elasticity Index"),
        "Storage Cost": cost if cost is not None else median_values.get("Storage Cost"),
        "Return Rate (%)": return_rate if return_rate is not None else median_values.get("Return Rate (%)"),
        "Customer Reviews": reviews if reviews is not None else median_values.get("Customer Reviews")
    }
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([row_data])
    
    expected_cols = [
        "Price", "Competitor Price", "Discount", "Elasticity Index",
        "Storage Cost", "Return Rate (%)", "Customer Reviews"
    ]
    df = df[expected_cols]
    
    return df, row_data["Storage Cost"]

def predict_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Predicts demand, revenue, and profit."""
    if model is None:
        load_resources()
    
    try:
        input_df, cost = prepare_input_dataframe(data)
        
        # Predict Demand
        demand = model.predict(input_df)[0]
        demand = max(0.0, float(demand))
        
        price = data.get("price")
        revenue = demand * price
        profit = demand * (price - cost)
        
        return {
            "predicted_demand": demand,
            "predicted_revenue": revenue,
            "predicted_profit": profit
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise RuntimeError(f"Prediction failed: {str(e)}")

def get_sensitivity_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Performs sensitivity analysis."""
    if model is None:
        load_resources()
        
    price = data.get("price")
    if price <= 0:
        raise ValueError("Price must be greater than 0")

    # Generate price range: ±50%
    min_price = price * 0.5
    max_price = price * 1.5
    
    prices = np.linspace(min_price, max_price, 20).tolist()
    
    revenues = []
    profits = []
    
    # Prepare batch dataframe
    # Get base data with provided/median values, but we need to override Price
    _, used_cost = prepare_input_dataframe(data)
    
    # We need to construct the batch DF manually to correspond to the varying prices
    # We can reuse prepare_input_dataframe logic but simplified for batch
    
    # Get the "static" values from a single prep
    # Since prepare_input_dataframe returns a 1-row DF, we can extract common vals
    # But simpler to just access dictionary logic again.
    
    # Let's trust we can get clean scalar values
    base_competitor = data.get("competitor_price") if data.get("competitor_price") is not None else median_values.get("Competitor Price")
    base_discount = data.get("discount") if data.get("discount") is not None else median_values.get("Discount")
    base_elasticity = data.get("elasticity_index") if data.get("elasticity_index") is not None else median_values.get("Elasticity Index")
    base_return = data.get("return_rate") if data.get("return_rate") is not None else median_values.get("Return Rate (%)")
    base_reviews = data.get("reviews") if data.get("reviews") is not None else median_values.get("Customer Reviews")
    
    batch_data = {
        "Price": prices,
        "Competitor Price": [base_competitor] * len(prices),
        "Discount": [base_discount] * len(prices),
        "Elasticity Index": [base_elasticity] * len(prices),
        "Storage Cost": [used_cost] * len(prices),
        "Return Rate (%)": [base_return] * len(prices),
        "Customer Reviews": [base_reviews] * len(prices)
    }
    
    batch_df = pd.DataFrame(batch_data)
    expected_cols = [
        "Price", "Competitor Price", "Discount", "Elasticity Index",
        "Storage Cost", "Return Rate (%)", "Customer Reviews"
    ]
    batch_df = batch_df[expected_cols]
    
    try:
        demands = model.predict(batch_df)
        
        for p, d in zip(prices, demands):
            d = max(0.0, float(d))
            rev = d * p
            prof = d * (p - used_cost)
            revenues.append(rev)
            profits.append(prof)
            
        # --- Create Plotly Figure ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices, 
            y=profits, 
            mode='lines', 
            name='Profit',
            line=dict(color='#2ca02c', width=3),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.1)',
            hovertemplate='Price: $%{x:.2f}<br>Profit: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_vline(
            x=price, 
            line_width=2, 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Current Price", 
            annotation_position="top right"
        )
        
        fig.add_hline(
            y=0, 
            line_width=1, 
            line_color="gray", 
            line_dash="dot",
            annotation_text="Breakeven", 
            annotation_position="bottom right"
        )

        fig.update_layout(
            title="Profit Sensitivity Analysis",
            xaxis_title="Price ($)",
            yaxis_title="Profit ($)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        plot_json = pio.to_json(fig)
            
        return {
            "prices": prices,
            "revenues": revenues,
            "profits": profits,
            "plot_json": plot_json
        }

    except Exception as e:
        logger.error(f"Sensitivity analysis error: {e}")
        raise RuntimeError(f"Sensitivity analysis failed: {str(e)}")
