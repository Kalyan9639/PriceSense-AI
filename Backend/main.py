
import os
import json
import logging
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import plotly.io as pio

# --- Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "rf_model.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MEDIAN_VALUES_PATH = "median_values.json"
HF_REPO_ID = "mr-checker/static-price-optimizer-model"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
model = None
median_values = {}

app = FastAPI(title="Price Optimization Backend", version="1.0.0")

# --- Schemas ---

class PredictionRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price")
    cost: Optional[float] = Field(None, ge=0, description="Product cost (Storage Cost)")
    competitor_price: Optional[float] = Field(None, ge=0, description="Competitor price")
    discount: Optional[float] = Field(None, ge=0, description="Discount amount")
    elasticity_index: Optional[float] = Field(None, gt=0, description="Elasticity index")
    return_rate: Optional[float] = Field(None, ge=0, le=100, description="Return rate percentage (0-100)")
    reviews: Optional[float] = Field(None, ge=0, le=5, description="Customer reviews (0-5)")

class PredictionResponse(BaseModel):
    predicted_demand: float
    predicted_revenue: float
    predicted_profit: float

class SensitivityResponse(BaseModel):
    prices: List[float]
    revenues: List[float]
    profits: List[float]
    plot_json: str

# --- Startup Events ---

@app.on_event("startup")
async def startup_event():
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
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load model from {MODEL_PATH}: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

    # 4. Load Median Values
    if not os.path.exists(MEDIAN_VALUES_PATH):
        logger.critical(f"Median values file missing: {MEDIAN_VALUES_PATH}")
        raise RuntimeError(f"Median values file missing: {MEDIAN_VALUES_PATH}")
    
    try:
        with open(MEDIAN_VALUES_PATH, "r") as f:
            median_values = json.load(f)
        logger.info("Median values loaded successfully.")
        
        # Verify required keys in median values
        required_keys = [
            "Price", "Competitor Price", "Discount", "Elasticity Index",
            "Storage Cost", "Return Rate (%)", "Customer Reviews"
        ]
        missing_keys = [key for key in required_keys if key not in median_values]
        if missing_keys:
             raise ValueError(f"Median values JSON missing keys: {missing_keys}")

    except Exception as e:
        logger.critical(f"Failed to load median values: {e}")
        raise RuntimeError(f"Failed to load median values: {e}")

# --- Helper Functions ---

def prepare_input_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """Prepares the input DataFrame for prediction, filling missing values with medians."""
    # Map API fields to model features exactly
    data = {
        "Price": request.price,
        "Competitor Price": request.competitor_price if request.competitor_price is not None else median_values["Competitor Price"],
        "Discount": request.discount if request.discount is not None else median_values["Discount"],
        "Elasticity Index": request.elasticity_index if request.elasticity_index is not None else median_values["Elasticity Index"],
        "Storage Cost": request.cost if request.cost is not None else median_values["Storage Cost"],
        "Return Rate (%)": request.return_rate if request.return_rate is not None else median_values["Return Rate (%)"],
        "Customer Reviews": request.reviews if request.reviews is not None else median_values["Customer Reviews"]
    }
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([data])
    
    # Validating column order just in case model is sensitive to order (RandomForest usually is)
    expected_cols = [
        "Price", "Competitor Price", "Discount", "Elasticity Index",
        "Storage Cost", "Return Rate (%)", "Customer Reviews"
    ]
    df = df[expected_cols]
    
    return df, data["Storage Cost"] # Return cost for profit calc

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_df, cost = prepare_input_dataframe(request)
        
        # Predict Demand
        demand = model.predict(input_df)[0]
        
        # Ensure non-negative demand (though RF regression *can* predict negative, businesses usually clamp at 0)
        # Requirement: "Never let backend return negative demand."
        demand = max(0.0, float(demand))
        
        revenue = demand * request.price
        profit = demand * (request.price - cost)
        
        return PredictionResponse(
            predicted_demand=demand,
            predicted_revenue=revenue,
            predicted_profit=profit
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/sensitivity", response_model=SensitivityResponse)
async def sensitivity_analysis(
    price: float,
    cost: Optional[float] = None,
    competitor_price: Optional[float] = None,
    discount: Optional[float] = None,
    elasticity_index: Optional[float] = None,
    return_rate: Optional[float] = None,
    reviews: Optional[float] = None
):
    """
    Performs sensitivity analysis by varying price ±50%.
    Inputs are passed as query parameters (re-using logic from PredictionRequest schema manually or via dependency)
    To keep it simple and consistent with POST body, let's construct a Request object.
    Actually, GET requests usually take query params. Let's use the same validation logic.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Manually validate 'price' as it's required and > 0
    if price <= 0:
        raise HTTPException(status_code=400, detail="Price must be greater than 0")

    # Construct a base request object to reuse logic
    # Note: validation for optional fields is relaxed here if we just pass them through, 
    # but strictly we should validate.
    try:
        base_request = PredictionRequest(
            price=price,
            cost=cost,
            competitor_price=competitor_price,
            discount=discount,
            elasticity_index=elasticity_index,
            return_rate=return_rate,
            reviews=reviews
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input parameters: {e}")

    # Generate price range: ±50%
    # Let's do 10 points below and 10 points above, or just a comprehensive range.
    # Requirement: "Generate price range: ±50% around input price" (implied continuous or steps)
    # Let's do 20 steps.
    min_price = price * 0.5
    max_price = price * 1.5
    
    # Create 20 evenly spaced prices
    # numpy linspace is good, but let's stick to standard python/pandas if possible to avoid extra deps if not needed (pandas loaded though)
    import numpy as np
    prices = np.linspace(min_price, max_price, 20).tolist()
    
    revenues = []
    profits = []
    
    # We can batch predict for efficiency, but loop is fine for 20 items.
    # Batch prediction is better.
    
    # Prepare batch dataframe
    # 1. Get base data with medians filled
    _, used_cost = prepare_input_dataframe(base_request) # Get the filled values
    
    # Re-construct base data dict to replicate
    base_data = {
        "Price": prices, # This will be the varying column
        "Competitor Price": [base_request.competitor_price if base_request.competitor_price is not None else median_values["Competitor Price"]] * len(prices),
        "Discount": [base_request.discount if base_request.discount is not None else median_values["Discount"]] * len(prices),
        "Elasticity Index": [base_request.elasticity_index if base_request.elasticity_index is not None else median_values["Elasticity Index"]] * len(prices),
        "Storage Cost": [used_cost] * len(prices),
        "Return Rate (%)": [base_request.return_rate if base_request.return_rate is not None else median_values["Return Rate (%)"]] * len(prices),
        "Customer Reviews": [base_request.reviews if base_request.reviews is not None else median_values["Customer Reviews"]] * len(prices)
    }
    
    batch_df = pd.DataFrame(base_data)
    
    # Ensure column order
    expected_cols = [
        "Price", "Competitor Price", "Discount", "Elasticity Index",
        "Storage Cost", "Return Rate (%)", "Customer Reviews"
    ]
    batch_df = batch_df[expected_cols]
    
    try:
        demands = model.predict(batch_df)
        
        # Process results
        for p, d in zip(prices, demands):
            d = max(0.0, float(d))
            rev = d * p
            prof = d * (p - used_cost)
            
            revenues.append(rev)
            profits.append(prof)
            
        # --- Create Plotly Figure ---
        fig = go.Figure()

        # Add Profit Line
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

        # Add Current Price Marker (Calculate exactly for the marker)
        # We need the profit at current price.
        # We can reuse the base prediction logic or just interpolate/find closest.
        # Let's calculate it exactly since we have the model.
        # Or, since we only did batch prediction, let's just use the closest point or add a separate prediction.
        # For simplicity and speed, let's just mark the x-line.
        
        # Add Vertical Line for Current Price
        fig.add_vline(
            x=price, 
            line_width=2, 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Current Price", 
            annotation_position="top right"
        )
        
        # Add Horizontal Line for Zero Profit (Breakeven)
        fig.add_hline(
            y=0, 
            line_width=1, 
            line_color="gray", 
            line_dash="dot",
            annotation_text="Breakeven", 
            annotation_position="bottom right"
        )

        # Update Layout for "Nice" Look
        fig.update_layout(
            title="Profit Sensitivity Analysis",
            xaxis_title="Price ($)",
            yaxis_title="Profit ($)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Determine Profit/Loss at current price for annotation (optional)
        # We'll skip exact calculation to assume the line speaks for itself.
        
        plot_json = pio.to_json(fig)
            
        return SensitivityResponse(
            prices=prices,
            revenues=revenues,
            profits=profits,
            plot_json=plot_json
        )

    except Exception as e:
        logger.error(f"Sensitivity analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {str(e)}")

