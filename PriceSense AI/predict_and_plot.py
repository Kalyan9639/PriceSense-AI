import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import Optional

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
MEDIAN_VALUES_PATH = "median_values.json"
MODEL_PATH = os.path.join("models", "rf_model.joblib")

# Default fallbacks if JSON is missing
DEFAULT_MEDIANS = {
    "Price": 1000.0,
    "Competitor Price": 1000.0,
    "Discount": 0.0,
    "Elasticity Index": 1.5,
    "Storage Cost": 500.0,
    "Return Rate (%)": 5.0,
    "Customer Reviews": 4.5
}

# -------------------------------------------------
# 1. INPUT SCHEMA & MAPPING
# -------------------------------------------------
class PriceInput(BaseModel):
    # Inputs from App (snake_case)
    price: float = Field(..., gt=0)
    cost: Optional[float] = None             # Maps to 'Storage Cost'
    elasticity_index: Optional[float] = None # Maps to 'Elasticity Index'
    competitor_price: Optional[float] = None # Maps to 'Competitor Price'
    discount: Optional[float] = None         # Maps to 'Discount'
    return_rate: Optional[float] = None      # Maps to 'Return Rate (%)'
    reviews: Optional[float] = None          # Maps to 'Customer Reviews'

    def get_filled_dict(self) -> dict:
        """
        Loads medians and fills missing values.
        Returns a dictionary with keys matching the MODEL'S feature names.
        """
        # 1. Load Medians (JSON keys are likely "Competitor Price", etc.)
        try:
            with open(MEDIAN_VALUES_PATH, "r") as f:
                loaded_medians = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            loaded_medians = DEFAULT_MEDIANS

        # 2. Map Pydantic fields to Model Feature Names
        # If user input is None, try to get from loaded_medians, else default
        def get_val(user_val, json_key, default_val):
            if json_key == "Discount":
                if user_val is not None:
                    return user_val
            else:
                # For Cost, Competitor Price, etc., treat 0 as "Use Median"
                if user_val is not None and user_val > 0:
                    return user_val
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
        """
        Returns a DataFrame with the EXACT columns required by the model.
        """
        data_dict = self.get_filled_dict()
        # Order matters for some models, but sklearn usually handles by name if DF is passed.
        # We ensure keys match the "Seen at fit time" list.
        return pd.DataFrame([data_dict])

# -------------------------------------------------
# 2. CORE LOGIC CLASS
# -------------------------------------------------
class PriceSense:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.model_loaded = False
        
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.model_loaded = True
            else:
                print(f"⚠️ Model not found at {model_path}. Using fallback logic.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def predict(self, input_data: PriceInput):
        """
        Returns: demand, revenue, profit
        """
        df = input_data.to_dataframe()
        
        if self.model_loaded:
            demand = float(self.model.predict(df)[0])
        else:
            # Fallback logic for testing without model
            filled = input_data.get_filled_dict()
            p = filled["Price"]
            e = filled["Elasticity Index"]
            demand = max(0, 1000 * (1 - p/2000) * e)

        # Business Metrics
        # Note: We use the filled dictionary to get the Price and Cost used in calc
        filled_vals = input_data.get_filled_dict()
        price_val = filled_vals["Price"]
        cost_val = filled_vals["Storage Cost"] # Using Storage Cost as unit cost

        revenue = demand * price_val
        profit = demand * (price_val - cost_val)

        return demand, revenue, profit

    def plot_profit_curve(self, input_data: PriceInput, save_dir="visualization"):
        """
        Generates the Plotly figure.
        """
        filled_original = input_data.get_filled_dict()
        base_price = filled_original["Price"]
        
        # Create price range
        prices = np.linspace(base_price * 0.5, base_price * 1.5, 20)
        
        profits = []
        revenues = []
        
        for p in prices:
            # Create a temporary input object
            # We keep other factors constant
            temp_input = PriceInput(
                price=p,
                cost=filled_original["Storage Cost"],
                elasticity_index=filled_original["Elasticity Index"],
                competitor_price=filled_original["Competitor Price"],
                discount=filled_original["Discount"],
                return_rate=filled_original["Return Rate (%)"],
                reviews=filled_original["Customer Reviews"]
            )
            
            _, rev, prof = self.predict(temp_input)
            revenues.append(rev)
            profits.append(prof)
            
        # Plotting
        fig = go.Figure()

        # Revenue
        fig.add_trace(go.Scatter(
            x=prices, y=revenues,
            mode='lines',
            name='Revenue',
            line=dict(color='#3b82f6', width=2, dash='dot')
        ))

        # Profit
        fig.add_trace(go.Scatter(
            x=prices, y=profits,
            mode='lines+markers',
            name='Profit',
            line=dict(color='#10b981', width=3)
        ))
        
        # User's current position
        _, curr_rev, curr_prof = self.predict(input_data)
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
        
        return fig, None