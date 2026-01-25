import streamlit as st
import os
import time

# Import your existing logic
from predict_and_plot import PriceSense, PriceInput
from download_model import rfmodel

# Ensure model is available before loading
rfmodel()


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="PriceSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. Custom CSS (Dark Theme & Professional UI)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Input Labels */
    .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }

    /* KPI Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    /* Metric Values Color Override */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important; 
    }

    /* Titles */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
        border: none;
        color: white;
    }
    
    /* Plotly Chart Container */
    .plot-container {
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 10px;
        background-color: #1e293b;
        margin-top: 20px;
    }
    
    /* Error Message Styling */
    .stAlert {
        border: 1px solid #991b1b;
        background-color: #450a0a;
        color: #fca5a5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 3. Sidebar Inputs
# --------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Parameters")
st.sidebar.markdown("Adjust the values below to simulate scenarios.")

with st.sidebar.form(key="prediction_form"):
    st.markdown("### 🏷️ Product Details")
    
    # 1. Price (Default: 1044.62)
    price = st.number_input("Price ($)", min_value=1.0, value=1044.62, step=10.0)
    
    # 2. Competitor Price
    competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=1045.0, step=10.0)
    
    # 3. Discount
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 📊 Market Factors")
    
    # 4. Elasticity Index
    elasticity = st.number_input("Elasticity Index", value=1.5, step=0.1, help="Higher means more sensitive to price changes")
    
    # 5. Storage Cost (Mapped to 'Cost')
    cost = st.number_input("Storage Cost ($)", min_value=0.0, value=450.0, step=10.0)
    
    # 6. Return Rate (Visual Input)
    return_rate = st.number_input("Return Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    
    # 7. Customer Reviews (Visual Input)
    reviews = st.number_input("Customer Reviews (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

    st.markdown("---")
    # Using st.form_submit_button correctly inside the form
    analyze_btn = st.form_submit_button("⚡ Analyze Impact")

# --------------------------------------------------
# 4. Main Dashboard Area
# --------------------------------------------------

# Header
st.title("📊 Price Impact Analysis")
st.markdown("Visualize how your pricing strategy affects **Profit** and **Revenue** in real-time.")
st.markdown("---")

# Instantiate Logic
@st.cache_resource
def load_price_sense():
    ps = PriceSense()
    
    if not ps.model_loaded:
        raise RuntimeError("ML model failed to load")

    return ps

try:
    price_sense = load_price_sense()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

if analyze_btn:
    # Use a spinner to show activity
    with st.spinner("🤖 Crunching numbers & running simulations..."):
        try:
            # Add a small delay so the animation is noticeable (optional, good for UX)
            time.sleep(0.5) 

            # 1. Process Inputs (INCLUDING NEW FIELDS)
            input_data = PriceInput(
                price=price,
                cost=cost,
                elasticity_index=elasticity,
                competitor_price=competitor_price,
                discount=discount,
                return_rate=return_rate,
                reviews=reviews
            )

            # 2. Get Predictions
            demand, revenue, profit = price_sense.predict(input_data)
            
            # 3. Display Metrics (Right side of sidebar, top of main area)
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(label="📦 Expected Demand", value=f"{int(demand):,} Units")
            
            with m2:
                st.metric(label="💰 Total Revenue", value=f"${revenue:,.2f}")
                
            with m3:
                # Conditional formatting logic
                delta_val = f"{(profit/revenue)*100:.1f}% Margin" if revenue > 0 else "0%"
                st.metric(label="📈 Net Profit", value=f"${profit:,.2f}", delta=delta_val)

            # 4. Visualization
            with st.container():  
                st.markdown("<h2>🔍 Sensitivity Analysis</h2>", unsafe_allow_html=True)
            
            # Generate Chart (predict_and_plot now correctly returns fig, None)
            fig, _ = price_sense.plot_profit_curve(input_data)
            st.plotly_chart(fig, use_container_width=True,width='stretch')
            
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            # Catch any error and print it to the screen with a visible error box
            st.error(f"⚠️ An unexpected error occurred during analysis:\n\n{str(e)}")
            st.info("💡 Tip: Check if your input values are within reasonable ranges.")

else:
    # Initial State / Welcome Screen
    st.info("👈 Please adjust the inputs in the sidebar and click **'Analyze Impact'** to generate the report.")
    
    # Placeholder visual to keep UI balanced before interaction
    st.markdown(
        """
        <div style="background-color: #1e293b; border-radius: 12px; padding: 40px; text-align: center; border: 1px dashed #475569;">
            <h3 style="color: #64748b;">Waiting for input...</h3>
            <p style="color: #475569;">Select your pricing parameters to see the profit curve.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import os
# import time

# # Import your existing logic
# from predict_and_plot import PriceSense, PriceInput

# # --------------------------------------------------
# # 1. Page Configuration
# # --------------------------------------------------
# st.set_page_config(
#     page_title="PriceSense Dashboard",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --------------------------------------------------
# # 2. Custom CSS (Dark Theme & Professional UI)
# # --------------------------------------------------
# st.markdown(
#     """
#     <style>
#     /* Main Background */
#     .stApp {
#         background-color: #0f172a;
#         color: #f1f5f9;
#     }

#     /* Sidebar Styling */
#     section[data-testid="stSidebar"] {
#         background-color: #1e293b;
#         border-right: 1px solid #334155;
#     }
    
#     /* Input Labels */
#     .stNumberInput label {
#         color: #cbd5e1 !important;
#         font-weight: 500;
#     }

#     /* KPI Metrics Cards */
#     div[data-testid="metric-container"] {
#         background-color: #1e293b;
#         border: 1px solid #334155;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
#         text-align: center;
#     }

#     /* Metric Values Color Override */
#     div[data-testid="stMetricValue"] {
#         font-size: 2rem !important; 
#     }

#     /* Titles */
#     h1, h2, h3 {
#         color: #f8fafc !important;
#     }
    
#     /* Button Styling */
#     div.stButton > button {
#         background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
#         color: white;
#         border-radius: 8px;
#         height: 3em;
#         font-weight: 600;
#         border: none;
#         width: 100%;
#         transition: all 0.3s ease;
#     }
#     div.stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
#         border: none;
#         color: white;
#     }
    
#     /* Plotly Chart Container */
#     .plot-container {
#         border: 1px solid #334155;
#         border-radius: 12px;
#         padding: 10px;
#         background-color: #1e293b;
#         margin-top: 20px;
#     }
    
#     /* Error Message Styling */
#     .stAlert {
#         border: 1px solid #991b1b;
#         background-color: #450a0a;
#         color: #fca5a5;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --------------------------------------------------
# # 3. Sidebar Inputs
# # --------------------------------------------------
# st.sidebar.markdown("## ⚙️ Model Parameters")
# st.sidebar.markdown("Adjust the values below to simulate scenarios.")

# with st.sidebar.form(key="prediction_form"):
#     st.markdown("### 🏷️ Product Details")
    
#     # 1. Price (Default: 1044.62)
#     price = st.number_input("Price ($)", min_value=1.0, value=1044.62, step=10.0)
    
#     # 2. Competitor Price
#     competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=1045.0, step=10.0)
    
#     # 3. Discount
#     discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
#     st.markdown("---")
#     st.markdown("### 📊 Market Factors")
    
#     # 4. Elasticity Index
#     elasticity = st.number_input("Elasticity Index", value=1.5, step=0.1, help="Higher means more sensitive to price changes")
    
#     # 5. Storage Cost (Mapped to 'Cost')
#     cost = st.number_input("Storage Cost ($)", min_value=0.0, value=450.0, step=10.0)
    
#     # 6. Return Rate (Visual Input)
#     return_rate = st.number_input("Return Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    
#     # 7. Customer Reviews (Visual Input)
#     reviews = st.number_input("Customer Reviews (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

#     st.markdown("---")
#     analyze_btn = st.form_submit_button("⚡ Analyze Impact")

# # --------------------------------------------------
# # 4. Main Dashboard Area
# # --------------------------------------------------

# # Header
# st.title("📊 Price Impact Analysis")
# st.markdown("Visualize how your pricing strategy affects **Profit** and **Revenue** in real-time.")
# st.markdown("---")

# # Instantiate Logic
# try:
#     price_sense = PriceSense()
# except Exception as e:
#     st.error(f"❌ Error loading model: {e}")
#     st.stop()

# if analyze_btn:
#     # --- DYNAMIC "STEP-BY-STEP" LOADING UI ---
#     # We use a placeholder to ensure the loading UI disappears completely once finished
#     loading_placeholder = st.empty()
    
#     try:
#         with loading_placeholder.container():
#             # Step 1
#             with st.spinner("🔍 Validating market parameters..."):
#                 time.sleep(0.6)
            
#             # Step 2
#             with st.spinner("🧠 Running price elasticity simulations..."):
#                 input_data = PriceInput(
#                     price=price,
#                     cost=cost,
#                     elasticity_index=elasticity,
#                     competitor_price=competitor_price,
#                     discount=discount,
#                     return_rate=return_rate,
#                     reviews=reviews
#                 )
#                 demand, revenue, profit = price_sense.predict(input_data)
#                 time.sleep(0.8)
            
#             # Step 3
#             with st.spinner("📈 Generating sensitivity curves..."):
#                 fig, _ = price_sense.plot_profit_curve(input_data)
#                 time.sleep(0.6)
        
#         # Clear the loading placeholder so results appear cleanly
#         loading_placeholder.empty()

#         # --- DISPLAY RESULTS ---
        
#         # 1. Metrics Layout
#         m1, m2, m3 = st.columns(3)
        
#         with m1:
#             st.metric(label="📦 Expected Demand", value=f"{int(demand):,} Units")
        
#         with m2:
#             st.metric(label="💰 Total Revenue", value=f"${revenue:,.2f}")
            
#         with m3:
#             delta_val = f"{(profit/revenue)*100:.1f}% Margin" if revenue > 0 else "0%"
#             st.metric(label="📈 Net Profit", value=f"${profit:,.2f}", delta=delta_val)

#         # 2. Sensitivity Analysis Section
#         st.markdown("<br>", unsafe_allow_html=True)
#         st.markdown("<h2>🔍 Sensitivity Analysis</h2>", unsafe_allow_html=True)
        
#         # Display the Plotly Chart
#         st.plotly_chart(fig, use_container_width=True)
            
#     except Exception as e:
#         loading_placeholder.empty()
#         st.error(f"⚠️ An unexpected error occurred during analysis:\n\n{str(e)}")
#         st.info("💡 Tip: Check if your input values are within reasonable ranges.")

# else:
#     # Initial State / Welcome Screen
#     st.info("👈 Please adjust the inputs in the sidebar and click **'Analyze Impact'** to generate the report.")
    
#     # Placeholder visual
#     st.markdown(
#         """
#         <div style="background-color: #1e293b; border-radius: 12px; padding: 60px; text-align: center; border: 1px dashed #475569; margin-top: 20px;">
#             <h1 style="color: #475569; font-size: 3rem; margin-bottom: 10px;">📊</h1>
#             <h3 style="color: #64748b;">Ready for Analysis</h3>
#             <p style="color: #475569; font-size: 1.1rem;">Configure your pricing model parameters to begin the simulation.</p>
#         </div>
#         """, 
#         unsafe_allow_html=True

#     )
