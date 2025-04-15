import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import os
import pickle
import chardet  # For file encoding detection

# Set page config - MUST be first Streamlit command
st.set_page_config(page_title="SKANEM FORECASTING", layout="wide")

# --------------------------
# App Configuration
# --------------------------

# Logo and title
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://via.placeholder.com/44", width=44)  # Replace with your logo path
with col2:
    st.title("Advanced Supply Chain Forecasting")

# --------------------------
# Data Management
# --------------------------
DATA_DIR = "forecast_data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_material_data(material_name, data):
    path = os.path.join(DATA_DIR, f"{material_name.replace(' ', '_')}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_material_data(material_name):
    path = os.path.join(DATA_DIR, f"{material_name.replace(' ', '_')}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def get_saved_materials():
    return [f.replace('.pkl', '').replace('_', ' ') for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]

def detect_encoding(file):
    rawdata = file.read()
    result = chardet.detect(rawdata)
    file.seek(0)  # Reset file pointer
    return result['encoding']

# --------------------------
# Forecasting Models
# --------------------------
def calculate_metrics(actual, predicted):
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
        'R2': r2_score(actual, predicted)
    }

def generate_forecast(current_balance, avg_consumption, variability, horizon):
    np.random.seed(42)
    dates = pd.date_range(datetime.now(), periods=horizon)
    
    deterministic = [max(0, current_balance - (i * avg_consumption)) for i in range(horizon)]
    daily_variation = 1 + (np.random.rand(horizon) - 0.5) * (variability/100)
    probabilistic = [max(0, current_balance - np.sum(avg_consumption * daily_variation[:i+1])) for i in range(horizon)]
    
    return dates, deterministic, probabilistic

def train_supervised_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def apply_unsupervised_learning(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# --------------------------
# SKU Consumption Tracker Model
# --------------------------
def sku_consumption_tracker():
    st.title("📊 SKU Consumption Tracker")
    
    uploaded_file = st.file_uploader(
        "Upload SKU Sales Data (CSV/Excel)", 
        type=['csv', 'xlsx'],
        help="Upload data with columns: Product Number, ProdDescr, Jan, Feb, Mar, etc."
    )
    
    if uploaded_file:
        try:
            # Detect file type and encoding
            if uploaded_file.name.endswith('.csv'):
                encoding = detect_encoding(uploaded_file)
                sku_data = pd.read_csv(uploaded_file, encoding=encoding)
            else:
                sku_data = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if not all(col in sku_data.columns for col in ['Product Number', 'ProdDescr'] + required_months):
                st.error("Required columns missing. Need: Product Number, ProdDescr, and all month columns")
            else:
                st.success("Data loaded successfully!")
                st.dataframe(sku_data.head())
                
                # Melt data for time series analysis
                melted_data = pd.melt(
                    sku_data, 
                    id_vars=['Product Number', 'ProdDescr'],
                    value_vars=required_months,
                    var_name='Month',
                    value_name='Sales'
                )
                
                # Convert month names to datetime
                month_map = {month: i+1 for i, month in enumerate(required_months)}
                melted_data['MonthNum'] = melted_data['Month'].map(month_map)
                melted_data['Year'] = 2023  # Assuming current year
                melted_data['Date'] = pd.to_datetime(
                    melted_data['Year'].astype(str) + '-' + 
                    melted_data['MonthNum'].astype(str) + '-01'
                )
                
                # Show sales trends
                selected_sku = st.selectbox(
                    "Select SKU to analyze",
                    options=sku_data['ProdDescr'].unique()
                )
                
                sku_sales = melted_data[melted_data['ProdDescr'] == selected_sku]
                
                fig = px.line(
                    sku_sales,
                    x='Date',
                    y='Sales',
                    title=f"Monthly Sales Trend for {selected_sku}",
                    markers=True
                )
                st.plotly_chart(fig)
                
                # Forecast future consumption
                if st.button("Forecast Future Consumption"):
                    with st.spinner("Training forecasting model..."):
                        try:
                            # Prepare data for forecasting
                            X = sku_sales[['MonthNum']]
                            y = sku_sales['Sales']
                            
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X, y)
                            
                            # Predict next 6 months
                            future_months = pd.DataFrame({
                                'MonthNum': range(13, 19),
                                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                                'Year': [2024]*6
                            })
                            
                            future_months['Date'] = pd.to_datetime(
                                future_months['Year'].astype(str) + '-' + 
                                future_months['MonthNum'].astype(str) + '-01'
                            )
                            future_months['Forecast'] = model.predict(future_months[['MonthNum']])
                            
                            # Combine actual and forecast
                            combined = pd.concat([
                                sku_sales[['Date', 'Sales']].rename(columns={'Sales': 'Value'}),
                                future_months[['Date', 'Forecast']].rename(columns={'Forecast': 'Value'})
                            ])
                            combined['Type'] = ['Actual']*len(sku_sales) + ['Forecast']*len(future_months)
                            
                            # Plot results
                            fig_forecast = px.line(
                                combined,
                                x='Date',
                                y='Value',
                                color='Type',
                                title=f"Sales Forecast for {selected_sku}",
                                markers=True
                            )
                            st.plotly_chart(fig_forecast)
                            
                            st.dataframe(future_months[['Month', 'Year', 'Forecast']])
                            
                        except Exception as e:
                            st.error(f"Error in forecasting: {str(e)}")
                            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# --------------------------
# UI Components
# --------------------------
# Initialize session states
if 'current_stocks' not in st.session_state:
    st.session_state.current_stocks = pd.DataFrame()
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = pd.DataFrame()
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
    ["Dashboard", "Monthly View", "Model Performance", "ML Insights", "Advanced Forecasting", "SKU Consumption Tracker"])

# Page routing
if page == "Dashboard":
    st.title("📊 Dashboard")
    st.write("Welcome to the dashboard.")
elif page == "Monthly View":
    st.title("📅 Monthly View")
    st.write("Monthly forecasting analysis here.")
elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write("Model metrics and evaluation.")
elif page == "ML Insights":
    st.title("🤖 Machine Learning Insights")
    st.write("Results from ML modeling.")
elif page == "Advanced Forecasting":
    # [Rest of your Advanced Forecasting code here...]
    # Make sure to update the file uploader to handle encoding:
    with st.sidebar.expander("📥 Upload Current Inventory", expanded=True):
        uploaded_stocks = st.file_uploader(
            "Upload current inventory (CSV/Excel)", 
            type=['csv', 'xlsx'],
            help="Upload file with columns: 'Item Description' and 'Quantity In Sqr Meters'"
        )
        
        if uploaded_stocks is not None:
            try:
                if uploaded_stocks.name.endswith('.csv'):
                    encoding = detect_encoding(uploaded_stocks)
                    st.session_state.current_stocks = pd.read_csv(uploaded_stocks, encoding=encoding)
                else:
                    st.session_state.current_stocks = pd.read_excel(uploaded_stocks)
                
                st.success(f"Uploaded {len(st.session_state.current_stocks)} records")
                
                # Validate columns
                required_cols = {'Item Description', 'Quantity In Sqr Meters'}
                if not required_cols.issubset(st.session_state.current_stocks.columns):
                    missing = required_cols - set(st.session_state.current_stocks.columns)
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.dataframe(st.session_state.current_stocks.head(3))
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # [Rest of your Advanced Forecasting implementation...]
    
elif page == "SKU Consumption Tracker":
    sku_consumption_tracker()

# --------------------------
# About Section
# --------------------------
st.sidebar.divider()
with st.sidebar.expander("About"):
    st.write("""
    **SKANEM Supply Chain Forecasting Tool**  
    Version 2.1  
    Developed for SKANEM AS  
    
    Features:  
    - Inventory forecasting  
    - SKU consumption tracking  
    - Machine learning insights  
    - Time-series analysis  
    
    © 2025 SKANEM AS
    """)