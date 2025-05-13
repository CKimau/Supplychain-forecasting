# === Imports ===
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import sqlite3
import os
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from PIL import Image
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# === Authentication ===
def check_credentials(username, password):
    """Check if username and password match"""
    # In a real app, use proper password hashing and database storage
    valid_users = {
        "chris kimau": "password",  # Note: In production, use hashed passwords
        "admin": "admin123"
    }
    return valid_users.get(username.lower()) == password

def authenticate():
    """Handle authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Skanem Forecasting - Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.stop()

# Call authentication at the beginning
authenticate()

# === Constants ===
DB_NAME = "skanem_forecasting.db"
PRIMARY_COLOR = "#0E4E4E"
BG_COLOR = "#E1EBAE"
TEXT_COLOR = "#31333F"
SECONDARY_BG_COLOR = "#F0F2F6"

# === Error Metric Functions ===
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    return 2 * np.mean(np.abs(y_pred - y_true) / np.maximum(denominator, 1e-8)) * 100

# === Load Logo ===
try:
    logo = Image.open(r"C:\Users\chris.mutuku\OneDrive - Skanem AS\Desktop\logo.jpg")
except:
    logo = None  # Fallback if logo not found

# === Streamlit Page Setup ===
st.set_page_config(page_title="Skanem Forecasting", layout="wide", page_icon=logo)

# Header
col1, col2 = st.columns([1, 20])
with col1:
    if logo:
        st.image(logo, width=88)
with col2:
    st.title("Skanem Forecasting")

# === Custom CSS ===
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BG_COLOR};
        }}
        .css-1d391kg, .css-1oe5cao {{
            background-color: {PRIMARY_COLOR} !important;
        }}
        .stTextInput>label, .stNumberInput>label, .stSelectbox>label,
        .stMultiselect>label, .stRadio>label, .stSlider>label,
        .stFileUploader>label, .stDateInput>label {{
            color: {TEXT_COLOR};
            font-weight: bold;
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
        }}
        .stButton>button:hover {{
            background-color: #1A6A6A;
            color: white;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
            background-color: {SECONDARY_BG_COLOR};
            color: {TEXT_COLOR};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {PRIMARY_COLOR} !important;
            color: white !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #1A6A6A;
            color: white;
        }}
        .stMetric {{
            background-color: {SECONDARY_BG_COLOR};
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stDataFrame {{
            border-radius: 8px;
        }}
    </style>
""", unsafe_allow_html=True)

# === Database Initialization ===
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS conversions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        input_value REAL,
        input_unit TEXT,
        output_value REAL,
        output_unit TEXT,
        thickness_microns REAL,
        density REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        forecast_type TEXT,
        horizon TEXT,
        rmse REAL,
        mape REAL,
        r2 REAL,
        forecast_data TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_name TEXT,
        simulation_params TEXT,
        results TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS uploaded_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        data_type TEXT,
        columns TEXT,
        row_count INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        quantity REAL,
        unit TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS consumption (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        material_name TEXT,
        date DATE,
        quantity REAL,
        unit TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

# === Unit Conversion Logic ===
def kg_to_sqm(kg, thickness_microns, density=0.92):
    thickness_m = thickness_microns * 1e-6
    return kg / (density * thickness_m)

def kg_to_meters(kg, width_m, thickness_microns, density=0.92):
    sqm = kg_to_sqm(kg, thickness_microns, density)
    return sqm / width_m

def kg_to_liters(kg, density=0.92):
    return kg / density

def convert_units(value, from_unit, to_unit, **kwargs):
    converters = {
        ('kg', 'sqm'): lambda x: kg_to_sqm(x, kwargs.get('thickness_microns', 35), kwargs.get('density', 0.92)),
        ('kg', 'meters'): lambda x: kg_to_meters(x, kwargs.get('width_m', 1), kwargs.get('thickness_microns', 35), kwargs.get('density', 0.92)),
        ('kg', 'liters'): lambda x: kg_to_liters(x, kwargs.get('density', 0.92)),
        ('sqm', 'kg'): lambda x: x * (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('meters', 'kg'): lambda x: x * kwargs.get('width_m', 1) * (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('liters', 'kg'): lambda x: x * kwargs.get('density', 0.92)
    }
    return converters.get((from_unit, to_unit), lambda x: x)(value)

# === Main Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecast Dashboard & SKU Simulator",
    "🔄 Unit Conversion",
    "📤 Unified Upload Center",
    "📈 Forecasting",
    "🧪 Train-Test Split",
    "🗃️ Database Viewer"
])

# Tab 1: Forecast Dashboard & SKU Simulator
with tab1:
    st.header("Forecast Overview")
    
    with st.expander("Basic Forecasting"):
        col1, col2 = st.columns(2)
        
        with col1:
            material_name = st.text_input("Material Name", "White BOPP 35 Mic Film 35", key="tab1_material")
            current_balance = st.number_input("Current Available Balance (kg)", min_value=0.0, value=1000.0, step=1.0, key="tab1_balance")
            avg_daily_consumption = st.number_input("Average Daily Consumption (kg)", min_value=0.0, value=50.0, step=1.0, key="tab1_consumption")
        
        with col2:
            consumption_variability = st.slider("Consumption Variability (%)", 0, 50, 10, key="tab1_variability")
            safety_stock = st.number_input("Safety Stock Level (kg)", min_value=0.0, value=200.0, step=1.0, key="tab1_safety")
            lead_time = st.number_input("Lead Time (days)", min_value=1, value=7, step=1, key="tab1_lead")
        
        forecast_horizon = st.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days", "6 months", "1 year", "5 years"], index=0, key="tab1_horizon")
        
        # KPI calculations
        days_until_stockout = int(current_balance / avg_daily_consumption) if avg_daily_consumption > 0 else 0
        stockout_date = (datetime.now() + timedelta(days=days_until_stockout)).strftime("%Y-%m-%d")
        reorder_point = safety_stock + (lead_time * avg_daily_consumption)
        days_until_reorder = int((current_balance - reorder_point) / avg_daily_consumption) if current_balance > reorder_point and avg_daily_consumption > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Balance", f"{current_balance:.2f} kg")
        col2.metric("Days Until Stockout", days_until_stockout, f"Expected by {stockout_date}")
        col3.metric("Reorder Point", f"{reorder_point:.2f} kg", f"{days_until_reorder} days until reorder" if days_until_reorder > 0 else "Below reorder point!")
        col4.metric("Avg Daily Consumption", f"{avg_daily_consumption:.2f} kg", f"±{consumption_variability}% variability")
        
        # Forecast generation
        if forecast_horizon.endswith("days"):
            horizon_days = int(forecast_horizon.split(" ")[0])
            dates = pd.date_range(datetime.now(), periods=horizon_days)
        elif forecast_horizon == "6 months":
            horizon_days = 180
            dates = pd.date_range(datetime.now(), periods=horizon_days)
        elif forecast_horizon == "1 year":
            horizon_days = 365
            dates = pd.date_range(datetime.now(), periods=horizon_days)
        elif forecast_horizon == "5 years":
            horizon_days = 365 * 5
            dates = pd.date_range(datetime.now(), periods=horizon_days)
        
        forecast_deterministic = [max(0, current_balance - (i * avg_daily_consumption)) for i in range(horizon_days)]
        
        np.random.seed(42)
        daily_variation = 1 + (np.random.rand(horizon_days) - 0.5) * (consumption_variability / 100)
        forecast_probabilistic = [max(0, current_balance - np.sum(avg_daily_consumption * daily_variation[:i+1])) for i in range(horizon_days)]
        
        df_forecast = pd.DataFrame({
            "Date": dates,
            "Deterministic Forecast": forecast_deterministic,
            "Probabilistic Forecast": forecast_probabilistic,
            "Reorder Point": reorder_point,
            "Safety Stock": safety_stock
        })
        
        df_melted = df_forecast.melt(
            id_vars="Date",
            value_vars=["Deterministic Forecast", "Probabilistic Forecast", "Reorder Point", "Safety Stock"],
            var_name="Metric",
            value_name="Value"
        )
        
        fig = px.line(df_melted, x="Date", y="Value", color="Metric",
                      title=f"Material Forecast: {material_name}",
                      labels={"Value": "Quantity (kg)", "Date": "Date"},
                      template="plotly_white")
        fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Stockout Level", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("💾 Save Forecast to Database"):
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            
            forecast_data = df_forecast.to_json(orient='records')
            
            c.execute('''INSERT INTO forecasts 
                         (material_name, forecast_type, horizon, forecast_data) 
                         VALUES (?, ?, ?, ?)''',
                         (material_name, "Deterministic", forecast_horizon, forecast_data))
            
            conn.commit()
            conn.close()
            st.success("Forecast saved to database!")

# Tab 2: Unit Conversion Hub
with tab2:
    st.header("🔁 Unit Conversion Hub")
    
    conv_col1, conv_col2 = st.columns([1, 2])
    
    with conv_col1:
        st.subheader("Single Conversion")
        conversion_type = st.radio("Conversion Type", 
                                  ["Standard Units", "Multi-layer Paper Weight"],
                                  help="Choose between standard unit conversions or specialized paper weight calculations")
        
        if conversion_type == "Standard Units":
            input_value = st.number_input("Input Value", min_value=0.0, value=1000.0)
            input_unit = st.selectbox("From Unit", ["kg", "sqm", "meters", "liters"])
            output_unit = st.selectbox("To Unit", ["kg", "sqm", "meters", "liters"])
            
            if input_unit in ["kg", "sqm", "meters"]:
                thickness = st.number_input("Thickness (microns)", value=35.0)
                density = st.number_input("Density (g/cm³)", value=0.92)
            else:
                thickness = 35.0
                density = 0.92
            
            if st.button("Convert Single"):
                result = convert_units(input_value, input_unit, output_unit,
                                    thickness_microns=thickness, density=density)
                st.metric("Result", f"{result:.2f} {output_unit}")
                
        else:  # Multi-layer Paper Weight Calculation
            st.markdown("**Paper Dimensions**")
            cols = st.columns(3)
            with cols[0]:
                width = st.number_input("Width (cm)", min_value=0.1, value=21.0)
            with cols[1]:
                length = st.number_input("Length (cm)", min_value=0.1, value=29.7)
            with cols[2]:
                sheets = st.number_input("Number of Sheets", min_value=1, value=1)
            
            st.markdown("**Layer Properties**")
            layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3)
            
            layer_props = []
            for i in range(layers):
                with st.expander(f"Layer {i+1} Properties"):
                    cols = st.columns(2)
                    with cols[0]:
                        thickness = st.number_input(f"Thickness (microns) - Layer {i+1}", 
                                                  min_value=1.0, value=35.0, key=f"thick_{i}")
                    with cols[1]:
                        density = st.number_input(f"Density (g/cm³) - Layer {i+1}", 
                                                min_value=0.1, value=0.92, key=f"density_{i}")
                    layer_props.append((thickness, density))
            
            if st.button("Calculate Total Weight"):
                total_weight = 0
                for thick, density in layer_props:
                    area_sqm = (width/100) * (length/100)  # cm² to m²
                    thickness_m = thick * 1e-6  # microns to meters
                    layer_weight = area_sqm * thickness_m * density * 1000
                    total_weight += layer_weight
                
                total_weight *= sheets  # Multiply by number of sheets
                st.metric("Total Weight", f"{total_weight:.4f} kg")
                st.metric("Weight per Sheet", f"{total_weight/sheets:.4f} kg")
    
    with conv_col2:
        st.subheader("Bulk Conversion")
        uploaded_file = st.file_uploader("Upload CSV/XLSX with columns matching these fields:", 
                                       type=["csv", "xlsx"],
                                       help="File must contain: input_value, input_unit, output_unit, thickness_microns (if applicable), density")
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_cols = ['input_value', 'input_unit', 'output_unit']
                optional_cols = ['thickness_microns', 'density', 'material_name']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                needs_thickness = any(unit in ['kg', 'sqm', 'meters'] for unit in pd.concat([df['input_unit'], df['output_unit']]))
                if needs_thickness and 'thickness_microns' not in df.columns:
                    st.warning("Some conversions require thickness but column not found. Using default 35 microns.")
                    df['thickness_microns'] = 35.0
                
                if 'density' not in df.columns:
                    st.warning("Density column not found. Using default 0.92 g/cm³.")
                    df['density'] = 0.92
                
                if 'material_name' not in df.columns:
                    df['material_name'] = "Bulk Conversion"
                
                if st.button("⚡ Convert All Rows"):
                    results = []
                    for _, row in df.iterrows():
                        try:
                            result = convert_units(
                                row['input_value'],
                                row['input_unit'],
                                row['output_unit'],
                                thickness_microns=row.get('thickness_microns', 35.0),
                                density=row.get('density', 0.92)
                            )
                            results.append(result)
                        except Exception as e:
                            st.warning(f"Row {_+1} failed: {str(e)}")
                            results.append(None)
                    
                    df['output_value'] = results
                    st.success(f"Converted {len(df)} rows!")
                    
                    st.dataframe(df)
                    
                    conn = sqlite3.connect(DB_NAME)
                    df[['material_name', 'input_value', 'input_unit', 
                        'output_value', 'output_unit', 'thickness_microns', 
                        'density']].to_sql('conversions', conn, if_exists='append', index=False)
                    st.success("Saved to database!")
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        "bulk_conversion_results.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"File processing error: {str(e)}")

# Tab 3: Unified Data Upload Center
with tab3:
    st.header("📤 Unified Data Upload Center")
    
    upload_tabs = st.tabs(["Inventory Data", "Consumption Data", "Other Data"])
    
    with upload_tabs[0]:
        st.subheader("Inventory Upload")
        inv_file = st.file_uploader("Upload current inventory (CSV/XLSX)", type=["csv", "xlsx"], key="inv_upload")
        if inv_file:
            try:
                if inv_file.name.endswith('.csv'):
                    inv_df = pd.read_csv(inv_file)
                else:
                    inv_df = pd.read_excel(inv_file)
                
                st.success(f"Uploaded {len(inv_df)} inventory records")
                st.dataframe(inv_df)
                
                if st.button("Save Inventory"):
                    conn = sqlite3.connect(DB_NAME)
                    inv_df.to_sql('inventory', conn, if_exists='replace', index=False)
                    st.success("Inventory data saved!")
            except Exception as e:
                st.error(str(e))
    
    with upload_tabs[1]:
        st.subheader("Consumption Upload")
        cons_file = st.file_uploader("Upload consumption data (CSV/XLSX)", type=["csv", "xlsx"], key="cons_upload")
        if cons_file:
            try:
                if cons_file.name.endswith('.csv'):
                    cons_df = pd.read_csv(cons_file)
                else:
                    cons_df = pd.read_excel(cons_file)
                
                st.success(f"Uploaded {len(cons_df)} consumption records")
                st.dataframe(cons_df)
                
                if st.button("Save Consumption"):
                    conn = sqlite3.connect(DB_NAME)
                    cons_df.to_sql('consumption', conn, if_exists='replace', index=False)
                    st.success("Consumption data saved!")
            except Exception as e:
                st.error(str(e))
# Helper functions
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
# Tab 4: Prophet Forecasting
with tab4:
    st.header("📈 Robust Prophet Forecasting")

    uploaded_forecast_file = st.file_uploader("Upload Time Series Data", type=["csv", "xlsx"], key="prophet_upload")

    if uploaded_forecast_file is not None:
        try:
            if uploaded_forecast_file.name.endswith('.csv'):
                forecast_df = pd.read_csv(uploaded_forecast_file)
            else:
                forecast_df = pd.read_excel(uploaded_forecast_file)

            if len(forecast_df) < 10:
                st.error("Insufficient data: Need at least 10 observations")
                st.stop()

            st.success("✅ Data loaded successfully")

            st.subheader("🔍 Data Validation")

            date_col = st.selectbox("Select Date Column", 
                                    forecast_df.select_dtypes(include=["object", "datetime64"]).columns.tolist())
            numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns.tolist()
            value_col = st.selectbox("Select Value Column", numeric_cols)

            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors='coerce')
            forecast_df = forecast_df.dropna(subset=[date_col]).sort_values(date_col)

            st.subheader("🧹 Data Cleansing")

            if (forecast_df[value_col] <= 0).any():
                st.warning("Negative/zero values detected - applying log transform requires positive values")
                min_val = forecast_df[value_col][forecast_df[value_col] > 0].min() / 2 if (forecast_df[value_col] > 0).any() else 0.1
                forecast_df[value_col] = forecast_df[value_col].clip(lower=min_val)

            st.markdown("**Outlier Management**")
            cap_method = st.radio("Method", 
                                  ["IQR Capping (recommended)", "Percentile Capping", "None"],
                                  index=0)

            if cap_method == "IQR Capping (recommended)":
                Q1 = forecast_df[value_col].quantile(0.25)
                Q3 = forecast_df[value_col].quantile(0.75)
                IQR = Q3 - Q1
                forecast_df[value_col] = forecast_df[value_col].clip(
                    Q1 - 1.5*IQR, 
                    Q3 + 1.5*IQR
                )
            elif cap_method == "Percentile Capping":
                forecast_df[value_col] = forecast_df[value_col].clip(
                    forecast_df[value_col].quantile(0.02),
                    forecast_df[value_col].quantile(0.98)
                )

            st.write("Cleaned Data Statistics:")
            st.dataframe(forecast_df[value_col].describe().to_frame().T)

            st.subheader("⚙️ Model Configuration")

            m = Prophet(
                growth='linear',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative',
                n_changepoints=min(15, len(forecast_df)//10),
                mcmc_samples=0
            )

            if len(forecast_df) >= 90:
                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            st.markdown("### ⏳ Forecast Horizon")
            forecast_unit = st.selectbox("Select Forecast Duration Unit", ["Days", "Months", "Years"], index=0)
            forecast_value = st.number_input(f"Enter number of {forecast_unit.lower()} to forecast", min_value=1, step=1, value=90)

            if st.button("🚀 Generate Forecast", key="forecast_button"):
                with st.spinner("Training model with robust initialization..."):
                    try:
                        prophet_df = forecast_df[[date_col, value_col]].copy()
                        prophet_df.columns = ['ds', 'y']
                        prophet_df = prophet_df.dropna()

                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                m.fit(prophet_df)
                                break
                            except Exception as e:
                                if attempt == max_attempts - 1:
                                    raise
                                st.warning(f"Initialization attempt {attempt+1} failed, retrying...")
                                m.changepoint_prior_scale *= 0.8
                                time.sleep(1)

                        # Calculate number of days from selected unit
                        if forecast_unit == "Days":
                            period_days = forecast_value
                        elif forecast_unit == "Months":
                            period_days = forecast_value * 30
                        else:
                            period_days = forecast_value * 365

                        future = m.make_future_dataframe(periods=period_days)
                        forecast = m.predict(future)

                        st.subheader("📊 Forecast Results")
                        fig1 = plot_plotly(m, forecast)
                        st.plotly_chart(fig1, use_container_width=True)

                        merged = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds')

                        if len(merged) > 10:
                            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
                            mape = safe_mape(merged['y'], merged['yhat'])
                            smape_val = smape(merged['y'], merged['yhat'])
                            r2 = r2_score(merged['y'], merged['yhat'])

                            st.subheader("📊 Forecast Accuracy Metrics")

                            RMSE_THRESHOLD = 0.1 * merged['y'].mean()
                            MAPE_THRESHOLD = 20
                            SMAPE_THRESHOLD = 20
                            R2_THRESHOLD = 0.7

                            cols = st.columns(4)

                            rmse_color = "green" if rmse <= RMSE_THRESHOLD else "red"
                            mape_color = "green" if mape <= MAPE_THRESHOLD else "red"
                            smape_color = "green" if smape_val <= SMAPE_THRESHOLD else "red"
                            r2_color = "green" if r2 >= R2_THRESHOLD else "red"

                            cols[0].markdown(f"""
                            <div style="
                                background-color: #f9f9f9;
                                border-radius: 8px;
                                padding: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border-left: 5px solid {rmse_color};
                            ">
                                <div style="font-size: 0.8em; color: #666;">RMSE</div>
                                <div style="font-size: 1.5em; color: {rmse_color};">{rmse:.2f}</div>
                                <div style="font-size: 0.7em; color: #666;">Lower is better</div>
                            </div>
                            """, unsafe_allow_html=True)

                            cols[1].markdown(f"""
                            <div style="
                                background-color: #f9f9f9;
                                border-radius: 8px;
                                padding: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border-left: 5px solid {mape_color};
                            ">
                                <div style="font-size: 0.8em; color: #666;">MAPE</div>
                                <div style="font-size: 1.5em; color: {mape_color};">{mape:.2f}%</div>
                                <div style="font-size: 0.7em; color: #666;">Lower is better</div>
                            </div>
                            """, unsafe_allow_html=True)

                            cols[2].markdown(f"""
                            <div style="
                                background-color: #f9f9f9;
                                border-radius: 8px;
                                padding: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border-left: 5px solid {smape_color};
                            ">
                                <div style="font-size: 0.8em; color: #666;">SMAPE</div>
                                <div style="font-size: 1.5em; color: {smape_color};">{smape_val:.2f}%</div>
                                <div style="font-size: 0.7em; color: #666;">Lower is better</div>
                            </div>
                            """, unsafe_allow_html=True)

                            cols[3].markdown(f"""
                            <div style="
                                background-color: #f9f9f9;
                                border-radius: 8px;
                                padding: 12px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border-left: 5px solid {r2_color};
                            ">
                                <div style="font-size: 0.8em; color: #666;">R² Score</div>
                                <div style="font-size: 1.5em; color: {r2_color};">{r2:.2f}</div>
                                <div style="font-size: 0.7em; color: #666;">Closer to 1 is better</div>
                            </div>
                            """, unsafe_allow_html=True)

                            if mape > MAPE_THRESHOLD:
                                st.warning("⚠️ High MAPE: Model may not be capturing patterns well")
                            if r2 < R2_THRESHOLD:
                                st.warning("⚠️ Low R²: Model explains little variance in the data")

                    except Exception as e:
                        st.error(f"""
                        ❌ Forecasting failed: {str(e)}

                        Common fixes:
                        1. Check for missing/irregular dates
                        2. Try different outlier handling
                        3. Reduce changepoint_prior_scale
                        4. Ensure sufficient historical data
                        """)

        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
    else:
        st.warning("Please upload time series data to begin")
# Tab 5: Train-Test Split
with tab5:
    st.header("🧪 Train-Test Split & Unsupervised Analysis")
    
    # Check if data is available from Tab4
    if 'forecast_df' not in globals() or forecast_df.empty:
        st.warning("Please upload and process data in the Prophet Forecasting tab first")
        st.stop()
    
    try:
        # ==================== DATA PREPARATION ====================
        st.subheader("🔍 Data Preparation")
        
        # Use the already processed data from Tab4
        df = forecast_df.copy()
        
        # Show basic info
        st.write(f"Working with {len(df)} records from {df[date_col].min().date()} to {df[date_col].max().date()}")
        
        # ==================== SUPERVISED TRAIN-TEST SPLIT ====================
        st.subheader("📊 Supervised Train-Test Split")
        
        # Split configuration
        cols = st.columns(2)
        with cols[0]:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        with cols[1]:
            split_date = st.date_input("Or select exact split date", 
                                     value=df[date_col].iloc[int(len(df)*0.8)].to_pydatetime(),
                                     min_value=df[date_col].min(),
                                     max_value=df[date_col].max())
        
        # Create splits
        if st.button("Create Split"):
            try:
                split_idx = int(len(df) * (1 - test_size/100))
                train = df.iloc[:split_idx]
                test = df.iloc[split_idx:]
                
                # Visualize split
                fig = px.line(df, x=date_col, y=value_col, title="Train-Test Split")
                fig.add_vline(x=train[date_col].iloc[-1], line_dash="dash", line_color="red")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Split created: {len(train)} train, {len(test)} test records")
                
                # ==================== MODEL EVALUATION ====================
                st.subheader("📈 Model Evaluation")
                
                # Initialize Prophet
                m = Prophet(
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                
                # Fit on train
                train_df = train[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
                m.fit(train_df)
                
                # Predict on test
                future = m.make_future_dataframe(periods=len(test))
                forecast = m.predict(future)
                
                # Merge actuals and predictions
                results = test[[date_col, value_col]].merge(
                    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    left_on=date_col, right_on='ds'
                )
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(results[value_col], results['yhat']))
                mape = safe_mape(results[value_col], results['yhat'])
                r2 = r2_score(results[value_col], results['yhat'])
                
                # Display metrics
                cols = st.columns(3)
                cols[0].metric("RMSE", f"{rmse:.2f}")
                cols[1].metric("MAPE", f"{mape:.2f}%")
                cols[2].metric("R² Score", f"{r2:.2f}")
                
                # Plot results
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train[date_col], y=train[value_col], name="Train"))
                fig.add_trace(go.Scatter(x=test[date_col], y=test[value_col], name="Test Actual"))
                fig.add_trace(go.Scatter(x=results['ds'], y=results['yhat'], name="Test Predicted"))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Model evaluation failed: {str(e)}")
        
        # ==================== UNSUPERVISED ANALYSIS ====================
        st.subheader("🕵️ Unsupervised Analysis")
        
        analysis_type = st.selectbox("Select Analysis Type", 
                                   ["Anomaly Detection", 
                                    "Demand Pattern Clustering",
                                    "Seasonal Decomposition"])
        
        if analysis_type == "Anomaly Detection":
            st.subheader("🔍 Anomaly Detection")
            
            from sklearn.ensemble import IsolationForest
            
            # Prepare features
            X = df[[value_col]].copy()
            X['day_of_week'] = df[date_col].dt.dayofweek
            X['month'] = df[date_col].dt.month
            
            # Model config
            contamination = st.slider("Expected Anomaly %", 0.1, 10.0, 1.0)
            
            if st.button("Detect Anomalies"):
                clf = IsolationForest(contamination=contamination/100, random_state=42)
                df['anomaly_score'] = clf.fit_predict(X)
                df['is_anomaly'] = df['anomaly_score'] == -1
                
                # Visualize
                fig = px.scatter(df, x=date_col, y=value_col, 
                               color='is_anomaly',
                               title=f"Anomaly Detection ({contamination}% threshold)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Anomaly Details:")
                st.dataframe(df[df['is_anomaly']].sort_values(date_col))
        
        elif analysis_type == "Demand Pattern Clustering":
            st.subheader("📊 Demand Pattern Clustering")
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Create features
            cluster_df = df.set_index(date_col)[value_col].resample('W').mean().reset_index()
            cluster_df['rolling_4w'] = cluster_df[value_col].rolling(4).mean()
            cluster_df = cluster_df.dropna()
            
            # Model config
            n_clusters = st.slider("Number of Clusters", 2, 5, 3)
            
            if st.button("Run Clustering"):
                X = StandardScaler().fit_transform(cluster_df[[value_col, 'rolling_4w']])
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_df['cluster'] = kmeans.fit_predict(X)
                
                # Visualize
                fig = px.scatter(cluster_df, x=date_col, y=value_col,
                               color='cluster', 
                               title=f"Demand Clusters (k={n_clusters})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster stats
                st.write("Cluster Characteristics:")
                cluster_stats = cluster_df.groupby('cluster').agg({
                    value_col: ['mean', 'std', 'count'],
                    'rolling_4w': 'mean'
                })
                st.dataframe(cluster_stats)
        
        elif analysis_type == "Seasonal Decomposition":
            st.subheader("📅 Seasonal Decomposition")
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Resample to consistent frequency
            ts = df.set_index(date_col)[value_col].asfreq('D').ffill()
            
            if st.button("Decompose"):
                result = seasonal_decompose(ts, model='additive', period=7)
                
                # Plot components
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,8))
                result.observed.plot(ax=ax1, title='Observed')
                result.trend.plot(ax=ax2, title='Trend')
                result.seasonal.plot(ax=ax3, title='Seasonal')
                result.resid.plot(ax=ax4, title='Residual')
                plt.tight_layout()
                st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
# Tab 6: Database Viewer
with tab6:
    st.header("🗃️ Database Content Viewer")
    
    conn = sqlite3.connect(DB_NAME)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    selected_table = st.selectbox("Select Table", tables['name'])
    
    if selected_table:
        data = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
        st.dataframe(data)
        
        if st.button(f"Clear {selected_table}"):
            conn.execute(f"DELETE FROM {selected_table}")
            conn.commit()
            st.success("Table cleared!")