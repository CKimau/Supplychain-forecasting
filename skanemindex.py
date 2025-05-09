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
logo = Image.open(r"C:\Users\chris.mutuku\OneDrive - Skanem AS\Desktop\logo.jpg")

# === Streamlit Page Setup ===
st.set_page_config(page_title="Skanem Forecasting", layout="wide", page_icon=logo)

# Header
col1, col2 = st.columns([1, 20])
with col1:
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
# Tab 1: Integrated Forecast Dashboard & SKU Simulator (from first code)
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

# Tab 2: Unit Conversion Hub (from first code)
# Tab 2: Unit Conversion Hub (Bulk matches Single conversion fields)
with tab2:
    st.header("🔁 Unit Conversion Hub")
    
    conv_col1, conv_col2 = st.columns([1, 2])
    
    with conv_col1:
        st.subheader("Single Conversion")
        conversion_type = st.radio("Conversion Type", 
                                  ["Standard Units", "Multi-layer Paper Weight"],
                                  help="Choose between standard unit conversions or specialized paper weight calculations")
        
        if conversion_type == "Standard Units":
            # Existing standard unit conversion
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
                    # Convert dimensions to meters
                    area_sqm = (width/100) * (length/100)  # cm² to m²
                    thickness_m = thick * 1e-6  # microns to meters
                    
                    # Weight in kg = area (m²) * thickness (m) * density (g/cm³) * 1000 (conversion factor)
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
                # Load file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Validate required columns
                required_cols = ['input_value', 'input_unit', 'output_unit']
                optional_cols = ['thickness_microns', 'density', 'material_name']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                # Show preview
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Auto-detect if thickness/density needed
                needs_thickness = any(unit in ['kg', 'sqm', 'meters'] for unit in pd.concat([df['input_unit'], df['output_unit']]))
                if needs_thickness and 'thickness_microns' not in df.columns:
                    st.warning("Some conversions require thickness but column not found. Using default 35 microns.")
                    df['thickness_microns'] = 35.0
                
                if 'density' not in df.columns:
                    st.warning("Density column not found. Using default 0.92 g/cm³.")
                    df['density'] = 0.92
                
                if 'material_name' not in df.columns:
                    df['material_name'] = "Bulk Conversion"
                
                # Process conversions
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
                    
                    # Show results
                    st.dataframe(df)
                    
                    # Save to database
                    conn = sqlite3.connect(DB_NAME)
                    df[['material_name', 'input_value', 'input_unit', 
                        'output_value', 'output_unit', 'thickness_microns', 
                        'density']].to_sql('conversions', conn, if_exists='append', index=False)
                    st.success("Saved to database!")
                    
                    # Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        "bulk_conversion_results.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"File processing error: {str(e)}")
# Tab 3: Unified Data Upload Center (from first code)
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

# Tab 4: Prophet Forecasting (from second code)
with tab4:
    st.header("📈Forecasting ")

    uploaded_forecast_file = st.file_uploader("Upload File (.csv or .xlsx)", type=["csv", "xlsx"], key="prophet_upload")

    if uploaded_forecast_file is not None:
        try:
            # Load data
            if uploaded_forecast_file.name.endswith('.csv'):
                encodings = ["utf-8", "ISO-8859-1", "latin1"]
                for enc in encodings:
                    try:
                        forecast_df = pd.read_csv(uploaded_forecast_file, encoding=enc)
                        break
                    except:
                        continue
            else:
                forecast_df = pd.read_excel(uploaded_forecast_file)

            st.success("✅ File uploaded successfully.")
            
            # First filter: Column Filter
            st.subheader("🔍 Step 1: Filter Columns")
            all_columns = forecast_df.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to include in analysis",
                all_columns,
                default=all_columns,
                help="Choose which columns to keep for forecasting"
            )
            
            if not selected_columns:
                st.warning("Please select at least one column")
                st.stop()
                
            filtered_df = forecast_df[selected_columns]
            
            # Second filter: Item Description Filter (if available)
            st.subheader("🔍 Step 2: Filter by Item")
            text_columns = filtered_df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            item_filter_column = None
            item_filter_value = None
            
            if text_columns:
                item_filter_column = st.selectbox(
                    "Select item description column",
                    ["None"] + text_columns,
                    help="Filter data by specific items before forecasting"
                )
                
                if item_filter_column != "None":
                    unique_items = filtered_df[item_filter_column].dropna().unique().tolist()
                    item_filter_value = st.multiselect(
                        f"Select {item_filter_column} values to include",
                        unique_items,
                        default=unique_items[:1] if unique_items else [],
                        help="Select specific items to forecast"
                    )
                    
                    if item_filter_value:
                        filtered_df = filtered_df[filtered_df[item_filter_column].isin(item_filter_value)]
            
            # Data selection for forecasting
            st.subheader("🔍 Step 3: Select Forecast Columns")
            cols = st.columns(2)
            
            with cols[0]:
                # Auto-detect or select date column
                date_options = filtered_df.select_dtypes(include=["object", "datetime64"]).columns.tolist()
                date_col = st.selectbox(
                    "Select Date Column", 
                    date_options,
                    help="Choose column containing dates"
                )
                
            with cols[1]:
                # Select value column
                numeric_options = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                value_col = st.selectbox(
                    "Select Value Column", 
                    numeric_options,
                    help="Choose numeric column to forecast"
                )
            
            # Convert date column and filter NA
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
            filtered_df = filtered_df.dropna(subset=[date_col, value_col])
            
            # Data preview
            st.write("Preview of filtered data:")
            st.dataframe(filtered_df.head())
            
            # Prophet configuration
            st.subheader("⚙️Configuration")
            
            # Model parameters
            growth = st.selectbox("Growth Type", ["linear", "logistic"])
            
            cols = st.columns(3)
            with cols[0]:
                yearly_seasonality = st.checkbox("Yearly Seasonality", True)
            with cols[1]:
                weekly_seasonality = st.checkbox("Weekly Seasonality", True)
            with cols[2]:
                daily_seasonality = st.checkbox("Daily Seasonality", False)
            
            # Advanced options
            with st.expander("Advanced Options"):
                cols = st.columns(2)
                with cols[0]:
                    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
                with cols[1]:
                    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.01)
                
                cols = st.columns(2)
                with cols[0]:
                    n_changepoints = st.slider("Number of Changepoints", 10, 100, 25)
                with cols[1]:
                    changepoint_range = st.slider("Changepoint Range", 0.7, 1.0, 0.8, 0.01)
            
            if st.button("🔮 Generate Forecast"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        df_prophet = filtered_df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
                        
                        # Apply log transform if values are positive
                        if df_prophet['y'].min() > 0:
                            df_prophet['y'] = np.log1p(df_prophet['y'])
                            st.info("Applied log transformation to stabilize variance")
                        
                        # Initialize and configure Prophet
                        m = Prophet(
                            growth=growth,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
                            n_changepoints=n_changepoints,
                            changepoint_range=changepoint_range
                        )
                        
                        # Add monthly seasonality
                        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                        
                        # Fit model
                        m.fit(df_prophet)
                        
                        # Create future dataframe
                        future = m.make_future_dataframe(periods=30)
                        
                        # Forecast
                        forecast = m.predict(future)
                        
                        # Reverse log transform if applied
                        if df_prophet['y'].min() > 0:
                            forecast['yhat'] = np.expm1(forecast['yhat'])
                            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
                            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
                            df_prophet['y'] = np.expm1(df_prophet['y'])
                        
                        # Clip negative values
                        forecast['yhat'] = forecast['yhat'].clip(lower=0)
                        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
                        
                        # Calculate metrics
                        merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='inner')
                        
                        if len(merged) > 10:
                            rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
                            mape = safe_mape(merged['y'], merged['yhat'])
                            smape_val = smape(merged['y'], merged['yhat'])
                            r2 = r2_score(merged['y'], merged['yhat'])
                            
                            st.subheader("📊 Forecast Accuracy Metrics")
                            cols = st.columns(4)
                            cols[0].metric("RMSE", f"{rmse:.2f}", help="Lower is better")
                            cols[1].metric("MAPE", f"{mape:.2f}%", help="Lower is better")
                            cols[2].metric("SMAPE", f"{smape_val:.2f}%", help="Lower is better")
                            cols[3].metric("R² Score", f"{r2:.2f}", help="Closer to 1 is better")
                            
                            # Interpretation guide
                            if mape > 50:
                                st.warning("⚠️ High MAPE: Model may not be capturing patterns well")
                            if r2 < 0.3:
                                st.warning("⚠️ Low R²: Model explains little variance in the data")
                        
                        # Plot forecast
                        st.subheader("📉 Forecast Plot")
                        fig = plot_plotly(m, forecast)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show components
                        st.subheader("🧩 Forecast Components")
                        fig2 = m.plot_components(forecast)
                        st.pyplot(fig2)
                        
                        # Show forecast data
                        st.subheader("📋 Forecast Data (Next 30 Days)")
                        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
                        forecast_output.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        
                        # Add item description back if filtered by item
                        if item_filter_column and item_filter_value:
                            forecast_output[item_filter_column] = ", ".join(item_filter_value)
                        
                        st.dataframe(forecast_output)
                        
                        # Download
                        csv = forecast_output.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Forecast CSV", 
                            csv, 
                            "filtered_forecast.csv", 
                            "text/csv",
                            help="Download the forecast data as CSV"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Forecasting failed: {str(e)}")
        
        except Exception as e:
            st.error(f"⚠️ Error processing file: {str(e)}")
# Tab 5: Train-Test Split (from second code)
with tab5:
    st.header("🧪 Train-Test Split")
    
    if 'uploaded_forecast_file' in globals() and uploaded_forecast_file is not None:
        try:
            if uploaded_forecast_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_forecast_file)
            else:
                df = pd.read_excel(uploaded_forecast_file)
            
            st.success("Using uploaded data from Prophet tab")
            
            # ==================== NEW ITEM FILTERING ====================
            st.subheader("🔍 Item Selection")
            
            # 1. First filter: Select item description column
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            item_desc_col = st.selectbox(
                "Select item description column", 
                ["None"] + text_cols,
                help="Column containing product/SKU descriptions"
            )
            
            filtered_df = df.copy()
            
            # 2. Second filter: Select specific items if description column chosen
            if item_desc_col != "None":
                unique_items = df[item_desc_col].dropna().unique().tolist()
                selected_items = st.multiselect(
                    f"Select items to analyze from '{item_desc_col}'",
                    unique_items,
                    default=unique_items[:1] if unique_items else []
                )
                
                if selected_items:
                    filtered_df = df[df[item_desc_col].isin(selected_items)]
            
            # ==================== DATA SELECTION ====================
            st.subheader("📈 Time Series Selection")
            cols = st.columns(2)
            with cols[0]:
                date_col = st.selectbox(
                    "Select Date Column", 
                    filtered_df.select_dtypes(include=["object", "datetime64"]).columns.tolist()
                )
            with cols[1]:
                value_col = st.selectbox(
                    "Select Value Column", 
                    filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                )
            
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
            filtered_df = filtered_df.sort_values(date_col).dropna(subset=[date_col, value_col])
            
            # ==================== TIME AGGREGATION ====================
            st.subheader("⏱ Time Aggregation Level")
            time_agg = st.radio(
                "Aggregate data by:",
                ["Daily", "Weekly", "Monthly", "Yearly"],
                horizontal=True
            )
            
            # Resample based on selection
            agg_df = filtered_df.set_index(date_col).groupby(item_desc_col if item_desc_col != "None" else None)[value_col]
            
            if time_agg == "Daily":
                resampled_df = agg_df.resample('D').mean()
            elif time_agg == "Weekly":
                resampled_df = agg_df.resample('W-MON').mean()
            elif time_agg == "Monthly":
                resampled_df = agg_df.resample('MS').mean()
            else:  # Yearly
                resampled_df = agg_df.resample('YS').mean()
            
            resampled_df = resampled_df.reset_index()
            
            # ==================== SUPERVISED TRAIN-TEST SPLIT ====================
            st.subheader("📊 Supervised Forecast Evaluation")
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
            split_idx = int(len(resampled_df) * (1 - test_size/100))
            train, test = resampled_df.iloc[:split_idx], resampled_df.iloc[split_idx:]
            
            if st.button("Run Supervised Evaluation"):
                with st.spinner("Training Prophet model..."):
                    m = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=(time_agg in ["Daily", "Weekly"]),
                        daily_seasonality=(time_agg == "Daily")
                    )
                    
                    train_prophet = train[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
                    m.fit(train_prophet)
                    
                    future = m.make_future_dataframe(periods=len(test), freq=time_agg[0])
                    forecast = m.predict(future)
                    
                    # Visualize
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train[date_col], y=train[value_col], name="Train"))
                    fig.add_trace(go.Scatter(x=test[date_col], y=test[value_col], name="Test"))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast"))
                    fig.update_layout(title=f"{time_agg} Forecast for {selected_items[0] if selected_items else 'All Items'}")
                    st.plotly_chart(fig)
            
            # ==================== UNSUPERVISED ANALYSIS ====================
            st.subheader("🕵️ Unsupervised Time Series Analysis")
            
            unsup_method = st.selectbox(
                "Select Technique", 
                ["Anomaly Detection", "Seasonal Decomposition", "Demand Clustering"],
                help="Choose unsupervised learning approach"
            )
            
            if unsup_method == "Anomaly Detection":
                if st.button("Detect Temporal Anomalies"):
                    from sklearn.ensemble import IsolationForest
                    
                    # Create time features
                    X = resampled_df.copy()
                    X['day_of_week'] = X[date_col].dt.dayofweek
                    X['month'] = X[date_col].dt.month
                    X['value_lag1'] = X[value_col].shift(1)
                    
                    # Train model
                    clf = IsolationForest(contamination=0.05)
                    anomalies = clf.fit_predict(X[[value_col, 'day_of_week', 'month', 'value_lag1']].dropna())
                    
                    # Visualize
                    fig = px.scatter(
                        X, x=date_col, y=value_col, 
                        color=anomalies == -1,
                        title=f"Anomalies in {time_agg} {value_col}",
                        color_discrete_map={True: 'red', False: 'blue'}
                    )
                    st.plotly_chart(fig)
            
            elif unsup_method == "Seasonal Decomposition":
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                if st.button("Decompose Seasonality"):
                    # Ensure regular frequency
                    ts = resampled_df.set_index(date_col)[value_col].asfreq(
                        'D' if time_agg == "Daily" else 
                        'W' if time_agg == "Weekly" else
                        'MS' if time_agg == "Monthly" else 'YS'
                    ).ffill()
                    
                    result = seasonal_decompose(ts, model='additive', period=12 if time_agg == "Monthly" else 4)
                    
                    # Plot components
                    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(12,8))
                    result.observed.plot(ax=ax1, title='Observed')
                    result.trend.plot(ax=ax2, title='Trend')
                    result.seasonal.plot(ax=ax3, title='Seasonal')
                    result.resid.plot(ax=ax4, title='Residual')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            elif unsup_method == "Demand Clustering":
                n_clusters = st.slider("Number of Clusters", 2, 5, 3)
                
                if st.button("Cluster Demand Patterns"):
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    # Create features
                    cluster_df = resampled_df.copy()
                    cluster_df['rolling_mean'] = cluster_df[value_col].rolling(4).mean()
                    cluster_df['pct_change'] = cluster_df[value_col].pct_change()
                    cluster_df = cluster_df.dropna()
                    
                    # Cluster
                    X = StandardScaler().fit_transform(cluster_df[[value_col, 'rolling_mean', 'pct_change']])
                    kmeans = KMeans(n_clusters=n_clusters)
                    cluster_df['cluster'] = kmeans.fit_predict(X)
                    
                    # Visualize clusters
                    fig = px.scatter(
                        cluster_df, x=date_col, y=value_col,
                        color='cluster',
                        title=f"{time_agg} Demand Clusters (k={n_clusters})"
                    )
                    st.plotly_chart(fig)
                    
                    # Cluster characteristics
                    st.write("Cluster Profiles:")
                    cluster_stats = cluster_df.groupby('cluster').agg({
                        value_col: ['mean', 'std', 'count'],
                        'pct_change': 'mean'
                    })
                    st.dataframe(cluster_stats)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please upload data in the Prophet Forecasting tab first")
# Tab 6: Database Viewer (from first code)
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