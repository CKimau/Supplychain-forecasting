from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
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
import calendar
import random

# === Authentication ===
def check_credentials(username, password):
    """Check if username and password match"""
    valid_users = {
        "chris kimau": "password",
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
    logo = None

# === Streamlit Page Setup ===
st.set_page_config(page_title="SForecast", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")

# Proper logo display without DeltaGenerator output
if logo:  # Only try to display if logo exists
    st.sidebar.image(logo, width=88)

# Define tabs in sidebar
app_mode = st.sidebar.radio("", [
    "📈 Forecast Dashboard",
    "🔄 Unit Conversion", 
    "📤 Data Upload",
    "📅 Demand Planning",
    "🔮 Forecasting",
    "🧪 Model Testing",
    "🗃️ Database"
])

# Header
st.title("SForecast - " + app_mode)

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

    c.execute('''CREATE TABLE IF NOT EXISTS production_schedule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        machine TEXT,
        start_time DATETIME,
        end_time DATETIME,
        quantity REAL,
        status TEXT,
        notes TEXT,
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

# === Session State Initialization ===
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'conversion_history' not in st.session_state:
    st.session_state.conversion_history = []
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = []
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = None
if 'consumption_data' not in st.session_state:
    st.session_state.consumption_data = None

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
        .stMetric {{
            background-color: {SECONDARY_BG_COLOR};
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stDataFrame {{
            border-radius: 8px;
        }}
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: {PRIMARY_COLOR}15;
            border-right: 1px solid {PRIMARY_COLOR}30;
        }}
        /* Sidebar radio buttons */
        [data-testid="stSidebar"] .stRadio div div label {{
            padding: 0.5rem 1rem;
            margin: 0.2rem 0;
            border-radius: 0.5rem;
            transition: all 0.2s;
        }}
        [data-testid="stSidebar"] .stRadio div div label:hover {{
            background-color: {PRIMARY_COLOR}20;
        }}
        [data-testid="stSidebar"] .stRadio div div [data-baseweb="radio"]:checked + label {{
            background-color: {PRIMARY_COLOR};
            color: white;
            font-weight: bold;
        }}
        /* Anchor links styling */
        .stMarkdown a {{
            color: {PRIMARY_COLOR};
            font-weight: bold;
        }}
        .interpretation.danger {{
            background-color: #FFEBEE;
            border-left: 4px solid #F44336;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}
        .interpretation.good {{
            background-color: #E8F5E9;
            border-left: 4px solid #4CAF50;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# === Tab Content ===
if app_mode == "📈 Forecast Dashboard":
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
    
    st.markdown("""
    **Next Steps:**
    - Upload your data in the 📤 Data Upload section
    - Convert units in the 🔄 Unit Conversion hub
    - Generate forecasts in the 🔮 Forecasting tab
    """)

elif app_mode == "🔄 Unit Conversion":
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
                
                # Save to session state
                st.session_state.conversion_history.append({
                    "input": f"{input_value} {input_unit}",
                    "output": f"{result:.2f} {output_unit}",
                    "type": "Standard"
                })
                
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
                
                # Save to session state
                st.session_state.conversion_history.append({
                    "input": f"{sheets} sheets, {layers} layers",
                    "output": f"{total_weight:.2f} kg",
                    "type": "Multi-layer"
                })
    
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
    
    # Show recent conversions from session state
    if st.session_state.conversion_history:
        with st.expander("Recent Conversions"):
            st.table(pd.DataFrame(st.session_state.conversion_history[-5:]))
    
    st.markdown("""
    **Connected Features:**
    - Use converted values in 📈 Forecast Dashboard
    - Upload bulk conversions to 🗃️ Database
    """)

elif app_mode == "📤 Data Upload":
    st.header("📤 Data Upload Center")
    
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
                    st.session_state.inventory_data = inv_df
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
                    st.session_state.consumption_data = cons_df
                    st.success("Consumption data saved!")
            except Exception as e:
                st.error(str(e))
    
    st.markdown("""
    **Data Usage:**
    - Use uploaded data in 🔮 Forecasting
    - Analyze in 🧪 Model Testing
    - View in 🗃️ Database
    """)

elif app_mode == "📅 Demand Planning":
    st.header("📅 Demand Planning")
    
    # Show data availability status
    st.sidebar.markdown("### Data Status")
    if st.session_state.inventory_data is not None:
        st.sidebar.success("Inventory Data Loaded")
    if st.session_state.consumption_data is not None:
        st.sidebar.success("Consumption Data Loaded")
    
    # Split view between calendar and scheduler
    view_type = st.radio("View Mode", ["Calendar View", "Scheduler View"], horizontal=True)
    
    if view_type == "Calendar View":
        # Calendar View Section
        st.subheader("🗓️ Forecast Selection Calendar")
        
        # Get current date and set up calendar navigation
        today = datetime.now()
        current_year = today.year
        current_month = today.month
        
        # Create columns for calendar navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            selected_year = st.selectbox("Year", range(current_year, current_year + 5), index=0)
        with col2:
            selected_month = st.selectbox("Month", [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ], index=current_month - 1)
        with col3:
            view_type = st.radio("View", ["Monthly", "Weekly"], horizontal=True, key="calendar_view")
        
        # Convert selected month to number
        month_num = datetime.strptime(selected_month, "%B").month
        
        # Generate calendar data
        if view_type == "Monthly":
            # Create monthly calendar
            cal = calendar.monthcalendar(selected_year, month_num)
            
            # Display calendar header
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            cols = st.columns(7)
            for i, day in enumerate(days):
                cols[i].write(f"**{day}**")
            
            # Display calendar days
            for week in cal:
                cols = st.columns(7)
                for i, day in enumerate(week):
                    if day == 0:
                        cols[i].write(" ")
                    else:
                        date_str = f"{selected_year}-{month_num:02d}-{day:02d}"
                        with cols[i]:
                            # Check if date has forecasts (placeholder logic)
                            has_forecast = random.random() > 0.7  # Replace with actual check
                            
                            if has_forecast:
                                st.markdown(f"""
                                    <div style='border: 2px solid {PRIMARY_COLOR}; border-radius: 5px; padding: 5px; text-align: center;'>
                                        <strong>{day}</strong>
                                        <div style='font-size: 0.3em; color: green;'>Forecast</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='border: 1px solid #ccc; border-radius: 5px; padding: 5px; text-align: center;'>
                                        <strong>{day}</strong>
                                    </div>
                                """, unsafe_allow_html=True)
        else:
            # Weekly view
            st.write("Weekly view coming soon")
        
        # Demand Planning Tools Section
        st.subheader("🛠️ Demand Planning Tools")
        
        tool_col1, tool_col2 = st.columns(2)
        
        with tool_col1:
            st.markdown("**📊 Forecast Summary**")
            # Placeholder data - replace with actual forecasts
            forecast_data = {
                "Material": ["BOPP 35µ", "BOPP 20µ", "White PE"],
                "This Month": [1200, 850, 950],
                "Next Month": [1350, 900, 1000],
                "Variance": ["+12.5%", "+5.9%", "+5.3%"]
            }
            st.dataframe(pd.DataFrame(forecast_data))
            
            st.markdown("**📅 Key Dates**")
            key_dates = {
                "Date": ["2023-11-15", "2023-12-01", "2023-12-15"],
                "Event": ["Inventory Count", "New Product Launch", "Year-End Close"]
            }
            st.dataframe(pd.DataFrame(key_dates))
        
        with tool_col2:
            st.markdown("**🔍 Forecast Comparison**")
            time_period = st.selectbox("Compare", ["Month-over-Month", "Year-over-Year"])
            
            # Placeholder comparison chart
            fig = go.Figure()
            if time_period == "Month-over-Month":
                fig.add_trace(go.Bar(
                    x=["Oct", "Nov", "Dec"],
                    y=[1000, 1200, 1350],
                    name="BOPP 35µ"
                ))
                fig.add_trace(go.Bar(
                    x=["Oct", "Nov", "Dec"],
                    y=[800, 850, 900],
                    name="BOPP 20µ"
                ))
            else:
                fig.add_trace(go.Bar(
                    x=["2022", "2023"],
                    y=[12000, 13500],
                    name="BOPP 35µ"
                ))
                fig.add_trace(go.Bar(
                    x=["2022", "2023"],
                    y=[9500, 10200],
                    name="BOPP 20µ"
                ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Demand Planning Actions
        st.subheader("🚀 Planning Actions")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Refresh Forecasts"):
                st.success("Forecasts refreshed for selected period")
        
        with action_col2:
            if st.button("📧 Export Plan"):
                st.success("Demand plan exported to Excel")
        
        with action_col3:
            if st.button("📌 Create Planning Task"):
                st.success("New planning task created")
    
    else:  # Scheduler View
        st.subheader("🏭 Production Scheduler")
        
        # Timeframe Selection
        timeframe = st.radio("Schedule View", 
                            ["Daily", "Weekly", "Monthly"], 
                            horizontal=True,
                            index=1,
                            key="scheduler_view")
        
        # Get current date and calculate date range
        today = datetime.now().date()
        start_date = st.date_input("Start Date", today, key="scheduler_date")
        
        if timeframe == "Daily":
            end_date = start_date + timedelta(days=1)
        elif timeframe == "Weekly":
            end_date = start_date + timedelta(weeks=1)
        else:  # Monthly
            end_date = start_date + relativedelta(months=1)
        
        # Placeholder production data - replace with real data
        production_data = [
            {
                "Product": "BOPP 35µ",
                "Machine": "Extruder 1",
                "Start": datetime.combine(start_date + timedelta(days=1), datetime.time(8, 0)),
                "End": datetime.combine(start_date + timedelta(days=1), datetime.time(16, 0)),
                "Quantity": 1200,
                "Status": "Scheduled"
            },
            {
                "Product": "BOPP 20µ",
                "Machine": "Extruder 2",
                "Start": datetime.combine(start_date + timedelta(days=2), datetime.time(10, 0)),
                "End": datetime.combine(start_date + timedelta(days=2), datetime.time(18, 0)),
                "Quantity": 800,
                "Status": "Confirmed"
            }
        ]
        
        # Convert to DataFrame
        df_schedule = pd.DataFrame(production_data)
        
        # Display as Gantt chart
        fig = px.timeline(
            df_schedule,
            x_start="Start",
            x_end="End",
            y="Machine",
            color="Product",
            title=f"Production Schedule ({timeframe} view)",
            hover_name="Product",
            hover_data=["Quantity", "Status"]
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
        
        # Production Planning Tools
        st.subheader("🛠️ Scheduling Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📦 Material Requirements**")
            # Placeholder MRP data
            mrp_data = {
                "Material": ["PP Granules", "Additives", "Masterbatch"],
                "Required": [2500, 120, 75],
                "On Hand": [1800, 100, 60],
                "Shortage": [700, 20, 15]
            }
            st.dataframe(pd.DataFrame(mrp_data))
            
            st.markdown("**⚙️ Machine Utilization**")
            utilization_data = {
                "Machine": ["Extruder 1", "Extruder 2", "Coater"],
                "Utilization": ["85%", "78%", "65%"],
                "Status": ["Optimal", "Good", "Underutilized"]
            }
            st.dataframe(pd.DataFrame(utilization_data))
        
        with col2:
            st.markdown("**📊 Schedule Metrics**")
            metrics_col1, metrics_col2 = st.columns(2)
            
            metrics_col1.metric("Scheduled Hours", "156", "+12% vs plan")
            metrics_col1.metric("Changeovers", "8", "3 planned")
            metrics_col2.metric("Utilization", "82%", "2% above target")
            metrics_col2.metric("OEE", "76%", "On track")
            
            st.markdown("**🔍 Schedule Analysis**")
            analysis_option = st.selectbox("View", 
                                         ["Capacity", "Changeovers", "Downtime"],
                                         index=0)
            
            # Placeholder analysis chart
            fig = go.Figure()
            if analysis_option == "Capacity":
                fig.add_trace(go.Bar(
                    x=["Extruder 1", "Extruder 2", "Coater"],
                    y=[85, 78, 65],
                    name="Utilization %"
                ))
            elif analysis_option == "Changeovers":
                fig.add_trace(go.Bar(
                    x=["Mon", "Tue", "Wed", "Thu", "Fri"],
                    y=[3, 2, 1, 2, 0],
                    name="Changeovers"
                ))
            else:
                fig.add_trace(go.Bar(
                    x=["Mechanical", "Electrical", "Cleaning", "Other"],
                    y=[12, 8, 15, 5],
                    name="Downtime Hours"
                ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Schedule Actions
        st.subheader("⚡ Schedule Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Optimize Schedule", key="optimize"):
                st.success("Schedule optimized using available capacity")
        
        with action_col2:
            if st.button("📋 Generate Work Orders", key="work_orders"):
                st.success("Work orders generated for selected period")
        
        with action_col3:
            if st.button("📤 Export Schedule", key="export_schedule"):
                st.success("Production schedule exported to PDF")
    
    st.markdown("""
    **Integration Points:**
    - Uses inventory from 📤 Data Upload
    - Connects to forecasts from 🔮 Forecasting
    """)

elif app_mode == "🔮 Forecasting":
    st.header("📈 Forecasting")

    st.subheader("📤 Upload Historical Stock or Consumption Data")
    uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"], key="forecast_upload")

    # Store uploaded file in session state
    if uploaded_file:
        st.session_state.uploaded_data = uploaded_file
        try:
            if uploaded_file.name.endswith(".csv"):
                for enc in ["utf-8", "ISO-8859-1", "latin1"]:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except:
                        continue
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ File successfully loaded.")
            st.write("📄 Preview of Uploaded Data:")
            st.dataframe(df.head())

            columns = df.columns.tolist()
            date_col = st.selectbox("🗓️ Select Date Column", columns, key="forecast_date_col")
            y_col = st.selectbox("📈 Select Value Column", df.select_dtypes(include='number').columns, key="forecast_value_col")
            item_col = st.selectbox("🏷️ Select Item Description Column (Optional)", ["None"] + columns, key="forecast_item_col")

            selected_item = None
            if item_col != "None":
                items = df[item_col].dropna().unique().tolist()
                selected_item = st.selectbox("🎯 Select Item to Forecast", items, key="forecast_item_select")

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])

            if selected_item:
                df = df[df[item_col] == selected_item]

            df = df[[date_col, y_col]].rename(columns={date_col: "ds", y_col: "y"}).dropna()
            df = df.sort_values("ds")

            st.subheader("⚙️ Forecast Settings")
            method = st.radio("Forecasting Method", ["Prophet", "Holt-Winters"], key="forecast_method")
            horizon_years = st.slider("📅 Forecast Horizon (Years)", 1, 5, 1, key="forecast_horizon")
            freq = st.radio("📆 Forecast Granularity", ["Daily", "Weekly", "Monthly"], key="forecast_freq")
            period_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            forecast_periods = horizon_years * (365 if freq == "Daily" else 52 if freq == "Weekly" else 12)

            if st.button("🔮 Generate Forecast", key="generate_forecast"):
                try:
                    if method == "Prophet":
                        m = Prophet()
                        m.fit(df)
                        future = m.make_future_dataframe(periods=forecast_periods, freq=period_map[freq])
                        forecast = m.predict(future)
                        merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="left")

                    else:  # Holt-Winters
                        df_hw = df.set_index("ds").asfreq(period_map[freq])
                        df_hw["y"] = df_hw["y"].interpolate()
                        model = ExponentialSmoothing(df_hw["y"], trend="add", seasonal="add", seasonal_periods={
                            "D": 7, "W": 52, "M": 12}[period_map[freq]])
                        fitted_model = model.fit()
                        forecast_values = fitted_model.forecast(forecast_periods)
                        future_dates = pd.date_range(start=df_hw.index[-1] + pd.Timedelta(days=1),
                                                     periods=forecast_periods, freq=period_map[freq])
                        forecast = pd.DataFrame({"ds": future_dates, "yhat": forecast_values})
                        merged = pd.concat([df.reset_index(), forecast], ignore_index=True)

                    # Metrics if sufficient data
                    if len(merged.dropna()) > 10:
                        y_actual = merged.dropna(subset=["y", "yhat"])
                        rmse = np.sqrt(mean_squared_error(y_actual["y"], y_actual["yhat"]))
                        mape = np.mean(np.abs((y_actual["y"] - y_actual["yhat"]) / y_actual["y"])) * 100
                        smape_val = 100/len(y_actual) * np.sum(
                            2 * np.abs(y_actual["yhat"] - y_actual["y"]) / (np.abs(y_actual["y"]) + np.abs(y_actual["yhat"])))
                        r2 = r2_score(y_actual["y"], y_actual["yhat"])

                        # Conditional color formatting
                        y_std = y_actual["y"].std()
                        thresholds = {
                            "rmse": ("green" if rmse < 0.5*y_std else "orange" if rmse < y_std else "red"),
                            "mape": ("green" if mape < 10 else "orange" if mape < 20 else "red"),
                            "smape": ("green" if smape_val < 10 else "orange" if smape_val < 20 else "red"),
                            "r2": ("green" if r2 > 0.7 else "orange" if r2 > 0.5 else "red")
                        }

                        st.subheader("📊 Forecast Accuracy Metrics")
                        cols = st.columns(4)

                        def display_metric(col, label, value, color, help_text=""):
                            col.markdown(f"""
                                <div style="
                                    background-color: {color}20;
                                    border-left: 4px solid {color};
                                    padding: 10px;
                                    border-radius: 4px;
                                ">
                                    <div style="font-weight: bold; color: {color}">{label}</div>
                                    <div style="font-size: 12px; font-weight: bold;">{value}</div>
                                    <div style="font-size: 6px; color: #666;">{help_text}</div>
                                </div>""", unsafe_allow_html=True)

                        display_metric(cols[0], "RMSE", f"{rmse:.2f}", thresholds["rmse"], "Lower is better")
                        display_metric(cols[1], "MAPE", f"{mape:.2f}%", thresholds["mape"], "Lower is better")
                        display_metric(cols[2], "SMAPE", f"{smape_val:.2f}%", thresholds["smape"], "Lower is better")
                        display_metric(cols[3], "R² Score", f"{r2:.2f}", thresholds["r2"], "1 is best")

                        # Interpretation
                        interpretation = []
                        if mape > 50:
                            interpretation.append("High MAPE (>50%): Model may not be capturing patterns well.")
                        if r2 < 0.3:
                            interpretation.append("Low R² (<0.3): Model explains little variance.")
                        if rmse > y_std:
                            interpretation.append(f"High RMSE (>std dev of {y_std:.2f}): Large errors.")
                        color_class = "danger" if interpretation else "good"

                        st.markdown(f"""
                            <div class="interpretation {'danger' if interpretation else 'good'}">
                                <strong>{'⚠️' if interpretation else '✓'} Model Interpretation:</strong><br>
                                {'<br>'.join(interpretation) if interpretation else 'Metrics indicate good model performance.'}
                            </div>""", unsafe_allow_html=True)

                    # Forecast Chart
                    st.subheader("📉 Forecast Visualization")
                    fig = px.line(merged, x="ds", y=["y", "yhat"], labels={"value": "Stock/Consumption", "ds": "Date"},
                                  title="Forecast vs Actual", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                    # Export
                    st.download_button("⬇️ Download Forecast CSV", forecast.to_csv(index=False), "forecast_output.csv")

                    # Save forecast to session state
                    if 'forecast_results' not in st.session_state:
                        st.session_state.forecast_results = []
                    st.session_state.forecast_results.append({
                        "item": selected_item if selected_item else "All Items",
                        "method": method,
                        "forecast": forecast.to_dict("records"),
                        "metrics": {"rmse": rmse, "mape": mape, "smape": smape_val, "r2": r2}
                    })
                    st.success("✅ Forecast saved for analysis in Model Testing tab.")

                except Exception as e:
                    st.error(f"❌ Forecasting failed: {e}")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    # Add navigation to test the model
    if st.session_state.get('forecast_results'):
        st.markdown(f"""
        **Next Steps:**
        - Test this model in [🧪 Model Testing](#model-testing)
        - View forecast in [🗃️ Database](#database)
        """, unsafe_allow_html=True)

elif app_mode == "🧪 Model Testing":
    st.header("🧪 Model Testing")

    # Check for available forecast results
    if not st.session_state.get('forecast_results'):
        st.warning("No forecast results available. Please generate forecasts in the 🔮 Forecasting tab first.")
        st.stop()

    st.sidebar.markdown("### Available Forecasts")
    for i, fr in enumerate(st.session_state.forecast_results):
        st.sidebar.markdown(f"{i+1}. {fr.get('item', 'Unnamed')}")

    st.subheader("Test Forecast Accuracy")
    
    selected_forecast = st.selectbox(
        "Select Forecast to Test",
        options=[f"{i+1}. {fr.get('item', 'Unnamed')}" for i, fr in enumerate(st.session_state.forecast_results)],
        index=0
    )
    
    selected_idx = int(selected_forecast.split(".")[0]) - 1
    forecast_data = st.session_state.forecast_results[selected_idx]
    
    st.write(f"Testing forecast for: **{forecast_data['item']}**")
    st.write(f"Method: {forecast_data['method']}")
    
    st.subheader("Train-Test Split Evaluation")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    
    if st.button("Run Evaluation"):
        try:
            # Convert forecast data back to DataFrame
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            
            # Split into train and test
            split_idx = int(len(forecast_df) * (1 - test_size/100))
            train = forecast_df.iloc[:split_idx]
            test = forecast_df.iloc[split_idx:]
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train['ds'], y=train['yhat'],
                name='Train Forecast',
                line=dict(color=PRIMARY_COLOR)
            ))
            fig.add_trace(go.Scatter(
                x=test['ds'], y=test['yhat'],
                name='Test Forecast',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Train-Test Forecast Evaluation",
                xaxis_title="Date",
                yaxis_title="Value"
            )
            st.plotly_chart(fig)
            
            # Calculate metrics
            if 'y' in forecast_df.columns:
                y_true = forecast_df.iloc[split_idx:]['y']
                y_pred = forecast_df.iloc[split_idx:]['yhat']
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                cols = st.columns(2)
                cols[0].metric("Test RMSE", f"{rmse:.2f}")
                cols[1].metric("Test MAPE", f"{mape:.2f}%")
            
            st.success("Evaluation complete!")
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
    
    st.subheader("Advanced Analysis Techniques")
    technique = st.selectbox(
        "Select Analysis Technique",
        ["Residual Analysis", "Error Distribution", "Feature Importance"]
    )
    
    if st.button(f"Run {technique}"):
        try:
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            
            if technique == "Residual Analysis":
                if 'y' in forecast_df.columns:
                    forecast_df['residual'] = forecast_df['y'] - forecast_df['yhat']
                    
                    fig = px.scatter(
                        forecast_df, x='yhat', y='residual',
                        title="Residuals vs Predicted Values",
                        trendline="lowess"
                    )
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig)
                else:
                    st.warning("Actual values (y) not available for residual analysis")
            
            elif technique == "Error Distribution":
                if 'y' in forecast_df.columns:
                    forecast_df['error'] = forecast_df['y'] - forecast_df['yhat']
                    
                    fig = px.histogram(
                        forecast_df, x='error',
                        title="Error Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("Actual values (y) not available for error analysis")
            
            elif technique == "Feature Importance":
                st.info("Feature importance analysis coming soon")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

elif app_mode == "🗃️ Database":
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
            
        if not data.empty:
            st.download_button(
                "Export to CSV",
                data.to_csv(index=False),
                f"{selected_table}_export.csv"
            )
    
    st.markdown("""
    **Database Contents:**
    - View all uploaded and processed data
    - Clear tables as needed
    """)