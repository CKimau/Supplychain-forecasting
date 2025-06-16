from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
import streamlit as st
# === Streamlit Config ===
st.set_page_config(
    page_title="SForecast",
    layout="wide",
    page_icon="C:/Users/access.control/Documents/Forecasting/logo.jpg"
)
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
import hashlib
from scipy import stats

# This must come FIRST before any other Streamlit code
# st.set_page_config(page_title="Skanem Forecast", layout="wide")  # Removed duplicate
DB_NAME = "skanem_forecasting.db"
PRIMARY_COLOR = "#0E4E4E"
BG_COLOR = "#E1EBAE"
TEXT_COLOR = "#31333F"
SECONDARY_BG_COLOR = "#F0F2F6"

# === Password hashing ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# === Enhanced user table ===
def init_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_user_table()

# === Authentication ===
def check_credentials(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username.lower(),))
    result = c.fetchone()
    conn.close()
    return result and hash_password(password) == result[0]

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username.lower(), hash_password(password)))
        conn.commit()
        conn.close()
        return True, "User registered successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def authenticate():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'mode' not in st.session_state:
        st.session_state.mode = "login"

    if not st.session_state.authenticated:
        st.title("SForecasting - Login")
        st.sidebar.title("Login")
        try:
            # logo = Image.open(...) replaced by display_logo()
            display_logo()
        except:
            pass

        if st.session_state.mode == "login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login")
                if login_btn:
                    if check_credentials(username, password):
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            if st.button("Sign Up"):
                st.session_state.mode = "signup"
                st.rerun()

        else:
            with st.form("signup_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                signup_btn = st.form_submit_button("Sign Up")
                if signup_btn:
                    success, msg = register_user(new_username, new_password)
                    if success:
                        st.success(msg)
                        st.session_state.mode = "login"
                        st.rerun()
                    else:
                        st.error(msg)
            if st.button("Back to Login"):
                st.session_state.mode = "login"
                st.rerun()
        st.stop()

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

def display_logo():
    try:
        # logo = Image.open(...) replaced by display_logo()
        display_logo()
    except:
        st.warning("Logo not found")

try:
    logo = Image.open(r"C:/Users/access.control/Documents/Forecasting/logo.jpg")
except:
    logo = None

# === Streamlit Page Setup ===
# st.set_page_config(page_title="Skanem Forecast", layout="wide")  # Removed duplicate

# Sidebar Navigation
tab_labels = ["📈 Forecast Dashboard", "🔄 Unit Conversion", "📤 Data Upload", "📅 Demand Planning", "🔮 Forecasting", "🧪 Model Testing", "🗃️ Database"]
tabs = st.tabs(tab_labels)
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
with tabs[0]:
    st.header("Forecast Overview")
    cols = st.columns([6, 1])
    try:
        # logo = Image.open(...) replaced by display_logo()
        display_logo()
    except:
        st.warning("Logo not found")
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

with tabs[1]:
  cols = st.columns([6, 1])
   
  with cols[1]:
      try:
          # logo = Image.open(...) replaced by display_logo()
          display_logo()
      except:
          st.warning("Logo not found")
  
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

with tabs[2]:
    st.header("📤 Data Upload Center")
    
    upload_tabs = st.tabs(["Inventory Data", "Consumption Data", "Other Data"])

    with upload_tabs[2]:
        st.subheader("Forecast Upload")
        forecast_file = st.file_uploader("Upload Forecast Data (CSV/XLSX)", type=["csv", "xlsx"], key="forecast_upload_tab2")

        if forecast_file:
            try:
                if forecast_file.name.endswith('.csv'):
                    forecast_df = pd.read_csv(forecast_file)
                else:
                    forecast_df = pd.read_excel(forecast_file)

                st.success(f"Uploaded {len(forecast_df)} forecast records")
                st.dataframe(forecast_df)

                if st.button("Save Forecast Upload"):
                    conn = sqlite3.connect(DB_NAME)
                    forecast_df.to_sql('uploaded_forecast', conn, if_exists='replace', index=False)
                    st.session_state.uploaded_forecast_data = forecast_df
                    st.success("Forecast data saved!")
            except Exception as e:
                st.error(str(e))

    
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

with tabs[3]:
    st.header("📅 Demand Planning")
    
    # Show data availability status
    st.sidebar.markdown("### Data Status")
    cols = st.columns([6, 1])
    with cols[1]:
        try:
            # logo = Image.open(...) replaced by display_logo()
            display_logo()
        except:
            st.warning("Logo not found")
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
with tabs[4]:
    # Add logo
    cols = st.columns([6, 1])
    with cols[0]:
        st.header("📈 Forecasting")
    with cols[1]:
        try:
            # logo = Image.open(...) replaced by display_logo()
            display_logo()
        except:
            st.warning("Logo not found")

    st.subheader("1. Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file with historical data", 
                                   type=["csv", "xlsx"], 
                                   key="forecast_upload")

    # Initialize with empty dataframe
    df = pd.DataFrame()
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            st.success("✅ Data loaded successfully")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

    if not df.empty:
        st.subheader("2. Configure Your Forecast")
        
        # Column selection
        cols = st.columns(2)
        with cols[0]:
            date_col = st.selectbox("Select Date Column", 
                                  df.columns, 
                                  key="forecast_date_col")
        with cols[1]:
            value_col = st.selectbox("Select Value Column", 
                                   df.select_dtypes(include='number').columns, 
                                   key="forecast_value_col")
        
        # Optional item filter
        item_col = st.selectbox("Filter by Item (optional)", 
                              ["No filter"] + [c for c in df.columns if c not in [date_col, value_col]],
                              key="forecast_item_col")
        
        selected_item = None
        if item_col != "No filter":
            selected_item = st.selectbox("Select specific item to forecast", 
                                       df[item_col].unique(),
                                       key="forecast_item_select")

        # Prepare data
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            if selected_item:
                df = df[df[item_col] == selected_item]
            
            df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
            df = df.dropna().sort_values("ds")
            
        except Exception as e:
            st.error(f"❌ Error preparing data: {e}")
            st.stop()

        st.subheader("3. Forecast Settings")
        
        # Forecast configuration
        method = st.radio("Forecasting method", 
                         ["Prophet (recommended)", "Holt-Winters"],
                         horizontal=True)
        
        cols = st.columns(2)
        with cols[0]:
            horizon = st.slider("Forecast period (months)", 
                              1, 24, 6)
        with cols[1]:
            freq = st.radio("Frequency", 
                          ["Daily", "Weekly", "Monthly"], 
                          horizontal=True)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Creating forecast..."):
                try:
                    # Forecasting
                    if method.startswith("Prophet"):
                        m = Prophet()
                        m.fit(df)
                        future = m.make_future_dataframe(periods=horizon, freq=freq[0])
                        forecast = m.predict(future)
                    else:
                        # Holt-Winters implementation
                        pass
                    
                    # Merge actuals and forecast
                    result = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')
                    
                    # Ensure no negative forecasts
                    result['yhat'] = result['yhat'].clip(lower=0)
                    result['yhat_lower'] = result['yhat_lower'].clip(lower=0)
                    result['yhat_upper'] = result['yhat_upper'].clip(lower=0)
                    
                    # Forecast Preview - show only selected item if filtered
                    st.subheader("📋 Forecast Preview")
                    preview_df = result.tail(horizon).copy()
                    
                    # Add item column back if filtered
                    if selected_item:
                        preview_df[item_col] = selected_item
                        preview_df = preview_df[[item_col, 'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
                    
                    st.dataframe(preview_df.style.format({
                        'yhat': '{:.2f}',
                        'yhat_lower': '{:.2f}',
                        'yhat_upper': '{:.2f}'
                    }))
                    
                    # Visualization
                    st.subheader("📊 Forecast Results")
                    
                    fig = go.Figure()
                    # Actual values
                    fig.add_trace(go.Scatter(
                        x=result['ds'], y=result['y'],
                        name='Actual',
                        line=dict(color='#1f77b4'),
                        mode='lines+markers'
                    ))
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=result['ds'], y=result['yhat'],
                        name='Forecast',
                        line=dict(color='#ff7f0e')
                    ))
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=result['ds'], y=result['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=result['ds'], y=result['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f'Forecast vs Actuals{" - " + selected_item if selected_item else ""}',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Accuracy Metrics with conditional formatting
                    if 'y' in result.columns:
                        actuals = result.dropna(subset=['y', 'yhat'])
                        if len(actuals) > 0:
                            st.subheader("🔍 Forecast Accuracy")
                            
                            mape = mean_absolute_percentage_error(actuals['y'], actuals['yhat'])
                            rmse = np.sqrt(mean_squared_error(actuals['y'], actuals['yhat']))
                            r2 = r2_score(actuals['y'], actuals['yhat'])
                            
                            # Determine colors based on thresholds
                            def get_color(metric, value):
                                if metric == 'mape':
                                    return 'green' if value < 10 else 'orange' if value < 20 else 'red'
                                elif metric == 'rmse':
                                    y_std = actuals['y'].std()
                                    return 'green' if value < 0.5*y_std else 'orange' if value < y_std else 'red'
                                elif metric == 'r2':
                                    return 'green' if value > 0.7 else 'orange' if value > 0.5 else 'red'
                                return 'gray'
                            
                            cols = st.columns(3)
                            metrics = [
                                ('MAPE', f"{mape:.1f}%", get_color('mape', mape)),
                                ('RMSE', f"{rmse:.2f}", get_color('rmse', rmse)),
                                ('R²', f"{r2:.2f}", get_color('r2', r2))
                            ]
                            
                            for col, (label, value, color) in zip(cols, metrics):
                                col.markdown(f"""
                                    <div style="
                                        border-left: 4px solid {color};
                                        padding: 8px;
                                        background-color: {color}10;
                                        border-radius: 4px;
                                        margin-bottom: 10px;
                                    ">
                                        <div style="font-size: 0.8em; color: #666;">{label}</div>
                                        <div style="font-size: 1.2em; font-weight: bold; color: {color}">{value}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                    
                    # Save to database
                    def save_to_database(forecast_data):
                        # Replace with your actual database connection
                        try:
                            conn = sqlite3.connect('forecasts.db')
                            forecast_data.to_sql('forecasts', conn, if_exists='append', index=False)
                            conn.close()
                            return True
                        except Exception as e:
                            st.error(f"Database error: {e}")
                            return False
                    
                    # Prepare download data
                    download_df = result.copy()
                    if selected_item:
                        download_df[item_col] = selected_item
                    
                    # Download button
                    st.download_button(
                        "⬇️ Download Forecast",
                        download_df.to_csv(index=False),
                        f"forecast_{selected_item if selected_item else 'all'}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
                    
                    # Save to session state and database
                    forecast_name = f"{selected_item if selected_item else 'All Items'} - {method.split()[0]}"
                    if 'forecast_results' not in st.session_state:
                        st.session_state.forecast_results = []
                    
                    forecast_result = {
                        'item': selected_item if selected_item else 'All Items',
                        'method': method.split()[0],
                        'forecast': result.to_dict('records'),
                        'metrics': {
                            'mape': mape,
                            'rmse': rmse,
                            'r2': r2
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.forecast_results.append(forecast_result)
                    
                    if save_to_database(download_df):
                        st.success("✅ Forecast saved to database!")
                    else:
                        st.warning("Forecast saved locally but not in database")
                    
                except Exception as e:
                    st.error(f"Forecast failed: {str(e)}")

    else:
        st.info("ℹ️ Please upload your data to begin forecasting")
with tabs[5]:
    # Add logo to the top right corner
    cols = st.columns([6, 1])
    with cols[0]:
        st.header("🧪 Model Testing & Insights")
    with cols[1]:
        try:
            # logo = Image.open(...) replaced by display_logo()
            display_logo()
        except:
            st.warning("Logo not found")

    # Initialize session state if empty
    if 'forecast_results' not in st.session_state:
        sample_forecast_data = {
            "forecast": pd.DataFrame({
                "ds": pd.date_range(start="2023-01-01", periods=24, freq="M"),
                "yhat": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 
                        150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 
                        200, 205, 210, 215],
                "y": [98, 107, 112, 113, 122, 123, 128, 137, 142, 143, 
                      152, 153, 158, 167, 172, 173, 182, 183, 188, 197, 
                      202, 203, 212, 213]
            }).to_dict("records"),
            "item": "Sample Product",
            "method": "Prophet",
            "metrics": {"rmse": 5.23, "mape": 3.45, "smape": 3.67, "r2": 0.92}
        }
        st.session_state.forecast_results = [sample_forecast_data]
        st.warning("Using sample data for demonstration. Generate forecasts in the Forecasting tab first.")

    # Sidebar with available forecasts
    st.sidebar.markdown("### Available Forecasts")
    for i, fr in enumerate(st.session_state.forecast_results):
        item_name = fr.get('item', 'Unnamed')
        method = fr.get('method', 'Unknown')
        st.sidebar.markdown(f"**{i+1}. {item_name}**")
        st.sidebar.caption(f"Method: {method}")

    # Main content
    st.subheader("1. Select Forecast for Analysis")
    select_options = [f"{i+1}. {fr.get('item', 'Unnamed')}" 
                     for i, fr in enumerate(st.session_state.forecast_results)]
    selected_forecast = st.selectbox("Choose forecast", select_options, index=0)
    
    try:
        selected_idx = int(selected_forecast.split(".")[0]) - 1
        forecast_data = st.session_state.forecast_results[selected_idx]
    except Exception as e:
        st.error(f"Error selecting forecast: {e}")
        st.stop()

    # Convert forecast data to DataFrame
    forecast_df = pd.DataFrame(forecast_data['forecast'])
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    # Display basic info
    item_name = forecast_data.get('item', 'Unnamed Item')
    st.write(f"**Analyzing forecast for:** {item_name}")
    st.write(f"**Forecasting method:** {forecast_data.get('method', 'Unknown')}")

    # Split into historical and future periods
    historical = forecast_df[forecast_df['y'].notna()]
    future = forecast_df[forecast_df['y'].isna()]

    st.subheader("2. Unsupervised Pattern Analysis")
    
    analysis_type = st.selectbox("Select analysis technique", 
                                ["Seasonal Decomposition", 
                                 "Anomaly Detection", 
                                 "Demand Clustering",
                                 "Trend Analysis"])
    
    if st.button("Run Analysis"):
        with st.spinner("Analyzing patterns..."):
            try:
                if analysis_type == "Seasonal Decomposition":
                    # Time series decomposition
                    st.markdown("### 📊 Seasonal-Trend Decomposition")
                    
                    # Ensure data is properly indexed
                    ts_data = historical.set_index('ds')['y']
                    ts_data = ts_data.asfreq('D').fillna(method='ffill')
                    
                    decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                    
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
                    decomposition.observed.plot(ax=ax1, title='Observed')
                    decomposition.trend.plot(ax=ax2, title='Trend')
                    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
                    decomposition.resid.plot(ax=ax4, title='Residual')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Generate insights
                    trend_slope = np.polyfit(range(len(decomposition.trend.dropna())), 
                                    decomposition.trend.dropna(), 1)[0]
                    
                    seasonal_impact = decomposition.seasonal.max() - decomposition.seasonal.min()
                    
                    st.markdown("#### 🔍 Insights")
                    cols = st.columns(2)
                    cols[0].metric("Trend Direction", 
                                  "Upward" if trend_slope > 0 else "Downward", 
                                  f"{trend_slope:.2f} slope")
                    cols[1].metric("Seasonal Impact", 
                                  f"{seasonal_impact:.1f} units", 
                                  "High" if seasonal_impact > historical['y'].std() else "Moderate")
                    
                    st.markdown("""
                    **Recommendations:**
                    - {} trend suggests {} inventory levels
                    - Seasonal pattern indicates {} during peak periods
                    """.format(
                        "Upward" if trend_slope > 0 else "Downward",
                        "increasing" if trend_slope > 0 else "decreasing",
                        "buffer stock needed" if seasonal_impact > historical['y'].std() else "moderate adjustments"
                    ))
                
                elif analysis_type == "Anomaly Detection":
                    st.markdown("### 🚨 Anomaly Detection")
                    
                    # Prepare data for anomaly detection
                    X = historical[['y']].values
                    
                    # Train isolation forest
                    clf = IsolationForest(contamination=0.05, random_state=42)
                    clf.fit(X)
                    historical['anomaly_score'] = clf.decision_function(X)
                    historical['anomaly'] = clf.predict(X)
                    
                    # Plot anomalies
                    fig = px.scatter(historical, x='ds', y='y', 
                                    color='anomaly', 
                                    color_discrete_map={-1: 'red', 1: 'green'},
                                    title="Anomaly Detection in Historical Data")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate insights
                    anomalies = historical[historical['anomaly'] == -1]
                    st.write(f"Detected {len(anomalies)} anomalies in historical data")
                    
                    if not anomalies.empty:
                        st.markdown("#### 📌 Anomaly Details")
                        st.dataframe(anomalies[['ds', 'y']].sort_values('y', ascending=False))
                        
                        st.markdown("#### 🔍 Insights")
                        st.write("""
                        **Potential Causes:**
                        - Data collection errors on {}
                        - Special events/promotions
                        - Supply chain disruptions
                        
                        **Recommendations:**
                        - Investigate root causes for anomalies
                        - Consider removing or adjusting anomalies before re-forecasting
                        - Implement anomaly detection monitoring
                        """.format(anomalies['ds'].dt.strftime('%Y-%m-%d').tolist()[0]))
                
                elif analysis_type == "Demand Clustering":
                    st.markdown("### 🔍 Demand Pattern Clustering")
                    
                    # Create features for clustering (day of week, month, etc.)
                    historical['day_of_week'] = historical['ds'].dt.dayofweek
                    historical['month'] = historical['ds'].dt.month
                    historical['quarter'] = historical['ds'].dt.quarter
                    
                    # Standardize features
                    features = historical[['y', 'day_of_week', 'month']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(features)
                    
                    # Determine optimal clusters
                    wcss = []
                    for i in range(1, 6):
                        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                        kmeans.fit(X_scaled)
                        wcss.append(kmeans.inertia_)
                    
                    # Plot elbow curve
                    fig1, ax = plt.subplots()
                    ax.plot(range(1, 6), wcss)
                    ax.set_title('Elbow Method for Optimal Clusters')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('WCSS')
                    st.pyplot(fig1)
                    
                    # Cluster with optimal k (using 3 for demo)
                    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
                    historical['cluster'] = kmeans.fit_predict(X_scaled)
                    
                    # Plot clusters
                    fig2 = px.scatter(historical, x='ds', y='y', 
                                     color='cluster', 
                                     title="Demand Pattern Clusters")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Cluster analysis
                    cluster_stats = historical.groupby('cluster')['y'].describe()
                    st.markdown("#### 📊 Cluster Statistics")
                    st.dataframe(cluster_stats)
                    
                    st.markdown("#### 🔍 Insights")
                    st.write("""
                    **Pattern Identification:**
                    - Cluster {}: High demand periods (avg {:.1f})
                    - Cluster {}: Medium demand periods (avg {:.1f})
                    - Cluster {}: Low demand periods (avg {:.1f})
                    
                    **Recommendations:**
                    - Differentiate inventory policies by demand cluster
                    - Plan promotions during low demand clusters
                    - Increase safety stock before high demand clusters
                    """.format(
                        cluster_stats.idxmax()['mean'], cluster_stats.max()['mean'],
                        cluster_stats.median()['mean'], cluster_stats.median()['mean'],
                        cluster_stats.idxmin()['mean'], cluster_stats.min()['mean']
                    ))
                
                elif analysis_type == "Trend Analysis":
                    st.markdown("### 📈 Advanced Trend Analysis")
                    
                    # Calculate rolling statistics
                    historical['7day_avg'] = historical['y'].rolling(window=7).mean()
                    historical['28day_avg'] = historical['y'].rolling(window=28).mean()
                    
                    # Plot trends
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], 
                                           name='Actual', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['7day_avg'], 
                                           name='7-Day Avg', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['28day_avg'], 
                                           name='28-Day Avg', line=dict(color='green')))
                    fig.update_layout(title="Short-Term vs Long-Term Trends",
                                    xaxis_title='Date',
                                    yaxis_title='Value')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend metrics
                    short_term_change = (historical['7day_avg'].iloc[-1] - historical['7day_avg'].iloc[-7]) / historical['7day_avg'].iloc[-7] * 100
                    long_term_change = (historical['28day_avg'].iloc[-1] - historical['28day_avg'].iloc[-28]) / historical['28day_avg'].iloc[-28] * 100
                    
                    st.markdown("#### 📊 Trend Metrics")
                    cols = st.columns(2)
                    cols[0].metric("7-Day Trend", 
                                 f"{short_term_change:.1f}%", 
                                 "Increasing" if short_term_change > 0 else "Decreasing")
                    cols[1].metric("28-Day Trend", 
                                 f"{long_term_change:.1f}%", 
                                 "Increasing" if long_term_change > 0 else "Decreasing")
                    
                    st.markdown("#### 🔍 Insights")
                    st.write("""
                    **Trend Interpretation:**
                    - Short-term trend: {}
                    - Long-term trend: {}
                    
                    **Recommendations:**
                    - {}
                    - {}
                    """.format(
                        "growth" if short_term_change > 0 else "decline",
                        "sustained growth" if long_term_change > 0 else "prolonged decline",
                        "Increase production capacity" if short_term_change > 5 and long_term_change > 5 else 
                        "Maintain current levels" if abs(short_term_change) < 5 and abs(long_term_change) < 5 else
                        "Investigate causes of decline",
                        "Plan promotions to boost demand" if short_term_change < -5 else
                        "Consider gradual inventory reduction" if long_term_change < -5 else
                        "Monitor for trend confirmation"
                    ))
                
                # Add forecast comparison to historical patterns
                st.subheader("3. Forecast Evaluation")
                
                if not future.empty:
                    # Calculate forecast change from last historical value
                    forecast_pct_change = (future['yhat'].iloc[0] - historical['y'].iloc[-1]) / historical['y'].iloc[-1] * 100
                    
                    st.markdown("#### 📈 Forecast vs Historical Patterns")
                    cols = st.columns(3)
                    cols[0].metric("First Forecast Value", 
                                 f"{future['yhat'].iloc[0]:.1f}", 
                                 f"{forecast_pct_change:.1f}% from last historical")
                    cols[1].metric("Forecast Horizon", 
                                 f"{len(future)} periods")
                    cols[2].metric("Forecast Volatility", 
                                 f"{future['yhat'].std():.1f}", 
                                 "High" if future['yhat'].std() > historical['y'].std() else "Low")
                    
                    # Compare forecast to historical patterns
                    historical_seasonal = historical['y'].diff(12).mean()  # Approximate seasonal impact
                    forecast_seasonal = future['yhat'].diff(12).mean() if len(future) > 12 else 0
                    
                    st.markdown("#### 🔍 Forecast Insights")
                    st.write("""
                    **Consistency Check:**
                    - Seasonal pattern: {}
                    - Trend direction: {}
                    - Volatility: {}
                    
                    **Recommendations:**
                    - {}
                    - {}
                    - {}
                    """.format(
                        "consistent" if abs(historical_seasonal - forecast_seasonal) < historical['y'].std()/2 else "diverging",
                        "aligned" if (trend_slope > 0) == (future['yhat'].iloc[-1] > future['yhat'].iloc[0]) else "contradictory",
                        "higher than historical" if future['yhat'].std() > historical['y'].std() else "within normal range",
                        "Adjust safety stock for higher volatility" if future['yhat'].std() > historical['y'].std() else "Maintain current inventory policies",
                        "Review model parameters if patterns diverge significantly" if abs(historical_seasonal - forecast_seasonal) > historical['y'].std()/2 else "Model captures seasonal patterns well",
                        "Consider external factors for trend changes" if not ((trend_slope > 0) == (future['yhat'].iloc[-1] > future['yhat'].iloc[0])) else "Trend projection appears valid"
                    ))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    # Add section for saving insights
    if st.button("💾 Save Analysis Report"):
        # Generate a report (simplified version)
        report = {
            "item": item_name,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "key_insights": "Generated insights would go here",
            "recommendations": "Generated recommendations would go here"
        }
        
        if 'analysis_reports' not in st.session_state:
            st.session_state.analysis_reports = []
        st.session_state.analysis_reports.append(report)
        st.success("Analysis report saved!")
    
    st.markdown("""
    **Next Steps:**
    - Generate forecasts in the 🔮 Forecasting tab
    - View saved reports in the 🗃️ Database section
    - Adjust models based on insights
    """)
with tabs[6]:
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