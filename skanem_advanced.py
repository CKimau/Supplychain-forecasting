import streamlit as st
import pandas as pd

st.title("Upload Consumption File")

uploaded_consumption = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_consumption is not None:
    try:
        df = pd.read_csv(uploaded_consumption, encoding='utf-8-sig', parse_dates=True)
        st.success("File loaded successfully with UTF-8 encoding.")
    except UnicodeDecodeError:
        try:
            uploaded_consumption.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_consumption, encoding='latin1', parse_dates=True)
            st.warning("File loaded with Latin-1 encoding (UTF-8 failed).")
        except Exception as e:
            st.error(f"Failed to load the file: {e}")
            df = None

    if 'df' in locals() and df is not None:
        st.dataframe(df.head())

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# --- Page Config ---
st.set_page_config(page_title="Skanem Forecasting", layout="wide")
st.image("C:/Users/chris.mutuku/OneDrive - Skanem AS/Desktop/logo.jpg", width=44)
st.title("Advanced Inventory Forecasting")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Forecast Dashboard", 
    "📂 Data Upload", 
    "🤖 Model Training", 
    "📈 Forecast Results", 
    "📋 Model Insights", 
    "🧾 Historical Viewer"])

# --- Session State ---
if 'df_consumption' not in st.session_state:
    st.session_state.df_consumption = pd.DataFrame()
if 'df_forecast' not in st.session_state:
    st.session_state.df_forecast = pd.DataFrame()

# --- Tab 1: Forecast Dashboard ---
with tab1:
    st.sidebar.header("📝 Input Parameters")
    material_name = st.sidebar.text_input("Material Name", "White BOPP 30 Micron Solid")
    current_balance = st.sidebar.number_input("Current Available Balance", min_value=0.0, value=1000.0)
    avg_daily_consumption = st.sidebar.number_input("Average Daily Consumption", min_value=0.0, value=50.0)
    consumption_variability = st.sidebar.slider("Consumption Variability (%)", 0, 50, 10)
    safety_stock = st.sidebar.number_input("Safety Stock Level", min_value=0.0, value=200.0)
    lead_time = st.sidebar.number_input("Lead Time (days)", min_value=1, value=7)
    forecast_horizon = st.sidebar.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days"])
    reorder_point = safety_stock + (lead_time * avg_daily_consumption)
    st.sidebar.info(f"Auto-calculated Reorder Point: {reorder_point:.2f}")

    st.header("Forecast Overview")
    days_until_stockout = int(current_balance / avg_daily_consumption)
    stockout_date = (datetime.now() + timedelta(days=days_until_stockout)).strftime("%Y-%m-%d")
    days_until_reorder = int((current_balance - reorder_point) / avg_daily_consumption) if current_balance > reorder_point else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Balance", f"{current_balance:.2f}")
    col2.metric("Days Until Stockout", days_until_stockout, f"Expected by {stockout_date}")
    col3.metric("Reorder Point", f"{reorder_point:.2f}", f"{days_until_reorder} days until reorder" if days_until_reorder > 0 else "Below reorder point!")
    col4.metric("Avg Daily Consumption", f"{avg_daily_consumption:.2f}", f"±{consumption_variability}% variability")

    horizon_days = int(forecast_horizon.split(" ")[0])
    dates = pd.date_range(datetime.now(), periods=horizon_days)
    forecast_deterministic = [max(0, current_balance - (i * avg_daily_consumption)) for i in range(horizon_days)]
    np.random.seed(42)
    daily_variation = 1 + (np.random.rand(horizon_days) - 0.5) * (consumption_variability/100)
    forecast_probabilistic = [max(0, current_balance - np.sum(avg_daily_consumption * daily_variation[:i+1])) for i in range(horizon_days)]

    df_forecast = pd.DataFrame({
        "Date": dates,
        "Deterministic Forecast": forecast_deterministic,
        "Probabilistic Forecast": forecast_probabilistic,
        "Reorder Point": reorder_point,
        "Safety Stock": safety_stock
    })
    st.session_state.df_forecast = df_forecast.copy()

    df_melted = df_forecast.melt(id_vars="Date", value_vars=["Deterministic Forecast", "Probabilistic Forecast", "Reorder Point", "Safety Stock"], var_name="Metric", value_name="Value")
    fig = px.line(df_melted, x="Date", y="Value", color="Metric", title=f"Material Forecast: {material_name}", template="plotly_white")
    fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Stockout Level", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Data Upload ---
with tab2:
    st.header("Upload Inventory & Consumption Data")
    unit = st.selectbox("Select Unit of Measure", ["kg", "liters", "units","m²","pcs","Boxes","Rolls","Ton","each","dozen"])
    uploaded_inventory = st.file_uploader("Upload Current Inventory File (CSV)", type=["csv"])
    uploaded_consumption = st.file_uploader("Upload Historical Consumption Data (CSV)", type=["csv"])
    nan_strategy = st.selectbox("Fill Missing Values With", ["0", "Mean", "Median"])

    if uploaded_consumption:
        df = pd.read_csv(uploaded_consumption, parse_dates=True)
        if nan_strategy == "0":
            df = df.fillna(0)
        elif nan_strategy == "Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif nan_strategy == "Median":
            df = df.fillna(df.median(numeric_only=True))
        st.session_state.df_consumption = df.copy()
        st.success("Consumption data uploaded and cleaned.")
        st.dataframe(df.head())

# --- Tab 3: Model Training ---
with tab3:
    st.header("Model Training & Forecasting")
    df = st.session_state.df_consumption
    if not df.empty:
        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])
        if features and target:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.session_state.model_predictions = predictions
            st.session_state.y_test = y_test
            st.success("Model trained successfully.")

# --- Tab 4: Forecast Results ---
with tab4:
    st.header("Forecast Results")
    if not st.session_state.df_forecast.empty:
        df = st.session_state.df_forecast
        st.dataframe(df)
        st.download_button("Download Forecast as CSV", df.to_csv(index=False).encode('utf-8'), file_name="forecast_output.csv")

# --- Tab 5: Model Insights ---
with tab5:
    st.header("Model Evaluation Metrics")
    if 'model_predictions' in st.session_state:
        y_test = st.session_state.y_test
        y_pred = st.session_state.model_predictions
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAPE", f"{mape:.2f}%")

# --- Tab 6: Historical Viewer ---
with tab6:
    st.header("Historical Consumption Data")
    df = st.session_state.df_consumption
    if not df.empty:
        st.dataframe(df)
        st.download_button("Download Historical Data", df.to_csv(index=False).encode('utf-8'), file_name="historical_consumption.csv")