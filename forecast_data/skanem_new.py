import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# App configuration
st.set_page_config(page_title="Advanced Supply Chain Forecasting", layout="wide")
st.image("C:/Users/chris.mutuku/OneDrive - Skanem AS/Desktop/logo.jpg", width=44)
st.title("Advanced Supply Chain Forecasting")

# Paths and global variables
CONFIG_PATH = "saved_configs.json"

# Helper to load/save configurations
def load_configs():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_config(material, config):
    all_configs = load_configs()
    all_configs[material] = config
    with open(CONFIG_PATH, "w") as f:
        json.dump(all_configs, f, indent=4)

configs = load_configs()

# Sidebar input parameters
with st.sidebar:
    st.header("📝 Input Parameters")

    selected_material = st.selectbox("Select Saved Configuration or Enter New", ["New"] + list(configs.keys()))

    if selected_material != "New":
        material_name = selected_material
        saved = configs[material_name]
        current_balance = saved["Current Balance"]
        avg_daily_consumption = saved["Avg Daily Consumption"]
        consumption_variability = saved["Consumption Variability"]
        safety_stock = saved["Safety Stock"]
        lead_time = saved["Lead Time"]
        forecast_horizon = saved["Forecast Horizon"]
    else:
        material_name = st.text_input("Material Name", "Steel Coil")
        current_balance = st.number_input("Current Available Balance", min_value=0.0, value=1000.0, step=1.0)
        avg_daily_consumption = st.number_input("Average Daily Consumption", min_value=0.0, value=50.0, step=1.0)
        consumption_variability = st.slider("Consumption Variability (%)", 0, 50, 10)
        safety_stock = st.number_input("Safety Stock Level", min_value=0.0, value=200.0, step=1.0)
        lead_time = st.number_input("Lead Time (days)", min_value=1, value=7, step=1)
        forecast_horizon = st.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days"], index=0)

    reorder_point = safety_stock + (lead_time * avg_daily_consumption)
    st.info(f"Auto-calculated Reorder Point: {reorder_point:.2f}")

    if st.button("💾 Save Configuration"):
        config = {
            "Current Balance": current_balance,
            "Avg Daily Consumption": avg_daily_consumption,
            "Consumption Variability": consumption_variability,
            "Safety Stock": safety_stock,
            "Lead Time": lead_time,
            "Forecast Horizon": forecast_horizon
        }
        save_config(material_name, config)
        st.success("Configuration saved!")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast Dashboard", 
    "🗓️ Detailed Forecast", 
    "⚙️ Data Entry", 
    "📊 Model Insights", 
    "🧠 ML Modeling"])

# Tab 1: Simulation and Forecast Dashboard
with tab1:
    st.header("Forecast Overview")

    days_until_stockout = int(current_balance / avg_daily_consumption) if avg_daily_consumption > 0 else 0
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

    df_melted = df_forecast.melt(id_vars="Date", 
                                value_vars=["Deterministic Forecast", "Probabilistic Forecast", "Reorder Point", "Safety Stock"],
                                var_name="Metric", value_name="Value")

    fig = px.line(df_melted, x="Date", y="Value", color="Metric", 
                 title=f"Material Forecast: {material_name}",
                 labels={"Value": "Quantity", "Date": "Date"},
                 template="plotly_white")

    fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Stockout Level", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

    df_forecast["Week"] = df_forecast["Date"].dt.isocalendar().week
    df_forecast["Month"] = df_forecast["Date"].dt.month_name()

    weekly_summary = df_forecast.groupby("Week").agg({
        "Deterministic Forecast": "min",
        "Probabilistic Forecast": "min"
    }).reset_index()

    monthly_summary = df_forecast.groupby("Month").agg({
        "Deterministic Forecast": "min",
        "Probabilistic Forecast": "min"
    }).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Weekly Forecast Summary")
        st.dataframe(weekly_summary.style.format({
            "Deterministic Forecast": "{:.2f}",
            "Probabilistic Forecast": "{:.2f}"
        }), use_container_width=True)

    with col2:
        st.subheader("Monthly Forecast Summary")
        st.dataframe(monthly_summary.style.format({
            "Deterministic Forecast": "{:.2f}",
            "Probabilistic Forecast": "{:.2f}"
        }), use_container_width=True)

    # Allow user to save simulation forecast as config
    if st.button("💾 Save This Forecast Simulation"):
        forecast_config = {
            "Current Balance": current_balance,
            "Avg Daily Consumption": avg_daily_consumption,
            "Consumption Variability": consumption_variability,
            "Safety Stock": safety_stock,
            "Lead Time": lead_time,
            "Forecast Horizon": forecast_horizon,
            "Deterministic Forecast": forecast_deterministic,
            "Probabilistic Forecast": forecast_probabilistic
        }
        save_config(f"Sim_{material_name}_{datetime.now().strftime('%Y%m%d_%H%M')}", forecast_config)
        st.success("Simulation forecast saved!")

# Tab 2: Daily Forecast Table
with tab2:
    st.header("Detailed Daily Forecast")
    st.dataframe(df_forecast.style.format({
        "Deterministic Forecast": "{:.2f}",
        "Probabilistic Forecast": "{:.2f}",
        "Reorder Point": "{:.2f}",
        "Safety Stock": "{:.2f}"
    }), use_container_width=True)

    st.download_button(
        label="Download Forecast as CSV",
        data=df_forecast.to_csv(index=False).encode('utf-8'),
        file_name=f"supply_forecast_{material_name.replace(' ', '_')}.csv",
        mime='text/csv'
    )

# Tab 3: Manual Data Entry
with tab3:
    st.header("Data Entry Form")
    with st.form("consumption_data_entry"):
        st.subheader("Record Daily Consumption")
        entry_date = st.date_input("Date", datetime.now())
        quantity_used = st.number_input("Quantity Consumed", min_value=0.0, step=1.0)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Save Entry")
        if submitted:
            st.success(f"Entry saved for {entry_date}: {quantity_used} units consumed")

    st.subheader("Historical Consumption Data")
    st.info("This section would display and manage historical consumption data in a production environment")

# Tab 4: Placeholder for Model Insights
with tab4:
    st.header("Model Insights and Performance")
    st.info("Upload historical inventory consumption to view model performance insights including R², RMSE, MAPE.")

# Tab 5: Placeholder for ML modeling
with tab5:
    st.header("ML Modeling (Supervised & Unsupervised")
    uploaded_file = st.file_uploader("Upload CSV file with historical data", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_uploaded.head())

        numeric_cols = df_uploaded.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            target_col = st.selectbox("Select column to forecast (target)", numeric_cols)
            feature_cols = st.multiselect("Select feature columns", [col for col in numeric_cols if col != target_col])

            if feature_cols:
                X = df_uploaded[feature_cols]
                y = df_uploaded[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=42))
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                st.subheader("Model Performance")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
                st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
                st.write(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred)*100:.2f}%")

                if st.checkbox("Run KMeans Clustering"):
                    num_clusters = st.slider("Select number of clusters", 2, 10, 3)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(X)
                    df_uploaded["Cluster"] = cluster_labels
                    st.write(df_uploaded.head())
                    st.success("Clustering applied and added as new column 'Cluster'.")
        else:
            st.warning("No numeric columns found in the uploaded data.")
    else:
        st.info("Upload your CSV data to begin machine learning modeling.")
