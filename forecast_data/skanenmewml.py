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

# Skanem Brand Colors
PRIMARY_COLOR = "#0e4e4e"  # Genuine Green
SECONDARY_COLOR = "#e1ebae"  # Friendly Bright Green
BACKGROUND_COLOR = "#FFF2DF"  # Sustainability Beige
ACCENT_COLOR = "#80bce6"  # Trusted Azure

# Custom CSS for Skanem branding
def set_skanem_theme():
    st.markdown(
        f"""
        <style>
            /* Main colors */
            .stApp {{
                background-color: {BACKGROUND_COLOR};
            }}
            .css-1d391kg, .st-bh, .st-c2, .st-c3, .st-c4, .st-c5 {{
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            .st-b7 {{
                background-color: {SECONDARY_COLOR};
            }}
            .st-b8 {{
                background-color: {ACCENT_COLOR};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {PRIMARY_COLOR};
                font-family: 'Sarvatrik', sans-serif;
            }}
            
            /* Buttons */
            .stButton>button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 8px;
                border: none;
                font-weight: bold;
            }}
            .stButton>button:hover {{
                background-color: #0a3a3a;
                color: white;
            }}
            
            /* Sidebar */
            .css-1lcbmhc {{
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 10px;
            }}
            .stTabs [data-baseweb="tab"] {{
                background-color: {BACKGROUND_COLOR};
                border-radius: 8px 8px 0px 0px;
                padding: 10px 20px;
                border: 1px solid {PRIMARY_COLOR};
            }}
            .stTabs [aria-selected="true"] {{
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            
            /* Dataframes */
            .stDataFrame {{
                border: 1px solid {PRIMARY_COLOR};
                border-radius: 8px;
            }}
            
            /* File uploader */
            .stFileUploader>div>div>div>div {{
                color: {PRIMARY_COLOR};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Setup
set_skanem_theme()
st.set_page_config(page_title="Supply Chain Forecasting App", layout="wide", page_icon="📦")

# Add Skanem logo (replace with actual path to your logo)
st.sidebar.image("https://via.placeholder.com/150x50.png?text=Skanem+Logo", width=150)
st.title("📦 Supply Chain Forecasting")

CONFIG_PATH = "saved_configs.json"

# Functions to handle saved configs
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

# File upload
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', parse_dates=True)
        st.sidebar.success("File loaded successfully!")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', parse_dates=True)
        st.sidebar.warning("File loaded with Latin-1 encoding.")

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    df_columns = df.columns.tolist()
else:
    df = None
    st.sidebar.info("Upload a file to enable forecasting features.")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecast Dashboard",
    "🗓️ Forecast Table",
    "⚙️ Data Entry",
    "📊 Model Insights",
    "🧠 ML Modeling",
    "ℹ️ About"
])

# Dashboard Tab
with tab1:
    st.header("Forecast Overview")
    if df is not None:
        if "Date" in df.columns or any("month" in col.lower() or "year" in col.lower() for col in df.columns):
            date_col = [col for col in df.columns if "date" in col.lower() or "month" in col.lower() or "year" in col.lower()][0]
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=["Date"])
            df = df.sort_values("Date")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            target_col = st.selectbox("Select column to forecast", numeric_cols)
            forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)

            avg_daily = df[target_col].mean()
            current_balance = df[target_col].iloc[-1]

            dates = pd.date_range(datetime.now(), periods=forecast_horizon)
            forecast = [max(0, current_balance - i * avg_daily) for i in range(forecast_horizon)]

            forecast_df = pd.DataFrame({
                "Date": dates,
                "Forecasted Balance": forecast
            })

            fig = px.line(
                forecast_df, 
                x="Date", 
                y="Forecasted Balance", 
                title="Forecasted Inventory Level",
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig.update_layout(
                plot_bgcolor=BACKGROUND_COLOR,
                paper_bgcolor=BACKGROUND_COLOR,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df)
        else:
            st.warning("Ensure your dataset contains a date, month, or year column.")

# Forecast Table
with tab2:
    st.header("Forecast Table")
    if df is not None and "forecast_df" in locals():
        st.dataframe(forecast_df)
        st.download_button(
            "📥 Download Forecast", 
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="skanem_forecast_output.csv", 
            mime="text/csv"
        )
    else:
        st.info("Generate forecast from Dashboard tab to view table here.")

# Data Entry
with tab3:
    st.header("Manual Data Entry")
    with st.form("manual_form"):
        date = st.date_input("Date", datetime.today())
        quantity = st.number_input("Quantity", min_value=0.0)
        notes = st.text_area("Notes")
        if st.form_submit_button("Submit"):
            st.success(f"Entry for {date} recorded: {quantity} units.")

# Model Insights
with tab4:
    st.header("Model Evaluation")
    st.info("Upload data and choose target column in ML Modeling tab to view model accuracy metrics here.")

# ML Modeling
with tab5:
    st.header("Machine Learning Forecasting")
    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            target = st.selectbox("Target Variable", numeric_cols)
            features = st.multiselect("Feature Columns", [col for col in numeric_cols if col != target])

            if features:
                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("rf", RandomForestRegressor())
                ])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                with col2:
                    st.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.2f}")
                with col3:
                    st.metric("MAPE", f"{mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")

                if st.checkbox("Cluster Data with KMeans"):
                    n_clusters = st.slider("Clusters", 2, 10, 3)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    df["Cluster"] = kmeans.fit_predict(X)
                    st.dataframe(df)
        else:
            st.warning("No numeric columns found.")
    else:
        st.info("Upload a CSV file to begin modeling.")

# About Tab
with tab6:
    st.header("About This App")
    st.markdown(f"""
    <div style='background-color:{SECONDARY_COLOR}; padding: 20px; border-radius: 10px;'>
        <h3 style='color:{PRIMARY_COLOR};'>Supply Chain Forecasting App</h3>
        <p>Developed for: <strong style='color:{PRIMARY_COLOR};'>Skanem</strong></p>
        <p>Purpose: Upload any inventory/consumption data and perform:</p>
        <ul>
            <li>Forecast simulation</li>
            <li>ML prediction</li>
            <li>Clustering</li>
            <li>Visualization & exports</li>
        </ul>
        <p style='color:{PRIMARY_COLOR}; font-weight: bold;'>
            Creating a responsible future in packaging
        </p>
    </div>
    """, unsafe_allow_html=True)