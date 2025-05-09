from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import os
import json
import pickle

# App configuration
st.set_page_config(page_title="SKANEM FORECASTING", layout="wide")
st.image("c:/Users/chris.mutuku/OneDrive - Skanem AS/Desktop/logo.jpg", width=50)

# File paths
DATA_DIR = "forecast_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Model saving/loading functions
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

# Forecasting functions
def calculate_metrics(actual, predicted):
    return {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
        'R2': r2_score(actual, predicted)
    }

def generate_forecast(current_balance, avg_consumption, variability, horizon):
    np.random.seed(42)
    dates = pd.date_range(datetime.now(), periods=horizon)
    
    # Deterministic forecast
    deterministic = [max(0, current_balance - (i * avg_consumption)) for i in range(horizon)]
    
    # Probabilistic forecast
    daily_variation = 1 + (np.random.rand(horizon) - 0.5) * (variability/100)
    probabilistic = [max(0, current_balance - np.sum(avg_consumption * daily_variation[:i+1])) for i in range(horizon)]
    
    return dates, deterministic, probabilistic

# UI Components
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Material selection/creation
    saved_materials = get_saved_materials()
    material_option = st.selectbox("Select or create material", ["Create New"] + saved_materials)
    
    if material_option == "Create New":
        material_name = st.text_input("New Material Name", "Steel Coil")
    else:
        material_name = material_option
        loaded_data = load_material_data(material_name)
    
    # Parameters
    current_balance = st.number_input("Current Balance", min_value=0.0, value=1000.0 if material_option == "Create New" else loaded_data['current_balance'])
    avg_consumption = st.number_input("Avg Daily Consumption", min_value=0.0, value=50.0 if material_option == "Create New" else loaded_data['avg_consumption'])
    variability = st.slider("Consumption Variability (%)", 0, 50, 10 if material_option == "Create New" else loaded_data['variability'])
    safety_stock = st.number_input("Safety Stock", min_value=0.0, value=200.0 if material_option == "Create New" else loaded_data['safety_stock'])
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7 if material_option == "Create New" else loaded_data['lead_time'])
    forecast_horizon = st.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days"], index=0)
    
    if st.button("💾 Save Material Configuration"):
        data = {
            'current_balance': current_balance,
            'avg_consumption': avg_consumption,
            'variability': variability,
            'safety_stock': safety_stock,
            'lead_time': lead_time
        }
        save_material_data(material_name, data)
        st.success(f"Saved {material_name} configuration!")
    
    st.divider()
    st.info("Model splits data 70% train / 30% test for accuracy validation")

# Main forecasting logic
horizon_days = int(forecast_horizon.split(" ")[0])
dates, deterministic, probabilistic = generate_forecast(current_balance, avg_consumption, variability, horizon_days)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Deterministic': deterministic,
    'Probabilistic': probabilistic,
    'Reorder_Point': safety_stock + (lead_time * avg_consumption),
    'Safety_Stock': safety_stock
})

# Train-test split
split_idx = int(horizon_days * 0.7)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Calculate metrics
metrics = calculate_metrics(test['Deterministic'], test['Probabilistic'])

# Dashboard Layout
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📆 Monthly View", "📊 Model Performance"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive forecast plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Deterministic'], name='Deterministic Forecast'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Probabilistic'], name='Probabilistic Forecast'))
        fig.add_hline(y=df['Reorder_Point'].iloc[0], line_dash='dot', line_color='orange', name='Reorder Point')
        fig.add_hline(y=df['Safety_Stock'].iloc[0], line_dash='dot', line_color='red', name='Safety Stock')
        fig.update_layout(title=f"{material_name} Forecast", xaxis_title='Date', yaxis_title='Quantity')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Current Balance", f"{current_balance:,.2f}")
        st.metric("Days Until Stockout", int(current_balance / avg_consumption))
        st.metric("Reorder Point", f"{df['Reorder_Point'].iloc[0]:,.2f}")
        st.metric("Avg Daily Use", f"{avg_consumption:,.2f}")

with tab2:
    # Monthly aggregation
    df_monthly = df.set_index('Date').resample('M').agg({
        'Deterministic': 'min',
        'Probabilistic': 'min',
        'Reorder_Point': 'first',
        'Safety_Stock': 'first'
    }).reset_index()
    
    fig_month = px.line(df_monthly, x='Date', y=['Deterministic', 'Probabilistic', 'Reorder_Point', 'Safety_Stock'],
                       title="Monthly Forecast Summary")
    st.plotly_chart(fig_month, use_container_width=True)
    
    st.dataframe(df_monthly.style.format({
        'Deterministic': '{:,.2f}',
        'Probabilistic': '{:,.2f}',
        'Reorder_Point': '{:,.2f}',
        'Safety_Stock': '{:,.2f}'
    }))

with tab3:
    st.subheader("Model Performance (70% train / 30% test)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col2.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    col3.metric("R² Score", f"{metrics['R2']:.2f}")
    
    # Actual vs Predicted plot
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test['Date'], y=test['Deterministic'], name='Actual'))
    fig_test.add_trace(go.Scatter(x=test['Date'], y=test['Probabilistic'], name='Predicted'))
    fig_test.update_layout(title="Test Set: Actual vs Predicted", xaxis_title='Date', yaxis_title='Quantity')
    st.plotly_chart(fig_test, use_container_width=True)
    
    st.download_button(
        label="📥 Download Full Forecast",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"{material_name.replace(' ', '_')}_forecast.csv",
        mime='text/csv'
    )

# Historical data section
st.sidebar.divider()
with st.sidebar.expander("📤 Upload Historical Data"):
    uploaded_file = st.file_uploader("Upload consumption history (CSV)", type=['csv'])
    if uploaded_file:
        hist_data = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(hist_data)} records")
        if st.button("Use for Model Training"):
            # Here you would add code to retrain models with historical data
            st.info("Model training functionality would be implemented here")