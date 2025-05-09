import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# App title and configuration
st.set_page_config(page_title="Skanem Forecasting", layout="wide")
st.title("📊 SKANEM FORECASTING")

# Sidebar for user inputs
with st.sidebar:
    st.header("📝 Input Parameters")
    
    # Material information
    material_name = st.text_input("Material Name", "Steel Coil")
    current_balance = st.number_input("Current Available Balance", min_value=0.0, value=1000.0, step=1.0)
    
    # Consumption parameters
    avg_daily_consumption = st.number_input("Average Daily Consumption", min_value=0.0, value=50.0, step=1.0)
    consumption_variability = st.slider("Consumption Variability (%)", 0, 50, 10)
    
    # Forecasting parameters
    safety_stock = st.number_input("Safety Stock Level", min_value=0.0, value=200.0, step=1.0)
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7, step=1)
    forecast_horizon = st.selectbox("Forecast Horizon", ["30 days", "60 days", "90 days"], index=0)
    
    # Reorder point calculation
    reorder_point = safety_stock + (lead_time * avg_daily_consumption)
    st.info(f"Auto-calculated Reorder Point: {reorder_point:.2f}")

# Main content area
tab1, tab2, tab3 = st.tabs(["📈 Forecast Dashboard", "🗓️ Detailed Forecast", "⚙️ Data Entry"])

with tab1:
    st.header("Forecast Overview")
    
    # Calculate days until stockout
    days_until_stockout = int(current_balance / avg_daily_consumption)
    stockout_date = (datetime.now() + timedelta(days=days_until_stockout)).strftime("%Y-%m-%d")
    
    # Calculate days until reorder point
    days_until_reorder = int((current_balance - reorder_point) / avg_daily_consumption) if current_balance > reorder_point else 0
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Balance", f"{current_balance:.2f}")
    col2.metric("Days Until Stockout", days_until_stockout, f"Expected by {stockout_date}")
    col3.metric("Reorder Point", f"{reorder_point:.2f}", f"{days_until_reorder} days until reorder" if days_until_reorder > 0 else "Below reorder point!")
    col4.metric("Avg Daily Consumption", f"{avg_daily_consumption:.2f}", f"±{consumption_variability}% variability")
    
    # Generate forecast data
    horizon_days = int(forecast_horizon.split(" ")[0])
    dates = pd.date_range(datetime.now(), periods=horizon_days)
    
    # Create deterministic forecast
    forecast_deterministic = [max(0, current_balance - (i * avg_daily_consumption)) for i in range(horizon_days)]
    
    # Create probabilistic forecast with variability
    np.random.seed(42)
    daily_variation = 1 + (np.random.rand(horizon_days) - 0.5) * (consumption_variability/100)
    forecast_probabilistic = [max(0, current_balance - np.sum(avg_daily_consumption * daily_variation[:i+1])) for i in range(horizon_days)]
    
    # Create dataframe
    df_forecast = pd.DataFrame({
        "Date": dates,
        "Deterministic Forecast": forecast_deterministic,
        "Probabilistic Forecast": forecast_probabilistic,
        "Reorder Point": reorder_point,
        "Safety Stock": safety_stock
    })
    
    # Melt for plotting
    df_melted = df_forecast.melt(id_vars="Date", 
                                value_vars=["Deterministic Forecast", "Probabilistic Forecast", "Reorder Point", "Safety Stock"],
                                var_name="Metric", value_name="Value")
    
    # Plot forecast
    fig = px.line(df_melted, x="Date", y="Value", color="Metric", 
                 title=f"Material Forecast: {material_name}",
                 labels={"Value": "Quantity", "Date": "Date"},
                 template="plotly_white")
    
    # Add annotations
    fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Stockout Level", annotation_position="bottom right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly and monthly summaries
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

with tab2:
    st.header("Detailed Daily Forecast")
    st.dataframe(df_forecast.style.format({
        "Deterministic Forecast": "{:.2f}",
        "Probabilistic Forecast": "{:.2f}",
        "Reorder Point": "{:.2f}",
        "Safety Stock": "{:.2f}"
    }), use_container_width=True)
    
    # Export options
    st.download_button(
        label="Download Forecast as CSV",
        data=df_forecast.to_csv(index=False).encode('utf-8'),
        file_name=f"supply_forecast_{material_name.replace(' ', '_')}.csv",
        mime='text/csv'
    )

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
    # Placeholder for historical data - in a real app this would connect to a database
    st.info("This section would display and manage historical consumption data in a production environment")