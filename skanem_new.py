import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import os

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
            /* Main styles */
            .stApp {{
                background-color: {BACKGROUND_COLOR};
            }}
            .css-18e3th9 {{
                background-color: {BACKGROUND_COLOR};
            }}
            .css-1d391kg {{
                background-color: {PRIMARY_COLOR};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {PRIMARY_COLOR};
            }}
            
            /* Buttons */
            .stButton>button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 4px;
                border: none;
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
            .stTabs [role="tablist"] {{
                background-color: {BACKGROUND_COLOR};
            }}
            .stTabs [aria-selected="true"] {{
                color: {PRIMARY_COLOR} !important;
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
            
            /* Dataframes */
            .stDataFrame {{
                border: 1px solid {PRIMARY_COLOR};
            }}
            
            /* Metrics */
            .stMetric {{
                background-color: {SECONDARY_COLOR};
                border-radius: 4px;
                padding: 10px;
            }}
            .stMetricLabel {{
                color: {PRIMARY_COLOR};
            }}
            .stMetricValue {{
                color: {PRIMARY_COLOR};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Setup
st.set_page_config(
    page_title="Skanem SKU Tracker",
    layout="wide",
    page_icon="📊"
)

# Apply custom theme
set_skanem_theme()

# Header with logo
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://via.placeholder.com/44", width=44)  # Replace with actual Skanem logo
with col2:
    st.title("📊 Skanem SKU Consumption Tracker")

# Main SKU Tracker Page
st.header("SKU Consumption Analysis")

# File upload section
uploaded_file = st.file_uploader(
    "📤 Upload SKU Consumption Data (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Upload file with columns: SKU, Date, Quantity"
)

if uploaded_file is not None:
    # Read file based on extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    try:
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Convert date column if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Ensure required columns exist
            required_cols = ['SKU', 'Date', 'Quantity']
            if all(col in df.columns for col in required_cols):
                st.success("✅ Data loaded successfully")
                
                # Show raw data
                with st.expander("📁 View Raw Data"):
                    st.dataframe(df)
                
                # SKU selection
                sku_list = df['SKU'].unique().tolist()
                selected_sku = st.selectbox(
                    "🔍 Select SKU to analyze",
                    sku_list,
                    help="Select an SKU from the dropdown to view consumption patterns"
                )
                
                # Filter data for selected SKU
                sku_data = df[df['SKU'] == selected_sku].copy()
                
                # Resample to monthly data
                monthly_data = sku_data.set_index('Date').resample('M')['Quantity'].sum().reset_index()
                monthly_data['Month'] = monthly_data['Date'].dt.to_period('M')
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Consumption", f"{sku_data['Quantity'].sum():,}")
                with col2:
                    st.metric("Avg Monthly", f"{monthly_data['Quantity'].mean():,.1f}")
                with col3:
                    last_date = sku_data['Date'].max().strftime('%b %Y')
                    st.metric("Last Consumption", f"{sku_data.iloc[-1]['Quantity']:,}", 
                             delta=f"{last_date}")
                
                # Time Series Visualization
                st.subheader(f"📈 Consumption Trend for {selected_sku}")
                
                fig = px.line(
                    monthly_data,
                    x='Date',
                    y='Quantity',
                    markers=True,
                    color_discrete_sequence=[PRIMARY_COLOR],
                    labels={'Quantity': 'Consumption', 'Date': 'Month'}
                )
                
                fig.update_layout(
                    plot_bgcolor=BACKGROUND_COLOR,
                    paper_bgcolor=BACKGROUND_COLOR,
                    hovermode="x unified",
                    xaxis_title="Month",
                    yaxis_title="Quantity Consumed"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecasting Section
                st.subheader("🔮 Consumption Forecasting")
                
                col1, col2 = st.columns(2)
                with col1:
                    forecast_periods = st.number_input(
                        "Forecast periods (months)",
                        min_value=1,
                        max_value=24,
                        value=6,
                        help="Number of months to forecast"
                    )
                
                with col2:
                    model_choice = st.selectbox(
                        "Forecasting model",
                        ["Exponential Smoothing", "Prophet"],
                        help="Select forecasting algorithm"
                    )
                
                if st.button("Generate Forecast", type="primary"):
                    with st.spinner("Building forecast model..."):
                        try:
                            # Prepare data for forecasting
                            ts_data = monthly_data.set_index('Date')['Quantity']
                            
                            if model_choice == "Exponential Smoothing":
                                # Holt-Winters model
                                model = ExponentialSmoothing(
                                    ts_data,
                                    trend='add',
                                    seasonal='add',
                                    seasonal_periods=12
                                ).fit()
                                
                                # Generate forecast
                                forecast = model.forecast(forecast_periods)
                            
                            else:  # Prophet model
                                prophet_df = monthly_data[['Date', 'Quantity']].rename(
                                    columns={'Date': 'ds', 'Quantity': 'y'}
                                )
                                
                                model = Prophet(
                                    yearly_seasonality=True,
                                    weekly_seasonality=False,
                                    daily_seasonality=False
                                )
                                model.fit(prophet_df)
                                
                                future = model.make_future_dataframe(
                                    periods=forecast_periods,
                                    freq='M'
                                )
                                forecast_df = model.predict(future)
                                forecast = forecast_df['yhat'][-forecast_periods:].values
                            
                            # Create forecast dates
                            last_date = monthly_data['Date'].iloc[-1]
                            forecast_dates = pd.date_range(
                                start=last_date + pd.DateOffset(months=1),
                                periods=forecast_periods,
                                freq='M'
                            )
                            
                            # Create forecast dataframe
                            forecast_results = pd.DataFrame({
                                'Date': forecast_dates,
                                'Quantity': forecast,
                                'Type': 'Forecast'
                            })
                            
                            # Combine historical and forecast data
                            historical_results = pd.DataFrame({
                                'Date': monthly_data['Date'],
                                'Quantity': monthly_data['Quantity'],
                                'Type': 'Historical'
                            })
                            
                            combined_results = pd.concat([historical_results, forecast_results])
                            
                            # Plot forecast
                            forecast_fig = go.Figure()
                            
                            # Historical data
                            forecast_fig.add_trace(go.Scatter(
                                x=historical_results['Date'],
                                y=historical_results['Quantity'],
                                name='Historical',
                                line=dict(color=PRIMARY_COLOR, width=3),
                                mode='lines+markers'
                            ))
                            
                            # Forecast data
                            forecast_fig.add_trace(go.Scatter(
                                x=forecast_results['Date'],
                                y=forecast_results['Quantity'],
                                name='Forecast',
                                line=dict(color=ACCENT_COLOR, width=3, dash='dot'),
                                mode='lines+markers'
                            ))
                            
                            forecast_fig.update_layout(
                                title=f"{forecast_periods}-Month Consumption Forecast for {selected_sku}",
                                plot_bgcolor=BACKGROUND_COLOR,
                                paper_bgcolor=BACKGROUND_COLOR,
                                xaxis_title="Month",
                                yaxis_title="Quantity",
                                hovermode="x unified",
                                showlegend=True
                            )
                            
                            st.plotly_chart(forecast_fig, use_container_width=True)
                            
                            # Display forecast table
                            st.subheader("📋 Forecast Details")
                            forecast_display = forecast_results.copy()
                            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m')
                            forecast_display = forecast_display.rename(
                                columns={'Date': 'Forecast Month', 'Quantity': 'Forecasted Quantity'}
                            )
                            
                            st.dataframe(forecast_display)
                            
                            # Download button
                            csv = forecast_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "💾 Download Forecast as CSV",
                                data=csv,
                                file_name=f"{selected_sku.replace(' ', '_')}_forecast.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating forecast: {str(e)}")
            else:
                st.warning("⚠️ Please ensure your data contains these columns: SKU, Date, Quantity")
        else:
            st.warning("⚠️ No 'Date' column found in the uploaded data")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
else:
    st.info("ℹ️ Please upload a CSV or Excel file to begin SKU analysis")

# Sample data download
with st.expander("💡 Don't have data? Download sample template"):
    sample_data = pd.DataFrame({
        'SKU': ['SKU001', 'SKU001', 'SKU002', 'SKU002'],
        'Date': ['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01'],
        'Quantity': [100, 150, 200, 180]
    })
    
    csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Sample CSV",
        data=csv,
        file_name="sample_sku_data.csv",
        mime="text/csv"
    )