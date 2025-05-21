from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import io
import sqlite3
import json
from werkzeug.utils import secure_filename
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'supply_chain.db'
app.config['SECRET_KEY'] = 'your-secret-key'

# Initialize database
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    c = conn.cursor()
    
    # Create tables for different functionalities
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 material_name TEXT,
                 forecast_type TEXT,
                 horizon TEXT,
                 parameters TEXT,
                 results TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS inventory (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 sku TEXT,
                 quantity REAL,
                 unit TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS demand_history (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 sku TEXT,
                 date DATE,
                 demand REAL,
                 unit TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS unit_conversions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 input_value REAL,
                 input_unit TEXT,
                 output_value REAL,
                 output_unit TEXT,
                 conversion_type TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def convert_units(value, from_unit, to_unit, **kwargs):
    """Handle unit conversions for supply chain metrics"""
    converters = {
        # Weight conversions
        ('kg', 'lbs'): lambda x: x * 2.20462,
        ('lbs', 'kg'): lambda x: x / 2.20462,
        ('kg', 'tons'): lambda x: x / 1000,
        ('tons', 'kg'): lambda x: x * 1000,
        
        # Volume conversions
        ('liters', 'gallons'): lambda x: x * 0.264172,
        ('gallons', 'liters'): lambda x: x / 0.264172,
        ('cubic_meters', 'cubic_feet'): lambda x: x * 35.3147,
        ('cubic_feet', 'cubic_meters'): lambda x: x / 35.3147,
        
        # Length conversions
        ('meters', 'feet'): lambda x: x * 3.28084,
        ('feet', 'meters'): lambda x: x / 3.28084,
        ('meters', 'yards'): lambda x: x * 1.09361,
        ('yards', 'meters'): lambda x: x / 1.09361,
        
        # Area conversions
        ('sqm', 'sqft'): lambda x: x * 10.7639,
        ('sqft', 'sqm'): lambda x: x / 10.7639,
        
        # Specialized conversions for packaging materials
        ('kg', 'sqm'): lambda x: x / (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('sqm', 'kg'): lambda x: x * (kwargs.get('thickness_microns', 35) * 1e-6 * kwargs.get('density', 0.92)),
        ('kg', 'rolls'): lambda x: x / kwargs.get('roll_weight', 1),
        ('rolls', 'kg'): lambda x: x * kwargs.get('roll_weight', 1)
    }
    
    if (from_unit, to_unit) in converters:
        return converters[(from_unit, to_unit)](value)
    else:
        return value  # Return original if no conversion found

# Routes
@app.route('/')
def dashboard():
    """Main dashboard with overview of supply chain metrics"""
    return render_template('dashboard.html')

@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    """Forecasting interface with multiple methods"""
    if request.method == 'POST':
        # Handle forecast request
        data = request.form
        forecast_type = data.get('forecast_type')
        horizon = int(data.get('horizon', 30))
        sku = data.get('sku')
        
        # Get historical data from database
        conn = sqlite3.connect(app.config['DATABASE'])
        df = pd.read_sql(f"SELECT date, demand FROM demand_history WHERE sku = '{sku}'", conn)
        conn.close()
        
        if df.empty:
            return jsonify({'error': 'No historical data found for this SKU'})
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if forecast_type == 'prophet':
            # Prophet forecasting
            model = Prophet()
            prophet_df = df.rename(columns={'date': 'ds', 'demand': 'y'})
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['demand'], name='Historical'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.update_layout(title=f'Demand Forecast for {sku}')
            plot_html = fig.to_html(full_html=False)
            
            # Save forecast to database
            conn = sqlite3.connect(app.config['DATABASE'])
            c = conn.cursor()
            c.execute('''INSERT INTO forecasts 
                         (material_name, forecast_type, horizon, parameters, results) 
                         VALUES (?, ?, ?, ?, ?)''',
                         (sku, 'Prophet', str(horizon), json.dumps({}), forecast.to_json()))
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'plot_html': plot_html,
                'forecast_data': forecast[['ds', 'yhat']].tail(horizon).to_dict('records')
            })
        
        elif forecast_type == 'random_forest':
            # Random Forest forecasting
            df['day_of_year'] = df['date'].dt.dayofyear
            X = df[['day_of_year']]
            y = df['demand']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            
            # Generate future dates
            last_date = df['date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]
            future_df = pd.DataFrame({
                'date': future_dates,
                'day_of_year': [d.dayofyear for d in future_dates]
            })
            
            predictions = model.predict(future_df[['day_of_year']])
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['demand'], name='Historical'))
            fig.add_trace(go.Scatter(x=future_df['date'], y=predictions, name='Forecast'))
            fig.update_layout(title=f'Demand Forecast for {sku}')
            plot_html = fig.to_html(full_html=False)
            
            # Save forecast to database
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions
            })
            
            conn = sqlite3.connect(app.config['DATABASE'])
            c = conn.cursor()
            c.execute('''INSERT INTO forecasts 
                         (material_name, forecast_type, horizon, parameters, results) 
                         VALUES (?, ?, ?, ?, ?)''',
                         (sku, 'Random Forest', str(horizon), json.dumps({}), forecast_df.to_json()))
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'plot_html': plot_html,
                'forecast_data': forecast_df.to_dict('records')
            })
    
    # GET request - show forecasting interface
    conn = sqlite3.connect(app.config['DATABASE'])
    skus = pd.read_sql("SELECT DISTINCT sku FROM demand_history", conn)['sku'].tolist()
    conn.close()
    
    return render_template('forecasting.html', skus=skus)

@app.route('/inventory', methods=['GET', 'POST'])
def inventory_management():
    """Inventory management interface"""
    if request.method == 'POST':
        # Handle inventory update
        sku = request.form.get('sku')
        quantity = float(request.form.get('quantity'))
        unit = request.form.get('unit')
        
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        
        # Check if inventory exists for this SKU
        c.execute("SELECT id FROM inventory WHERE sku = ?", (sku,))
        existing = c.fetchone()
        
        if existing:
            c.execute("UPDATE inventory SET quantity = ?, unit = ? WHERE sku = ?", 
                     (quantity, unit, sku))
        else:
            c.execute("INSERT INTO inventory (sku, quantity, unit) VALUES (?, ?, ?)",
                     (sku, quantity, unit))
        
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success'})
    
    # GET request - show inventory
    conn = sqlite3.connect(app.config['DATABASE'])
    inventory = pd.read_sql("SELECT * FROM inventory", conn).to_dict('records')
    conn.close()
    
    return render_template('inventory.html', inventory=inventory)

@app.route('/conversion', methods=['GET', 'POST'])
def unit_conversion():
    """Unit conversion tool"""
    if request.method == 'POST':
        input_value = float(request.form.get('input_value'))
        input_unit = request.form.get('input_unit')
        output_unit = request.form.get('output_unit')
        conversion_type = request.form.get('conversion_type', 'standard')
        
        # Handle specialized conversions
        kwargs = {}
        if conversion_type == 'packaging':
            kwargs['thickness_microns'] = float(request.form.get('thickness', 35))
            kwargs['density'] = float(request.form.get('density', 0.92))
        
        result = convert_units(input_value, input_unit, output_unit, **kwargs)
        
        # Save conversion to database
        conn = sqlite3.connect(app.config['DATABASE'])
        c = conn.cursor()
        c.execute('''INSERT INTO unit_conversions 
                    (input_value, input_unit, output_value, output_unit, conversion_type)
                    VALUES (?, ?, ?, ?, ?)''',
                    (input_value, input_unit, result, output_unit, conversion_type))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'result': result,
            'output_unit': output_unit
        })
    
    # GET request - show conversion interface
    return render_template('conversion.html')

@app.route('/upload', methods=['GET', 'POST'])
def data_upload():
    """Data upload interface"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file based on type
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Determine data type (inventory or demand history)
                data_type = request.form.get('data_type')
                
                conn = sqlite3.connect(app.config['DATABASE'])
                if data_type == 'inventory':
                    # Process inventory data
                    required_cols = {'sku', 'quantity', 'unit'}
                    if not required_cols.issubset(df.columns):
                        return jsonify({'error': f'Missing required columns: {required_cols}'})
                    
                    # Clear existing data if requested
                    if request.form.get('overwrite') == 'true':
                        conn.execute("DELETE FROM inventory")
                    
                    df.to_sql('inventory', conn, if_exists='append', index=False)
                
                elif data_type == 'demand':
                    # Process demand history
                    required_cols = {'sku', 'date', 'demand', 'unit'}
                    if not required_cols.issubset(df.columns):
                        return jsonify({'error': f'Missing required columns: {required_cols}'})
                    
                    if request.form.get('overwrite') == 'true':
                        conn.execute("DELETE FROM demand_history")
                    
                    df.to_sql('demand_history', conn, if_exists='append', index=False)
                
                conn.commit()
                conn.close()
                
                return jsonify({
                    'status': 'success',
                    'rows_processed': len(df)
                })
            
            except Exception as e:
                return jsonify({'error': str(e)})
    
    # GET request - show upload interface
    return render_template('upload.html')

@app.route('/reports')
def reports():
    """Generate various supply chain reports"""
    conn = sqlite3.connect(app.config['DATABASE'])
    
    # Inventory report
    inventory = pd.read_sql("SELECT * FROM inventory", conn)
    
    # Demand forecast accuracy report
    forecasts = pd.read_sql('''SELECT f.material_name, f.forecast_type, f.horizon, 
                              d.date, d.demand, f.results
                              FROM forecasts f
                              JOIN demand_history d ON f.material_name = d.sku''', conn)
    
    # Process forecast accuracy if data exists
    forecast_accuracy = None
    if not forecasts.empty:
        # This would need more sophisticated processing in a real app
        forecast_accuracy = forecasts.groupby(['material_name', 'forecast_type']).size().reset_index(name='count')
    
    conn.close()
    
    return render_template('reports.html', 
                         inventory=inventory.to_dict('records'),
                         forecast_accuracy=forecast_accuracy.to_dict('records') if forecast_accuracy is not None else None)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)