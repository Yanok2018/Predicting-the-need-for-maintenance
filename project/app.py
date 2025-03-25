from flask import Flask, render_template, request, redirect, url_for, session
from joblib import load
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv

app = Flask(__name__)

# Load model
model = load('model.pkl')

# User roles
USERS = {
    "admin": {"role": "admin", "password": "admin123"},
    "user": {"role": "user", "password": "user123"},
    "driver": {"role": "user", "password": "driver123"},
    "logistic": {"role": "logistic", "password": "logistic123"}
}

# Load service history
history_df = pd.read_csv('failures_data.csv')

# Secret key for session
app.secret_key = 'your_secret_key'

# Function to get the list of vehicles
def get_vehicle_list():
    return history_df['vehicle_number'].unique().tolist()

# Login
@app.route('/', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username  # Store user in session
            session['role'] = USERS[username]['role']  # Store role in session
            return redirect(url_for('dashboard', role=session['role']))
        return "Invalid username or password!"
    return render_template('login.html')

# Role-based dashboard
@app.route('/dashboard/<role>', methods=["GET", "POST"])
def dashboard(role):
    if 'username' not in session:
        return redirect(url_for('login'))  # If there's no session, redirect to login
    
    if request.method == "POST":
        mileage = float(request.form['mileage'])
        load = float(request.form['load'])
        features = np.array([[mileage, load]])
        prediction = model.predict(features)
        return render_template('result.html', role=role, prediction=prediction[0], mileage=mileage, load=load)
    
    return render_template('dashboard.html', role=role)

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove user from session
    session.pop('role', None)  # Remove role from session
    return redirect(url_for('login'))  # Redirect to login page

# Function for date filtering
def filter_by_date(df, date_filter):
    if date_filter == "last_month":
        last_month = datetime.now() - timedelta(days=30)
        return df[df['date'] >= last_month.strftime('%Y-%m-%d')]
    elif date_filter == "last_year":
        last_year = datetime.now() - timedelta(days=365)
        return df[df['date'] >= last_year.strftime('%Y-%m-%d')]
    return df  # If no filtering

# Service history
@app.route('/history', methods=["GET", "POST"])
def history():
    if 'username' not in session:
        return redirect(url_for('login'))  # Session check

    selected_vehicle = request.form.get('vehicle_number')
    date_filter = request.form.get('date_filter')
    vehicle_list = get_vehicle_list()  # Get list of vehicles
    
    # Filter history by selected vehicle
    filtered_history = history_df.copy()
    if selected_vehicle:
        filtered_history = filtered_history[filtered_history['vehicle_number'] == selected_vehicle]
    
    # Add date filtering
    filtered_history = filter_by_date(filtered_history, date_filter)
    
    # Convert to list of dictionaries
    filtered_history = filtered_history.to_dict(orient='records')
    
    return render_template('history.html', 
                           filtered_history=filtered_history, 
                           unique_cars=vehicle_list, 
                           selected_vehicle=selected_vehicle,
                           date_filter=date_filter)

# Days to failure prediction and forecast
@app.route('/predictions', methods=["GET", "POST"])
def predictions():
    if 'username' not in session:
        return redirect(url_for('login'))  # Session check

    selected_vehicle = request.form.get('vehicle_number')
    estimated_days = request.form.get('estimated_days')  # Estimated days to failure
    print(f"Selected Vehicle: {selected_vehicle}")  # Logging selected vehicle

    vehicle_list = get_vehicle_list()
    
    # Third feature (fixed value or value from file)
    third_feature = 10.0  # This can be replaced with the required value

    if selected_vehicle:
        print(f"Filtering data for vehicle {selected_vehicle}")
        filtered_data = history_df[history_df['vehicle_number'] == selected_vehicle].copy()
        
        # Logging data
        print(f"Filtered Data: {filtered_data}")
        
        mileage = filtered_data['mileage'].iloc[-1]  # Last mileage value
        load = filtered_data['daily_load'].iloc[-1]  # Last load value
        
        features = np.array([[mileage, load, third_feature]])  # Third feature
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
        
        categories = ['Axle', 'Brakes', 'Electrical', 'Engine', 'Suspension', 'Transmission']
        probabilities_dict = dict(zip(categories, probabilities))
        
        predicted_failure = categories[np.argmax(probabilities)]
        
        # If estimated days to failure is not entered, set default value (e.g., 90 days)
        if not estimated_days:
            estimated_days = 90
        else:
            estimated_days = int(estimated_days)

    else:
        predicted_failure = None
        probabilities_dict = {}
        estimated_days = None

    return render_template('predictions.html',
                           vehicle_list=vehicle_list,
                           selected_vehicle=selected_vehicle,
                           predicted_failure=predicted_failure,
                           probabilities_dict=probabilities_dict,
                           estimated_days=estimated_days)

# Add repair history
@app.route('/add_repair', methods=["GET", "POST"])
def add_repair():
    if 'username' not in session:
        return redirect(url_for('login'))  # Session check
    
    # Role check
    if session['role'] == 'logistic':
        return redirect(url_for('history'))  # Redirect to service history if role is logistic

    if request.method == "POST":
        vehicle_number = request.form['vehicle_number']
        repair_date = request.form['repair_date']
        repair_type = request.form['repair_type']
        repair_cost = request.form['repair_cost']

        # Add new entry to CSV
        with open('repair_history.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["vehicle_number", "repair_date", "repair_type", "repair_cost"])
            writer.writerow({
                "vehicle_number": vehicle_number,
                "repair_date": repair_date,
                "repair_type": repair_type,
                "repair_cost": repair_cost
            })

        return redirect(url_for('add_repair'))

    vehicle_list = get_vehicle_list()
    return render_template('add_repair.html', vehicle_list=vehicle_list)

if __name__ == "__main__":
    app.run(debug=True)
