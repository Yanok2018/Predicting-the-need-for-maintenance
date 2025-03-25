import numpy as np
import pandas as pd
import random
import os
import joblib
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from dotenv import load_dotenv
import argparse


start_time = time.time()

# Load environment variables
load_dotenv()

DEFAULT_VEHICLE = os.getenv("DEFAULT_VEHICLE", "BK4570CK")
DEFAULT_DAYS = int(os.getenv("DEFAULT_DAYS", "90"))
DATA_PATH = os.getenv("DATA_PATH", "./data")

def generate_synthetic_data(num_samples=1000):
    mileage = np.random.normal(20000, 5000, num_samples)
    daily_load = np.random.randint(1, 100, num_samples)
    failure_categories = ['Axle', 'Brakes', 'Electrical', 'Engine', 'Suspension', 'Transmission']
    failure_category = np.random.choice(failure_categories, num_samples, p=[0.15, 0.20, 0.25, 0.25, 0.10, 0.05])
    
    return pd.DataFrame({
        'mileage': mileage,
        'daily_load': daily_load,
        'failure_category': failure_category
    })

# Generate synthetic data
synthetic_data = generate_synthetic_data(2000)

class VehicleMaintenanceSystem:
    def __init__(self):
        self.failure_types = {
            'Engine': {
                'Engine Failure': {'cost': (2000, 5000), 'duration': (14, 28), 'symptoms': 'Engine knocking', 'probability': 0.15},
                'Turbo Failure': {'cost': (1500, 3000), 'duration': (7, 14), 'symptoms': 'Whistling noise', 'probability': 0.18},
                'Oil Leak': {'cost': (500, 1500), 'duration': (3, 7), 'symptoms': 'Oil spots', 'probability': 0.12},
                'Radiator Failure': {'cost': (800, 2000), 'duration': (5, 10), 'symptoms': 'Overheating', 'probability': 0.1}
            },
            'Axle': {
                'Axle Failure': {'cost': (1500, 3000), 'duration': (14, 21), 'symptoms': 'Vehicle stops', 'probability': 0.12},
                'Differential Failure': {'cost': (1000, 2000), 'duration': (7, 14), 'symptoms': 'Grinding noise', 'probability': 0.16},
                'Spring Breakage': {'cost': (400, 800), 'duration': (2, 5), 'symptoms': 'Unusual noises', 'probability': 0.08}
            },
            'Transmission': {
                'Transmission Failure': {'cost': (2500, 6000), 'duration': (15, 30), 'symptoms': 'Slipping gears', 'probability': 0.1},
                'Clutch Failure': {'cost': (800, 2500), 'duration': (2, 5), 'symptoms': 'Difficulty shifting', 'probability': 0.08},
                'Gearbox Failure': {'cost': (2000, 4000), 'duration': (10, 20), 'symptoms': 'Grinding noise', 'probability': 0.09}
            },
            'Brakes': {
                'Brake Pad Wear': {'cost': (150, 300), 'duration': (1, 2), 'symptoms': 'Squeaking noise', 'probability': 0.2},
                'Brake Fluid Leak': {'cost': (200, 500), 'duration': (2, 5), 'symptoms': 'Soft brake pedal', 'probability': 0.15},
                'Brake Line Failure': {'cost': (300, 700), 'duration': (1, 3), 'symptoms': 'Brake warning light', 'probability': 0.07}
            },
            'Electrical': {
                'Battery Failure': {'cost': (100, 300), 'duration': (1, 2), 'symptoms': 'Engine won’t start', 'probability': 0.2},
                'Alternator Failure': {'cost': (200, 500), 'duration': (1, 3), 'symptoms': 'Battery warning light', 'probability': 0.15},
                'Starter Motor Failure': {'cost': (150, 400), 'duration': (1, 2), 'symptoms': 'Clicking noise', 'probability': 0.1}
            },
            'Suspension': {
                'Shock Absorber Failure': {'cost': (400, 1000), 'duration': (5, 10), 'symptoms': 'Bumpy ride', 'probability': 0.1},
                'Ball Joint Failure': {'cost': (200, 600), 'duration': (2, 4), 'symptoms': 'Clunking noise', 'probability': 0.09}
            }
        }

    def generate_failures(self, vehicle_data, start_date, end_date):
        print(f"Generating failures for vehicle {vehicle_data['vehicle_number']}")
        failures = []
        current_date = start_date
        current_mileage = vehicle_data['initial_mileage']
        daily_mileage = random.randint(700, 800)

        while current_date <= end_date:
            daily_load = random.randint(500, 1500)
            print(f"Generating failure for date: {current_date.strftime('%Y-%m-%d')}, Mileage: {current_mileage}")

            for failure_category, failure_types in self.failure_types.items():
                for failure_name, failure_data in failure_types.items():
                    probability_adjustment = daily_load / 1000.0 * 0.02
                    adjusted_probability = min(failure_data['probability'] + probability_adjustment, 0.5)

                    if random.random() < adjusted_probability:
                        cost = round(random.uniform(*failure_data['cost']), 2)
                        duration = random.randint(*failure_data['duration'])
                        failure = {
                            'date': current_date.strftime('%Y-%m-%d'),
                            'vehicle_number': vehicle_data['vehicle_number'],
                            'mileage': current_mileage,
                            'failure_category': failure_category,
                            'failure_name': failure_name,
                            'symptoms': failure_data['symptoms'],
                            'cost': cost,
                            'repair_duration': duration,
                            'daily_load': daily_load
                        }
                        failures.append(failure)

            current_date += timedelta(days=1)
            current_mileage += daily_mileage

        return pd.DataFrame(failures)

def create_vehicle_maintenance_dataset(start_date, end_date):
    maintenance_system = VehicleMaintenanceSystem()
    vehicles = {
        'BK4570CK': {'model': 'Mercedes 814', 'year': 1993, 'initial_mileage': 850000, 'vehicle_number': 'BK4570CK'},
        'BK9057HC': {'model': 'Mercedes Atego', 'year': 1998, 'initial_mileage': 550000, 'vehicle_number': 'BK9057HC'},
        'BK2218ET': {'model': 'Mercedes Atego', 'year': 1999, 'initial_mileage': 400000, 'vehicle_number': 'BK2218ET'}
    }

    all_failures = pd.DataFrame()

    for vehicle_data in vehicles.values():
        vehicle_failures = maintenance_system.generate_failures(vehicle_data, start_date, end_date)
        all_failures = pd.concat([all_failures, vehicle_failures], ignore_index=True)

    return all_failures.sort_values('date').reset_index(drop=True)

def prepare_sequential_data(df):
    records = []
    for vehicle, group in df.groupby('vehicle_number'):
        group = group.sort_values('date').reset_index(drop=True)
        for i in range(len(group) - 1):
            current = group.iloc[i]
            nxt = group.iloc[i + 1]
            record = {
                'vehicle_number': vehicle,
                'mileage': current['mileage'],
                'date': current['date'],
                'failure_category': current['failure_category'],
                'daily_load': current['daily_load'],
                'interval': (pd.to_datetime(nxt['date']) - pd.to_datetime(current['date'])).days,
                'next_failure_category': nxt['failure_category'],
            }
            records.append(record)

    seq_df = pd.DataFrame(records)
    encoder_next_failure = LabelEncoder()
    seq_df['next_failure_category_encoded'] = encoder_next_failure.fit_transform(seq_df['next_failure_category'])
    
    return seq_df, encoder_next_failure

def train_random_forest_model(failures_df):
    print("Preparing sequential data for training the model...")
    sequential_data, next_failure_encoder = prepare_sequential_data(failures_df)

    encoder_failure_category = LabelEncoder()
    sequential_data['failure_category_encoded'] = encoder_failure_category.fit_transform(sequential_data['failure_category'])

    features = ['mileage', 'daily_load', 'failure_category_encoded']
    target_category = 'next_failure_category_encoded'

    X = sequential_data[features]
    y_failure_category = sequential_data[target_category]

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train_category, y_test_category = train_test_split(X, y_failure_category, test_size=0.2, random_state=42)

    print("Balancing classes...")
    undersample = RandomUnderSampler(random_state=42)
    X_train_unders, y_train_unders = undersample.fit_resample(X_train, y_train_category)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_unders, y_train_unders)

    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train_balanced)

    joblib.dump(rf_model, "model.pkl")
    print("Model saved as model.pkl")

    return rf_model, scaler, encoder_failure_category

def predict_next_failure_event(vehicle_number, last_mileage, last_load, last_failure_category, rf_model, scaler, encoder_failure_category):
    print(f"Predicting next failure event for vehicle {vehicle_number}...")
    last_failure_category_encoded = encoder_failure_category.transform([last_failure_category])[0]
    input_features = np.array([[last_mileage, last_load, last_failure_category_encoded]])
    input_features_scaled = scaler.transform(input_features)

    predicted_probabilities = rf_model.predict_proba(input_features_scaled)[0]
    predicted_failure_category_encoded = rf_model.predict(input_features_scaled)[0]
    predicted_failure_category = encoder_failure_category.inverse_transform([predicted_failure_category_encoded])[0]

    return predicted_failure_category, predicted_probabilities

def main(vehicle_number, num_days, output_file, generate_only):
    start_date = datetime(2021, 5, 1)
    end_date = datetime.now()

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    failures_file_path = os.path.join(DATA_PATH, "failures_data.csv")

    if not os.path.exists(failures_file_path):
        print("Generating data...")
        all_failures = create_vehicle_maintenance_dataset(start_date, end_date)
        all_failures.to_csv(failures_file_path, index=False, encoding='utf-8')
        print(f"Data generated and saved to {failures_file_path}")
    else:
        all_failures = pd.read_csv(failures_file_path)
        print("Data loaded")

    if generate_only:
        print("Data generated. Exiting.")
        return

    # Train model
    rf_model, scaler, encoder_failure_category = train_random_forest_model(all_failures)

    # Get vehicle failures
    vehicle_failures = all_failures[all_failures['vehicle_number'] == vehicle_number].sort_values('date', ascending=False)

    if vehicle_failures.empty:
        print(f"No failure data found for vehicle {vehicle_number}. Generating additional data...")
        vehicle_data = {'model': 'Mercedes Atego', 'year': 1999, 'initial_mileage': 400000, 'vehicle_number': vehicle_number}
        new_failures = VehicleMaintenanceSystem().generate_failures(vehicle_data, start_date, end_date)
        all_failures = pd.concat([all_failures, new_failures], ignore_index=True)
        all_failures.to_csv(failures_file_path, index=False, encoding='utf-8')
        print(f"Generated failure data for {vehicle_number} and saved.")

    # Display service history
    service_history = vehicle_failures[['date', 'failure_name', 'cost', 'repair_duration']]
    print("\nService History:")
    print(service_history.to_string(index=False))

    latest_failure = vehicle_failures.iloc[0]
    last_mileage = latest_failure['mileage']
    last_load = latest_failure['daily_load']
    last_failure_category = latest_failure['failure_category']

    # Predict next failure event
    next_failure_prediction, predicted_probabilities = predict_next_failure_event(
        vehicle_number, last_mileage, last_load, last_failure_category,
        rf_model, scaler, encoder_failure_category
    )

    # Format probabilities for output
    probability_info = {cat: prob for cat, prob in zip(encoder_failure_category.classes_, predicted_probabilities)}
    probabilities_text = "\n".join([f"- {category}: {probability:.2%}" for category, probability in probability_info.items()])

    # Prepare output
    output = (f"\nVehicle: {vehicle_number}\n"
              f"Predicted Next Failure Category: {next_failure_prediction}\n"
              f"Probabilities of Each Category:\n{probabilities_text}\n"
              f"Estimated Days to Next Failure: {num_days}")

    print(output)

    # Save to file
    if output_file:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(output)
        print(f"Results saved to file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Maintenance Prediction.")
    parser.add_argument("--vehicle", type=str, default=DEFAULT_VEHICLE, help="Vehicle number for prediction.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of days for prediction.")
    parser.add_argument("--output", type=str, help="File to save the results.")
    parser.add_argument("--generate-only", action="store_true", help="Generate data only, without prediction.")
    args = parser.parse_args()

    main(args.vehicle, args.days, args.output, args.generate_only)
