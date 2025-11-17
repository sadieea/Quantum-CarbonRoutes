import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import json
import joblib
import os

def load_training_data():
    try:
        df = pd.read_csv('nyc_taxi_data.csv')
    except FileNotFoundError:
        # Create dummy training data: daily demand per location for 4 years
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        locations = range(1, 11)  # Assume 10 locations
        data = []
        for d in dates:
            for loc in locations:
                # Simulate demand with some variation by location and day
                base_demand = 20 + loc * 2  # Higher demand for higher location IDs
                day_factor = 1 + 0.5 * (d.weekday() < 5)  # Higher on weekdays
                demand = np.random.poisson(base_demand * day_factor)
                data.append({'date': d.date(), 'location_id': loc, 'demand': demand})
        df = pd.DataFrame(data)

    # Aggregate to ensure it's daily demand per location (sum if multiple entries per day-loc)
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['date', 'location_id']).agg({'demand': 'sum'}).reset_index()

    return df

def train_forecasting_model(data):
    # Prepare features
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month

    # Features: dayofweek, month, location_id
    features = ['dayofweek', 'month', 'location_id']
    X = data[features]
    y = data['demand']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LGBMRegressor
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    return model

def generate_demand_predictions(customer_locations_df):
    # Load training data
    data = load_training_data()

    # Load or train model
    model_path = 'forecasting_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = train_forecasting_model(data)
        joblib.dump(model, model_path)

    # Predict for next day
    tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
    tomorrow_dayofweek = tomorrow.weekday()
    tomorrow_month = tomorrow.month

    predictions = {}
    unique_locations = customer_locations_df['location_id'].unique()

    for loc_id in unique_locations:
        # Prepare prediction features
        pred_features = pd.DataFrame({
            'dayofweek': [tomorrow_dayofweek],
            'month': [tomorrow_month],
            'location_id': [loc_id]
        })

        # Predict demand
        predicted_demand = model.predict(pred_features)[0]
        predicted_demand = int(round(predicted_demand))

        # Assign to all customers with this location_id
        customers_at_loc = customer_locations_df[customer_locations_df['location_id'] == loc_id]
        for _, row in customers_at_loc.iterrows():
            customer_id = str(int(row['customer_id']))
            predictions[customer_id] = predicted_demand

    # Save predictions to JSON
    with open('predicted_demand.json', 'w') as f:
        json.dump(predictions, f, indent=2)

    return predictions

if __name__ == "__main__":
    # Sample usage: Create a sample customer_locations_df
    # Assuming 100 customers with random location assignments (1-10)
    np.random.seed(42)
    sample_customers = list(range(1, 101))
    sample_locations = np.random.randint(1, 11, 100)
    customer_locations_df = pd.DataFrame({
        'customer_id': sample_customers,
        'location_id': sample_locations
    })

    # Generate predictions
    predictions = generate_demand_predictions(customer_locations_df)
    print("Predictions generated and saved to predicted_demand.json")
    print(f"Sample predictions: {dict(list(predictions.items())[:5])}")
