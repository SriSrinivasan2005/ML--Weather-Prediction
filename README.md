# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
Weather prediction is a complex task due to the variability and interdependence of atmospheric conditions such as temperature, humidity, wind speed, and pressure. Traditional prediction methods may not provide high accuracy when dealing with large and complex datasets.

The objective of this experiment is to implement the Random Forest Algorithm to analyze historical weather data and accurately predict future weather conditions (such as Rain/No Rain or Sunny/Cloudy). The model aims to improve prediction accuracy by combining multiple decision trees and reducing overfitting


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect the dataset by obtaining a weather dataset that includes features such as temperature, humidity, wind speed, pressure, and outlook, along with the target variable representing the weather condition (Rain/No Rain or Sunny/Cloudy).

2. Identify variables by defining X as the set of independent variables (weather attributes) and Y as the dependent variable (weather condition).

3. Preprocess the data by handling missing values, converting categorical data into numerical form, and splitting the dataset into training and testing sets.

4. Initialize Random Forest parameters by selecting the number of trees (N), number of features (m) considered at each split, and the splitting criterion such as Gini Index or Entropy.

5. Create bootstrap samples by generating multiple random samples with replacement from the training dataset.

6. Build decision trees by, for each bootstrap sample, selecting a random subset of features, finding the best split using Gini Index or Entropy, and recursively splitting the data to grow the tree.

7. Apply stopping conditions by stopping the tree growth when all data belongs to the same class, maximum depth is reached, or the minimum number of samples per node is met.

8. Assign class labels by labeling each leaf node with the majority class such as Rain or No Rain.

9. Train the Random Forest model by constructing multiple decision trees using different bootstrap samples and feature subsets.

10. Test the model by using the testing dataset to predict weather conditions through all the trees.

11. Aggregate predictions by combining the outputs of all trees using majority voting for classification or averaging for regression.

12. Evaluate model performance by calculating metrics such as accuracy, precision, recall, and confusion matrix.

13. Output the result by displaying the predicted weather condition for the given input data.
 

## Program:
```PYTHON
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: SRI SRINIVASAN K
RegisterNumber:  212224220104
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
# Read raw CSV and fix column whitespace
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

# 1.1 Chronological Sorting: Ensure time flows correctly
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 1.2 Interpolation: Fill gaps (up to 10 rows) to keep the timeline continuous
cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
# 2.1 Cyclical Time Features: Convert hour into circle coordinates (Sin/Cos)
df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 2.2 Lag Features: Give the model 'Memory' of what happened 1 and 2 steps ago
targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

# 2.3 Cleanup: Drop rows where lags are NaN and save processed data
processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

# Define the final high-performance feature set
features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
# Print summary of feature engineering
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)
# ==============================================================================
# 3. TRAIN-TEST SPLIT (Chronological)
# ==============================================================================
# Take the first 80% for training and the final 20% for testing (no shuffling)
split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}

# ==============================================================================
# 4. TRAINING & PERFORMANCE EVALUATION
# ==============================================================================
target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    
    # Random Forest with high-depth logic for complex weather patterns
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    models[target] = model
    
    # Store metrics for interpretation
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
    # Plot 1: Actual vs Predicted (Showing the last 150 points for detail)
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Which sensors influenced this target most?)
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

# ==============================================================================
# 6. REAL-TIME PREDICTION (Next Step)
# ==============================================================================
last_row = processed_df.iloc[-1]
latest_data = pd.DataFrame([{
    'hum': last_row['hum'], 'pressure': last_row['pressure'], 'wind_speed': last_row['wind_speed'],
    'illumination': last_row['illumination'], 'co2': last_row['co2'],
    'hour_sin': last_row['hour_sin'], 'hour_cos': last_row['hour_cos'],
    'tem_lag1': last_row['tem'], 'pm2_5_lag1': last_row['pm2_5'], 'tsr_lag1': last_row['tsr']
}])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")
for target in targets:
    pred_val = models[target].predict(latest_data)[0]
    print(f"Predicted {target_meta[target][0]}: {pred_val:.2f} {target_meta[target][1]}")
```

## Output:
<img width="1126" height="531" alt="image" src="https://github.com/user-attachments/assets/59b3a97c-c189-44b1-9ca2-6ef792aa8df7" />

<img width="1148" height="858" alt="image" src="https://github.com/user-attachments/assets/16dc83de-b5f6-4ea1-b354-72425713dcc1" />

<img width="1131" height="115" alt="image" src="https://github.com/user-attachments/assets/0b111541-7d2c-45a7-864a-474db5ea4350" />

## Result:
The Random Forest algorithm was successfully implemented for weather prediction using the dataset. The model handled multiple features effectively, achieved high accuracy, and reduced overfitting compared to a single decision tree. Overall, it provided reliable and efficient weather predictions.
