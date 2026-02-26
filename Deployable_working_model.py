import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Data
print("Loading data...")
df = pd.read_excel("demo.xlsx") 
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
 
# Ensure chronological order
df = df.sort_values('Timestamp')

# 2. FEATURE ENGINEERING
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# THE SECRET WEAPON: 24-Hour Lag Feature (Yesterday's Demand)
# This allows the model to adjust to new summer trends!
df['Demand_Lag_24h'] = df['Demand (MW)'].shift(24)

# Drop the first 24 rows because they don't have a "yesterday" to look back at
df = df.dropna()

# 3. Chronological Split (Train: Jan-May, Test: June)
train_data = df[df['Month'] <= 5]
test_data = df[df['Month'] == 6]

# 4. Define Inputs (X) and Target (y)
# Notice we added our new lag feature to the inputs!
features = ['Hour', 'DayOfWeek', 'Month', 'Solar(MW)', 'Wind(MW)', 'Demand_Lag_24h']
target = 'Demand (MW)'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print(f"Training on {len(X_train)} hours. Testing on {len(X_test)} hours.")

# 5. Initialize and Train the Model
print("Training the upgraded model... (Please wait)")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions
predictions = model.predict(X_test)

# 7. Evaluate the Model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n=== FINAL MODEL RESULTS ===")
print(f"R-squared (R2): {r2:.3f}")
print(f"MAE: {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print("===========================\n")

# 8. Save the Visual Results (Fixing the missing graph issue!)
print("Saving graph to your folder...")
plt.figure(figsize=(14, 6))
plt.plot(y_test.values[:168], label='Actual Electricity Demand', color='blue', linewidth=2)
plt.plot(predictions[:168], label='Model Predicted Demand', color='orange', linestyle='dashed', linewidth=2)

plt.title('Machine Learning Demand Forecasting: Actual vs Predicted (June 2024)', fontsize=16)
plt.xlabel('Hours', fontsize=12)
plt.ylabel('Demand (MW)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save as an image file instead of opening a popup
plt.savefig('demand_forecast_graph.png')
print("✅ Success! Open the file 'demand_forecast_graph.png' in your folder to see the graph.")