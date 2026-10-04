import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load your dataset
df = pd.read_csv('data.csv')

# 2. Preprocess and sort chronologically
df['Timestamp_Sensor_Read'] = pd.to_datetime(df['Timestamp_Sensor_Read'])
df = df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)

# --- FEATURE ENGINEERING (Adding Memory & Cumulative Effects) ---
df['ultrasonic_lag1'] = df['ultrasonic'].shift(1)
df['ultrasonic_lag3'] = df['ultrasonic'].shift(3)
df['ultrasonic_lag6'] = df['ultrasonic'].shift(6)

df['rain_rolling_3h'] = df['rain'].rolling(window=3).sum()
df['rain_rolling_6h'] = df['rain'].rolling(window=6).sum()
df['rain_rolling_12h'] = df['rain'].rolling(window=12).sum()

# 3. Create Targets for 1h, 6h, 12h, and 24h ahead
df['target_1h'] = df['ultrasonic'].shift(-1)
df['target_6h'] = df['ultrasonic'].shift(-6)
df['target_12h'] = df['ultrasonic'].shift(-12)
df['target_24h'] = df['ultrasonic'].shift(-24)

# Drop rows with NaN values created by shifting and rolling windows (accommodates up to 24h shift)
df = df.dropna().reset_index(drop=True)

# 4. Define Expanded Features (X)
features = [
    'ultrasonic', 'rain', 'temperature', 'humidity', 'soil', 'flow',
    'ultrasonic_lag1', 'ultrasonic_lag3', 'ultrasonic_lag6',
    'rain_rolling_3h', 'rain_rolling_6h', 'rain_rolling_12h'
]
X = df[features]

# Define Targets
y_1h = df['target_1h']
y_6h = df['target_6h']
y_12h = df['target_12h']
y_24h = df['target_24h']

# 5. Train-Test Split (Using random split with matching random_state to keep test indices aligned)
X_train, X_test, y_train_1h, y_test_1h = train_test_split(X, y_1h, test_size=0.3, random_state=42)
_, _, y_train_6h, y_test_6h = train_test_split(X, y_6h, test_size=0.3, random_state=42)
_, _, y_train_12h, y_test_12h = train_test_split(X, y_12h, test_size=0.3, random_state=42)
_, _, y_train_24h, y_test_24h = train_test_split(X, y_24h, test_size=0.3, random_state=42)

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}\n")

# --- MODEL 1: 1-Hour Ahead ---
rf_1h = RandomForestRegressor(n_estimators=100, random_state=42)
rf_1h.fit(X_train, y_train_1h)
preds_1h = rf_1h.predict(X_test)
rmse_1h = np.sqrt(mean_squared_error(y_test_1h, preds_1h))
r2_1h = r2_score(y_test_1h, preds_1h)

# --- MODEL 2: 6-Hours Ahead ---
rf_6h = RandomForestRegressor(n_estimators=100, random_state=42)
rf_6h.fit(X_train, y_train_6h)
preds_6h = rf_6h.predict(X_test)
rmse_6h = np.sqrt(mean_squared_error(y_test_6h, preds_6h))
r2_6h = r2_score(y_test_6h, preds_6h)

# --- MODEL 3: 12-Hours Ahead ---
rf_12h = RandomForestRegressor(n_estimators=100, random_state=42)
rf_12h.fit(X_train, y_train_12h)
preds_12h = rf_12h.predict(X_test)
rmse_12h = np.sqrt(mean_squared_error(y_test_12h, preds_12h))
r2_12h = r2_score(y_test_12h, preds_12h)

# --- MODEL 4: 24-Hours Ahead ---
rf_24h = RandomForestRegressor(n_estimators=100, random_state=42)
rf_24h.fit(X_train, y_train_24h)
preds_24h = rf_24h.predict(X_test)
rmse_24h = np.sqrt(mean_squared_error(y_test_24h, preds_24h))
r2_24h = r2_score(y_test_24h, preds_24h)

# 6. Print Results
print("=== 1-HOUR AHEAD FORECAST ===")
print(f"RMSE: {rmse_1h:.4f} | R² Score: {r2_1h * 100:.2f}%")

print("\n=== 6-HOURS AHEAD FORECAST ===")
print(f"RMSE: {rmse_6h:.4f} | R² Score: {r2_6h * 100:.2f}%")

print("\n=== 12-HOURS AHEAD FORECAST ===")
print(f"RMSE: {rmse_12h:.4f} | R² Score: {r2_12h * 100:.2f}%")

print("\n=== 24-HOURS AHEAD FORECAST ===")
print(f"RMSE: {rmse_24h:.4f} | R² Score: {r2_24h * 100:.2f}%")

# Optional: Side-by-side comparison table
comparison_df = pd.DataFrame({
    'Actual_24h': y_test_24h.values,
    'Pred_1h': preds_1h,
    'Pred_6h': preds_6h,
    'Pred_12h': preds_12h,
    'Pred_24h': preds_24h
})
print("\nSample Predictions Comparison:")
print(comparison_df.head(10))