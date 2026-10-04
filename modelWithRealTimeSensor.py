import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv('data.csv')
df['Timestamp_Sensor_Read'] = pd.to_datetime(df['Timestamp_Sensor_Read'])
df = df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)

# 2. Feature Engineering
df['ultrasonic_lag1'] = df['ultrasonic'].shift(1)
df['ultrasonic_lag3'] = df['ultrasonic'].shift(3)
df['ultrasonic_lag6'] = df['ultrasonic'].shift(6)

df['rain_rolling_3h'] = df['rain'].rolling(window=3).sum()
df['rain_rolling_6h'] = df['rain'].rolling(window=6).sum()
df['rain_rolling_12h'] = df['rain'].rolling(window=12).sum()

# Create Targets (extended up to 92 hours)
df['target_1h'] = df['ultrasonic'].shift(-1)
df['target_6h'] = df['ultrasonic'].shift(-6)
df['target_12h'] = df['ultrasonic'].shift(-12)
df['target_24h'] = df['ultrasonic'].shift(-24)
df['target_48h'] = df['ultrasonic'].shift(-48)
df['target_72h'] = df['ultrasonic'].shift(-72)
df['target_92h'] = df['ultrasonic'].shift(-92)

# Drop rows with NaN values (accommodates up to 92h shift)
df = df.dropna().reset_index(drop=True)

features = [
    'ultrasonic', 'rain', 'temperature', 'humidity', 'soil', 'flow',
    'ultrasonic_lag1', 'ultrasonic_lag3', 'ultrasonic_lag6',
    'rain_rolling_3h', 'rain_rolling_6h', 'rain_rolling_12h'
]
X = df[features]

# 3. Train Models & Capture Test Metrics for Accuracy Reference
X_train, X_test, y_train_1h, y_test_1h = train_test_split(X, df['target_1h'], test_size=0.3, random_state=42)
_, _, y_train_6h, y_test_6h = train_test_split(X, df['target_6h'], test_size=0.3, random_state=42)
_, _, y_train_12h, y_test_12h = train_test_split(X, df['target_12h'], test_size=0.3, random_state=42)
_, _, y_train_24h, y_test_24h = train_test_split(X, df['target_24h'], test_size=0.3, random_state=42)
_, _, y_train_48h, y_test_48h = train_test_split(X, df['target_48h'], test_size=0.3, random_state=42)
_, _, y_train_72h, y_test_72h = train_test_split(X, df['target_72h'], test_size=0.3, random_state=42)
_, _, y_train_92h, y_test_92h = train_test_split(X, df['target_92h'], test_size=0.3, random_state=42)

rf_1h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_1h)
rf_6h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_6h)
rf_12h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_12h)
rf_24h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_24h)
rf_48h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_48h)
rf_72h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_72h)
rf_92h = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train_92h)

# Accuracies
acc_1h = r2_score(y_test_1h, rf_1h.predict(X_test)) * 100
rmse_1h = np.sqrt(mean_squared_error(y_test_1h, rf_1h.predict(X_test)))

acc_6h = r2_score(y_test_6h, rf_6h.predict(X_test)) * 100
rmse_6h = np.sqrt(mean_squared_error(y_test_6h, rf_6h.predict(X_test)))

acc_12h = r2_score(y_test_12h, rf_12h.predict(X_test)) * 100
rmse_12h = np.sqrt(mean_squared_error(y_test_12h, rf_12h.predict(X_test)))

acc_24h = r2_score(y_test_24h, rf_24h.predict(X_test)) * 100
rmse_24h = np.sqrt(mean_squared_error(y_test_24h, rf_24h.predict(X_test)))

acc_48h = r2_score(y_test_48h, rf_48h.predict(X_test)) * 100
rmse_48h = np.sqrt(mean_squared_error(y_test_48h, rf_48h.predict(X_test)))

acc_72h = r2_score(y_test_72h, rf_72h.predict(X_test)) * 100
rmse_72h = np.sqrt(mean_squared_error(y_test_72h, rf_72h.predict(X_test)))

acc_92h = r2_score(y_test_92h, rf_92h.predict(X_test)) * 100
rmse_92h = np.sqrt(mean_squared_error(y_test_92h, rf_92h.predict(X_test)))

# 4. Grab a sample storm/high-water index from your dataset to demonstrate a dynamic event
storm_sample_idx = X[X['rain'] > 3.0].index[0] if len(X[X['rain'] > 3.0]) > 0 else -1
sample_row = X.iloc[[storm_sample_idx]]
sample_time = df['Timestamp_Sensor_Read'].iloc[storm_sample_idx]

# 5. Predictions
pred_1h = rf_1h.predict(sample_row)[0]
pred_6h = rf_6h.predict(sample_row)[0]
pred_12h = rf_12h.predict(sample_row)[0]
pred_24h = rf_24h.predict(sample_row)[0]
pred_48h = rf_48h.predict(sample_row)[0]
pred_72h = rf_72h.predict(sample_row)[0]
pred_92h = rf_92h.predict(sample_row)[0]

# 6. Display Dashboard Output
print("=" * 60)
print("BALILI RIVER FLOOD EARLY WARNING SYSTEM - SAMPLE SIMULATION")
print("=" * 60)
print(f"Timestamp Simulated: {sample_time}\n")

print("--- CURRENT SENSOR TELEMETRY (ESP32 Node) ---")
print(f"• Water Level (Ultrasonic) : {sample_row['ultrasonic'].values[0]:.3f} m")
print(f"• Rainfall Rate            : {sample_row['rain'].values[0]:.2f} mm/h")
print(f"• Temperature              : {sample_row['temperature'].values[0]:.1f} °C")
print(f"• Humidity                 : {sample_row['humidity'].values[0]:.1f} %")
print(f"• Soil Moisture            : {sample_row['soil'].values[0]:.1f} %")
print(f"• Water Flow               : {sample_row['flow'].values[0]:.2f} m³/s\n")

print("--- WATER LEVEL FORECASTS & MODEL ACCURACY ---")
print(f"1. 1-Hour Ahead Forecast   : {pred_1h:.3f} m")
print(f"   └ Accuracy (R²): {acc_1h:.2f}% | Error (RMSE): ±{rmse_1h:.4f} m\n")

print(f"2. 6-Hours Ahead Forecast  : {pred_6h:.3f} m")
print(f"   └ Accuracy (R²): {acc_6h:.2f}% | Error (RMSE): ±{rmse_6h:.4f} m\n")

print(f"3. 12-Hours Ahead Forecast : {pred_12h:.3f} m")
print(f"   └ Accuracy (R²): {acc_12h:.2f}% | Error (RMSE): ±{rmse_12h:.4f} m\n")

print(f"4. 24-Hours Ahead Forecast : {pred_24h:.3f} m")
print(f"   └ Accuracy (R²): {acc_24h:.2f}% | Error (RMSE): ±{rmse_24h:.4f} m\n")

print(f"5. 48-Hours Ahead Forecast : {pred_48h:.3f} m")
print(f"   └ Accuracy (R²): {acc_48h:.2f}% | Error (RMSE): ±{rmse_48h:.4f} m\n")

print(f"6. 72-Hours Ahead Forecast : {pred_72h:.3f} m")
print(f"   └ Accuracy (R²): {acc_72h:.2f}% | Error (RMSE): ±{rmse_72h:.4f} m\n")

print(f"7. 92-Hours Ahead Forecast : {pred_92h:.3f} m")
print(f"   └ Accuracy (R²): {acc_92h:.2f}% | Error (RMSE): ±{rmse_92h:.4f} m")
print("=" * 60)