import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = (
    "https://flood-watch-3862f-default-rtdb."
    "asia-southeast1.firebasedatabase.app/"
)

SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"
DATASET_FILE = "data.csv"


# ============================================================
# 1. CONNECT TO FIREBASE
# ============================================================

print("=" * 60)
print("FLOOD WATCH - FIREBASE + RANDOM FOREST INTEGRATION")
print("=" * 60)

print("\nConnecting to Firebase...")

cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": DATABASE_URL
    }
)

print("Firebase connection successful!")


# ============================================================
# 2. READ LATEST SENSOR DATA FROM FIREBASE
# ============================================================

print("\nReading latest ESP32 sensor data...")

sensor_ref = db.reference("/sensor_data")
live_data = sensor_ref.get()

if not live_data:
    print("ERROR: No sensor data found in Firebase.")
    exit()

print("\nLIVE FIREBASE SENSOR DATA")
print("-" * 60)

print(f"Water Level   : {live_data.get('water_level', 0)} m")
print(f"Rain          : {live_data.get('rain', 0)}")
print(f"Temperature   : {live_data.get('temperature', 0)} °C")
print(f"Humidity      : {live_data.get('humidity', 0)} %")
print(f"Soil Moisture : {live_data.get('soil_moisture', 0)} %")
print(f"Flow Rate     : {live_data.get('flow_rate', 0)} L/min")
print(f"Timestamp     : {live_data.get('timestamp', 'N/A')}")


# ============================================================
# 3. LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

df = pd.read_csv(DATASET_FILE)

df["Timestamp_Sensor_Read"] = pd.to_datetime(
    df["Timestamp_Sensor_Read"]
)

df = df.sort_values(
    "Timestamp_Sensor_Read"
).reset_index(drop=True)

print(f"Training records: {len(df)}")


# ============================================================
# 4. CREATE LAG FEATURES
# ============================================================

df["ultrasonic_lag1"] = df["ultrasonic"].shift(1)
df["ultrasonic_lag3"] = df["ultrasonic"].shift(3)
df["ultrasonic_lag6"] = df["ultrasonic"].shift(6)


# ============================================================
# 5. CREATE RAIN ROLLING FEATURES
# ============================================================

df["rain_rolling_3h"] = (
    df["rain"].rolling(window=3).sum()
)

df["rain_rolling_6h"] = (
    df["rain"].rolling(window=6).sum()
)

df["rain_rolling_12h"] = (
    df["rain"].rolling(window=12).sum()
)


# ============================================================
# 6. CREATE FUTURE TARGETS
# ============================================================

df["target_1h"] = df["ultrasonic"].shift(-1)
df["target_6h"] = df["ultrasonic"].shift(-6)
df["target_12h"] = df["ultrasonic"].shift(-12)
df["target_24h"] = df["ultrasonic"].shift(-24)
df["target_48h"] = df["ultrasonic"].shift(-48)
df["target_72h"] = df["ultrasonic"].shift(-72)
df["target_92h"] = df["ultrasonic"].shift(-92)


# ============================================================
# 7. REMOVE MISSING VALUES
# ============================================================

df = df.dropna().reset_index(drop=True)


# ============================================================
# 8. DEFINE FEATURES
# ============================================================

features = [
    "ultrasonic",
    "rain",
    "temperature",
    "humidity",
    "soil",
    "flow",
    "ultrasonic_lag1",
    "ultrasonic_lag3",
    "ultrasonic_lag6",
    "rain_rolling_3h",
    "rain_rolling_6h",
    "rain_rolling_12h"
]

X = df[features]


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train_1h, y_test_1h = train_test_split(
    X,
    df["target_1h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_6h, y_test_6h = train_test_split(
    X,
    df["target_6h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_12h, y_test_12h = train_test_split(
    X,
    df["target_12h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_24h, y_test_24h = train_test_split(
    X,
    df["target_24h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_48h, y_test_48h = train_test_split(
    X,
    df["target_48h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_72h, y_test_72h = train_test_split(
    X,
    df["target_72h"],
    test_size=0.3,
    random_state=42
)

_, _, y_train_92h, y_test_92h = train_test_split(
    X,
    df["target_92h"],
    test_size=0.3,
    random_state=42
)


# ============================================================
# 10. TRAIN RANDOM FOREST MODELS
# ============================================================

print("\nTraining Random Forest models...")

rf_1h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_6h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_12h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_24h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_48h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_72h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_92h = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


rf_1h.fit(X_train, y_train_1h)
rf_6h.fit(X_train, y_train_6h)
rf_12h.fit(X_train, y_train_12h)
rf_24h.fit(X_train, y_train_24h)
rf_48h.fit(X_train, y_train_48h)
rf_72h.fit(X_train, y_train_72h)
rf_92h.fit(X_train, y_train_92h)

print("Random Forest training complete!")


# ============================================================
# 11. CREATE LIVE INPUT FROM FIREBASE
# ============================================================

print("\nPreparing live Firebase data for the model...")

current_ultrasonic = float(
    live_data.get("water_level", 0)
)

current_rain = float(
    live_data.get("rain", 0)
)

current_temperature = float(
    live_data.get("temperature", 0)
)

current_humidity = float(
    live_data.get("humidity", 0)
)

current_soil = float(
    live_data.get("soil_moisture", 0)
)

current_flow = float(
    live_data.get("flow_rate", 0)
)


# ============================================================
# 12. USE TRAINING HISTORY FOR LAG / ROLLING CONTEXT
# ============================================================

# The Firebase demonstration currently contains only a short
# amount of history. Therefore, the existing training dataset
# supplies the historical lag and rolling features required by
# the current Random Forest model.

recent_history = df.tail(12).copy()

lag1 = recent_history["ultrasonic"].iloc[-1]
lag3 = recent_history["ultrasonic"].iloc[-3]
lag6 = recent_history["ultrasonic"].iloc[-6]

rain_rolling_3h = (
    recent_history["rain"].tail(3).sum()
)

rain_rolling_6h = (
    recent_history["rain"].tail(6).sum()
)

rain_rolling_12h = (
    recent_history["rain"].tail(12).sum()
)


# ============================================================
# 13. BUILD MODEL INPUT
# ============================================================

live_features = pd.DataFrame(
    [[
        current_ultrasonic,
        current_rain,
        current_temperature,
        current_humidity,
        current_soil,
        current_flow,
        lag1,
        lag3,
        lag6,
        rain_rolling_3h,
        rain_rolling_6h,
        rain_rolling_12h
    ]],
    columns=features
)


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

prediction_1h = rf_1h.predict(live_features)[0]
prediction_6h = rf_6h.predict(live_features)[0]
prediction_12h = rf_12h.predict(live_features)[0]
prediction_24h = rf_24h.predict(live_features)[0]
prediction_48h = rf_48h.predict(live_features)[0]
prediction_72h = rf_72h.predict(live_features)[0]
prediction_92h = rf_92h.predict(live_features)[0]


# ============================================================
# 15. DETERMINE FLOOD STATUS
# ============================================================

if prediction_24h >= 2.5:
    status = "HIGH WATER LEVEL"
elif prediction_24h >= 2.0:
    status = "MODERATE WATER LEVEL"
else:
    status = "NORMAL WATER LEVEL"


# ============================================================
# 16. DISPLAY PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("FLOOD WATCH PREDICTIONS")
print("=" * 60)

print(f"\nCurrent Water Level : {current_ultrasonic:.3f} m")

print("\nPredicted Water Level")
print("-" * 60)

print(f"1 hour   : {prediction_1h:.3f} m")
print(f"6 hours  : {prediction_6h:.3f} m")
print(f"12 hours : {prediction_12h:.3f} m")
print(f"24 hours : {prediction_24h:.3f} m")
print(f"48 hours : {prediction_48h:.3f} m")
print(f"72 hours : {prediction_72h:.3f} m")
print(f"92 hours : {prediction_92h:.3f} m")


# ============================================================
# 17. DISPLAY FLOOD STATUS
# ============================================================

print("\n")
print("=" * 60)
print("FLOOD WATCH STATUS")
print("=" * 60)

print(f"\n24-Hour Prediction: {prediction_24h:.3f} m")
print(f"Status: {status}")


# ============================================================
# 18. SAVE PREDICTIONS TO FIREBASE
# ============================================================

print("\nSaving predictions to Firebase...")

prediction_data = {
    "timestamp": live_data.get(
        "timestamp",
        "N/A"
    ),

    "current_water_level": float(
        current_ultrasonic
    ),

    "prediction_1h": float(
        prediction_1h
    ),

    "prediction_6h": float(
        prediction_6h
    ),

    "prediction_12h": float(
        prediction_12h
    ),

    "prediction_24h": float(
        prediction_24h
    ),

    "prediction_48h": float(
        prediction_48h
    ),

    "prediction_72h": float(
        prediction_72h
    ),

    "prediction_92h": float(
        prediction_92h
    ),

    "status": status
}


prediction_ref = db.reference(
    "/predictions"
)

prediction_ref.set(
    prediction_data
)

print("Predictions successfully saved to Firebase!")


# ============================================================
# 19. COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)