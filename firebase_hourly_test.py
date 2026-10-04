import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np

# ============================================================
# FIREBASE CONFIGURATION
# ============================================================

DATABASE_URL = "https://flood-watch-3862f-default-rtdb.asia-southeast1.firebasedatabase.app/"

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": DATABASE_URL
})

# ============================================================
# READ FIREBASE HISTORY
# ============================================================

ref = db.reference("/sensor_history")

history = ref.get()

if not history:
    print("No Firebase history found.")
    exit()

# ============================================================
# CONVERT FIREBASE DATA TO DATAFRAME
# ============================================================

records = []

for key, value in history.items():

    records.append({
        "firebase_id": key,
        "timestamp": value.get("timestamp", 0),
        "ultrasonic": value.get("water_level", np.nan),
        "rain": value.get("rain", np.nan),
        "temperature": value.get("temperature", np.nan),
        "humidity": value.get("humidity", np.nan),
        "soil": value.get("soil_moisture", np.nan),
        "flow": value.get("flow_rate", np.nan)
    })

df = pd.DataFrame(records)

# ============================================================
# SHOW RAW DATA
# ============================================================

print("=" * 60)
print("RAW FIREBASE DATA")
print("=" * 60)

print("Number of records:", len(df))

print("\nFirst 5 records:")
print(df.head())

# ============================================================
# CONVERT MILLISECONDS TO DATETIME
# ============================================================

# ESP32 timestamp uses millis()
# Convert milliseconds to seconds first.

df["datetime"] = pd.to_datetime(
    df["timestamp"] / 1000,
    unit="s"
)

# ============================================================
# SORT BY TIME
# ============================================================

df = df.sort_values("datetime").reset_index(drop=True)

# ============================================================
# CREATE HOURLY DATA
# ============================================================

hourly = (
    df.set_index("datetime")
      .resample("1h")
      .agg({
          "ultrasonic": "mean",
          "rain": "mean",
          "temperature": "mean",
          "humidity": "mean",
          "soil": "mean",
          "flow": "mean"
      })
      .dropna()
      .reset_index()
)

# ============================================================
# DISPLAY HOURLY DATA
# ============================================================

print("\n" + "=" * 60)
print("HOURLY FIREBASE DATA")
print("=" * 60)

print("Number of hourly records:", len(hourly))

print("\nHourly data:")
print(hourly.head(10))

print("\nLatest hourly reading:")
print(hourly.tail(1).to_string(index=False))

print("=" * 60)