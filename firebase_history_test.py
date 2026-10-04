import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

# ============================================
# FIREBASE CONFIGURATION
# ============================================

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
# CONVERT TO DATAFRAME
# ============================================================

records = []

for key, value in history.items():

    timestamp = value.get("timestamp")

    # Only keep records with the new real timestamp
    if isinstance(timestamp, str):

        records.append({
            "firebase_id": key,
            "timestamp": timestamp,
            "water_level": value.get("water_level"),
            "rain": value.get("rain"),
            "temperature": value.get("temperature"),
            "humidity": value.get("humidity"),
            "soil_moisture": value.get("soil_moisture"),
            "flow_rate": value.get("flow_rate")
        })

df = pd.DataFrame(records)

# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("NEW FIREBASE TIMESTAMP TEST")
print("=" * 60)

print("New timestamped records:", len(df))

if len(df) > 0:

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    print("\nFirst new record:")
    print(df.iloc[0].to_string())

    print("\nLatest new record:")
    print(df.iloc[-1].to_string())

    print("\nTime difference between last two records:")

    if len(df) >= 2:

        difference = (
            df["timestamp"].iloc[-1]
            - df["timestamp"].iloc[-2]
        )

        print(difference)

else:

    print(
        "\nNo new timestamped records found."
    )

print("=" * 60)