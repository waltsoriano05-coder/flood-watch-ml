import os
import json
import pandas as pd
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
# FIREBASE CONNECTION
# ============================================================

def initialize_firebase():

    if firebase_admin._apps:
        return

    # Render will use an environment variable.
    firebase_credentials = os.environ.get(
        "FIREBASE_CREDENTIALS"
    )

    if firebase_credentials:

        cred_dict = json.loads(
            firebase_credentials
        )

        cred = credentials.Certificate(
            cred_dict
        )

    else:

        # Local computer uses serviceAccountKey.json
        cred = credentials.Certificate(
            SERVICE_ACCOUNT_FILE
        )

    firebase_admin.initialize_app(
        cred,
        {
            "databaseURL": DATABASE_URL
        }
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def run_prediction():

    print("=" * 60)
    print("FLOOD WATCH - FIREBASE + RANDOM FOREST")
    print("=" * 60)

    # --------------------------------------------------------
    # CONNECT TO FIREBASE
    # --------------------------------------------------------

    print("\nConnecting to Firebase...")

    initialize_firebase()

    print("Firebase connection successful!")

    # --------------------------------------------------------
    # READ SENSOR DATA
    # --------------------------------------------------------

    print("\nReading latest ESP32 sensor data...")

    sensor_ref = db.reference(
        "/sensor_data"
    )

    live_data = sensor_ref.get()

    if not live_data:

        raise Exception(
            "No sensor data found in Firebase."
        )

    print("\nLIVE FIREBASE SENSOR DATA")
    print("-" * 60)

    print(
        f"Water Level   : "
        f"{live_data.get('water_level', 0)} m"
    )

    print(
        f"Rain          : "
        f"{live_data.get('rain', 0)}"
    )

    print(
        f"Temperature   : "
        f"{live_data.get('temperature', 0)} °C"
    )

    print(
        f"Humidity      : "
        f"{live_data.get('humidity', 0)} %"
    )

    print(
        f"Soil Moisture : "
        f"{live_data.get('soil_moisture', 0)} %"
    )

    print(
        f"Flow Rate     : "
        f"{live_data.get('flow_rate', 0)} L/min"
    )

    print(
        f"Timestamp     : "
        f"{live_data.get('timestamp', 'N/A')}"
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    df = pd.read_csv(
        DATASET_FILE
    )

    df["Timestamp_Sensor_Read"] = pd.to_datetime(
        df["Timestamp_Sensor_Read"]
    )

    df = df.sort_values(
        "Timestamp_Sensor_Read"
    ).reset_index(drop=True)

    print(
        f"Training records: {len(df)}"
    )

    # --------------------------------------------------------
    # CREATE LAG FEATURES
    # --------------------------------------------------------

    df["ultrasonic_lag1"] = (
        df["ultrasonic"].shift(1)
    )

    df["ultrasonic_lag3"] = (
        df["ultrasonic"].shift(3)
    )

    df["ultrasonic_lag6"] = (
        df["ultrasonic"].shift(6)
    )

    # --------------------------------------------------------
    # CREATE RAIN FEATURES
    # --------------------------------------------------------

    df["rain_rolling_3h"] = (
        df["rain"]
        .rolling(window=3)
        .sum()
    )

    df["rain_rolling_6h"] = (
        df["rain"]
        .rolling(window=6)
        .sum()
    )

    df["rain_rolling_12h"] = (
        df["rain"]
        .rolling(window=12)
        .sum()
    )

    # --------------------------------------------------------
    # CREATE FUTURE TARGETS
    # --------------------------------------------------------

    df["target_1h"] = (
        df["ultrasonic"].shift(-1)
    )

    df["target_6h"] = (
        df["ultrasonic"].shift(-6)
    )

    df["target_12h"] = (
        df["ultrasonic"].shift(-12)
    )

    df["target_24h"] = (
        df["ultrasonic"].shift(-24)
    )

    df["target_48h"] = (
        df["ultrasonic"].shift(-48)
    )

    df["target_72h"] = (
        df["ultrasonic"].shift(-72)
    )

    df["target_92h"] = (
        df["ultrasonic"].shift(-92)
    )

    # --------------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------------

    df = df.dropna().reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # DEFINE FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train_1h, y_test_1h = (
        train_test_split(
            X,
            df["target_1h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_6h, y_test_6h = (
        train_test_split(
            X,
            df["target_6h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_12h, y_test_12h = (
        train_test_split(
            X,
            df["target_12h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_24h, y_test_24h = (
        train_test_split(
            X,
            df["target_24h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_48h, y_test_48h = (
        train_test_split(
            X,
            df["target_48h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_72h, y_test_72h = (
        train_test_split(
            X,
            df["target_72h"],
            test_size=0.3,
            random_state=42
        )
    )

    _, _, y_train_92h, y_test_92h = (
        train_test_split(
            X,
            df["target_92h"],
            test_size=0.3,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # TRAIN RANDOM FOREST
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest models..."
    )

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

    rf_1h.fit(
        X_train,
        y_train_1h
    )

    rf_6h.fit(
        X_train,
        y_train_6h
    )

    rf_12h.fit(
        X_train,
        y_train_12h
    )

    rf_24h.fit(
        X_train,
        y_train_24h
    )

    rf_48h.fit(
        X_train,
        y_train_48h
    )

    rf_72h.fit(
        X_train,
        y_train_72h
    )

    rf_92h.fit(
        X_train,
        y_train_92h
    )

    print(
        "Random Forest training complete!"
    )

    # --------------------------------------------------------
    # LIVE FIREBASE INPUT
    # --------------------------------------------------------

    print(
        "\nPreparing live Firebase data..."
    )

    current_ultrasonic = float(
        live_data.get(
            "water_level",
            0
        )
    )

    current_rain = float(
        live_data.get(
            "rain",
            0
        )
    )

    current_temperature = float(
        live_data.get(
            "temperature",
            0
        )
    )

    current_humidity = float(
        live_data.get(
            "humidity",
            0
        )
    )

    current_soil = float(
        live_data.get(
            "soil_moisture",
            0
        )
    )

    current_flow = float(
        live_data.get(
            "flow_rate",
            0
        )
    )

    # --------------------------------------------------------
    # HISTORICAL CONTEXT
    # --------------------------------------------------------

    recent_history = df.tail(
        12
    ).copy()

    lag1 = (
        recent_history["ultrasonic"]
        .iloc[-1]
    )

    lag3 = (
        recent_history["ultrasonic"]
        .iloc[-3]
    )

    lag6 = (
        recent_history["ultrasonic"]
        .iloc[-6]
    )

    rain_rolling_3h = (
        recent_history["rain"]
        .tail(3)
        .sum()
    )

    rain_rolling_6h = (
        recent_history["rain"]
        .tail(6)
        .sum()
    )

    rain_rolling_12h = (
        recent_history["rain"]
        .tail(12)
        .sum()
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    prediction_1h = (
        rf_1h.predict(
            live_features
        )[0]
    )

    prediction_6h = (
        rf_6h.predict(
            live_features
        )[0]
    )

    prediction_12h = (
        rf_12h.predict(
            live_features
        )[0]
    )

    prediction_24h = (
        rf_24h.predict(
            live_features
        )[0]
    )

    prediction_48h = (
        rf_48h.predict(
            live_features
        )[0]
    )

    prediction_72h = (
        rf_72h.predict(
            live_features
        )[0]
    )

    prediction_92h = (
        rf_92h.predict(
            live_features
        )[0]
    )

    # --------------------------------------------------------
    # FLOOD STATUS
    # --------------------------------------------------------

    if prediction_24h >= 2.5:

        status = (
            "HIGH WATER LEVEL"
        )

    elif prediction_24h >= 2.0:

        status = (
            "MODERATE WATER LEVEL"
        )

    else:

        status = (
            "NORMAL WATER LEVEL"
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FLOOD WATCH PREDICTIONS")
    print("=" * 60)

    print(
        f"\nCurrent Water Level : "
        f"{current_ultrasonic:.3f} m"
    )

    print("\nPredicted Water Level")
    print("-" * 60)

    print(
        f"1 hour   : "
        f"{prediction_1h:.3f} m"
    )

    print(
        f"6 hours  : "
        f"{prediction_6h:.3f} m"
    )

    print(
        f"12 hours : "
        f"{prediction_12h:.3f} m"
    )

    print(
        f"24 hours : "
        f"{prediction_24h:.3f} m"
    )

    print(
        f"48 hours : "
        f"{prediction_48h:.3f} m"
    )

    print(
        f"72 hours : "
        f"{prediction_72h:.3f} m"
    )

    print(
        f"92 hours : "
        f"{prediction_92h:.3f} m"
    )

    print("\n")
    print("=" * 60)
    print("FLOOD WATCH STATUS")
    print("=" * 60)

    print(
        f"\n24-Hour Prediction: "
        f"{prediction_24h:.3f} m"
    )

    print(
        f"Status: {status}"
    )

    # --------------------------------------------------------
    # SAVE TO FIREBASE
    # --------------------------------------------------------

    print(
        "\nSaving predictions to Firebase..."
    )

    prediction_data = {

        "timestamp": live_data.get(
            "timestamp",
            "N/A"
        ),

        "current_water_level":
            float(
                current_ultrasonic
            ),

        "prediction_1h":
            float(
                prediction_1h
            ),

        "prediction_6h":
            float(
                prediction_6h
            ),

        "prediction_12h":
            float(
                prediction_12h
            ),

        "prediction_24h":
            float(
                prediction_24h
            ),

        "prediction_48h":
            float(
                prediction_48h
            ),

        "prediction_72h":
            float(
                prediction_72h
            ),

        "prediction_92h":
            float(
                prediction_92h
            ),

        "status":
            status
    }

    prediction_ref = db.reference(
        "/predictions"
    )

    prediction_ref.set(
        prediction_data
    )

    print(
        "Predictions successfully saved "
        "to Firebase!"
    )

    print("\n")
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    # --------------------------------------------------------
    # RETURN RESULTS TO FLASK
    # --------------------------------------------------------

    return prediction_data


# ============================================================
# ALLOW DIRECT LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    run_prediction()