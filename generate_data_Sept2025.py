import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for September 2025 (Peak Wet Season / Tropical Storm Nando)
date_range = pd.date_range(start='2025-09-01 00:00:00', end='2025-09-30 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad September Environmental Parameters:
# Temperature: Daily highs ~21°C to 24°C, lows ~18°C to 20°C
temperature = 21.0 + 2.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)
temperature = np.clip(temperature, 17.5, 24.5)

# Humidity: Exceptionally high, frequently hitting near-saturation points
humidity = 90.0 + 5.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.5, n)
humidity = np.clip(humidity, 82.0, 99.0)

# Soil Moisture: Heavily saturated throughout the month due to continuous precipitation
soil = 88.0 + 4.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.8, n)
soil = np.clip(soil, 78.0, 98.0)

# 4. Simulate Rainfall: Volatile with regular thundershowers and a major peak for Tropical Storm Nando (approx. Sept 22)
rain = np.zeros(n)
for day in range(30):
    # Simulate Tropical Storm Nando impact around September 22 (day index 21-23)
    if day in [21, 22, 23]:
        start_h = day * 24
        end_h = min(n, (day + 2) * 24)
        rain[start_h:end_h] = np.random.uniform(4.0, 11.0, end_h - start_h)
    elif np.random.rand() < 0.40:  # Frequent afternoon/evening wet-season thundershowers
        peak_hour = day * 24 + np.random.randint(14, 19)
        start_h = max(0, peak_hour - 2)
        end_h = min(n, peak_hour + 3)
        rain[start_h:end_h] = np.random.uniform(0.8, 3.8, end_h - start_h)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Higher wet-season baseline with dramatic swelling during Tropical Storm Nando
base_water = 1.15  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    current_water = base_water + (rain[i] * 0.07) + (soil[i] * 0.002)
    
    # Introduce major flood surge during Tropical Storm Nando (around hours corresponding to Sept 22)
    if (21 * 24) <= i <= (24 * 24):
        current_water += 1.15  # Major surge reaching road/strawberry farm flood thresholds
    
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.02 + np.random.normal(0, 0.008)
    ultrasonic[i] = max(0.95, ultrasonic[i])
    
    # Flow velocity: Dangerous torrent during severe storms
    calculated_flow = 2.2 + (ultrasonic[i] - base_water) * 5.5 + (rain[i] * 0.45)
    flow[i] = np.clip(calculated_flow, 1.3, 7.0) + np.random.normal(0, 0.1)

# 6. Assemble into DataFrame for September
df_september = pd.DataFrame({
    'Timestamp_Sensor_Read': date_range,
    'ultrasonic': np.round(ultrasonic, 3),
    'rain': np.round(rain, 2),
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'soil': np.round(soil, 1),
    'flow': np.round(flow, 2)
})

# 7. Combine with existing data.csv and sort chronologically
if os.path.exists('data.csv'):
    existing_df = pd.read_csv('data.csv')
    combined_df = pd.concat([existing_df, df_september], ignore_index=True)
    
    # Ensure proper chronological sorting from September onwards
    combined_df['Timestamp_Sensor_Read'] = pd.to_datetime(combined_df['Timestamp_Sensor_Read'])
    combined_df = combined_df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)
    
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully added September 2025 data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_september.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with September 2025 data ({n} rows).")