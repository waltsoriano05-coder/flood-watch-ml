import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for April 2026 (Pre-monsoon transition / Late-month heavy rains)
date_range = pd.date_range(start='2026-04-01 00:00:00', end='2026-04-30 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad April Environmental Parameters:
# Temperatures: Daily mean ~20.5°C, highs peaking at ~25°C, lows dipping to ~16°C
temperature = 20.5 + 4.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 15.0, 26.5)

# Humidity: High relative humidity hovering around an average of 86%
humidity = 86.0 + 4.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 76.0, 95.0)

# Soil Moisture: Low to moderate early in the month, saturating during the late-April storms
soil = np.zeros(n)
for i in range(n):
    day = i // 24
    if day < 23:
        # Dry to moderate baseline during early/mid April
        soil[i] = 48.0 + 3.0 * np.sin(2 * np.pi * time_index[i] / 24) + np.random.normal(0, 0.5)
    else:
        # Rapidly saturating during the heavy storms in the last week
        soil[i] = 78.0 + 5.0 * np.sin(2 * np.pi * time_index[i] / 24) + np.random.normal(0, 0.8)
soil = np.clip(soil, 35.0, 92.0)

# 4. Simulate Rainfall: Low to negligible early on, with atypical "heavy rains" during the last week
rain = np.zeros(n)
for day in range(30):
    if day < 23:
        # Occasional light localized afternoon showers
        if np.random.rand() < 0.25:
            peak_hour = day * 24 + 16
            start_h = max(0, peak_hour - 1)
            end_h = min(n, peak_hour + 2)
            rain[start_h:end_h] = np.random.uniform(0.1, 1.2, end_h - start_h)
    else:
        # Heavy rain events and storms during the last week of April 2026
        storm_start = day * 24 + 13
        storm_end = min(n, storm_start + 8)
        rain[storm_start:storm_end] = np.random.uniform(2.0, 7.5, storm_end - storm_start)

rain = np.clip(rain, 0.0, 8.5)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
base_water = 0.88  # Normal low-to-moderate summer baseline
ultrasonic = np.zeros(n)
flow = np.zeros(n)
current_water = base_water

for i in range(n):
    # Runoff effect increases significantly once soil gets saturated in late April
    runoff_factor = 0.15 if (i // 24) >= 23 else 0.05
    runoff_effect = rain[i] * runoff_factor + (soil[i] / 100) * 0.01
    
    current_water += runoff_effect - 0.04
    if rain[i] == 0:
        current_water = max(base_water, current_water - 0.05)
    
    current_water = min(current_water, 2.15)  # Surge ceiling during late-month storms
    ultrasonic[i] = current_water + np.random.normal(0, 0.005)
    
    # Flow rate calculation reflecting the sudden surge and quick-flow runoff from Baguio tributaries
    calculated_flow = 1.1 + (ultrasonic[i] - base_water) * 9.0 + (rain[i] * 0.6)
    flow[i] = np.clip(calculated_flow, 0.8, 9.5) + np.random.normal(0, 0.08)

# 6. Assemble into DataFrame for April
df_april = pd.DataFrame({
    'Timestamp_Sensor_Read': date_range,
    'ultrasonic': np.round(ultrasonic, 3),
    'rain': np.round(rain, 2),
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'soil': np.round(soil, 1),
    'flow': np.round(flow, 2)
})

# 7. Combine with existing data.csv or create a new one
if os.path.exists('data.csv'):
    existing_df = pd.read_csv('data.csv')
    combined_df = pd.concat([existing_df, df_april], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended April data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_april.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of April data.")