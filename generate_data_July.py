import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for July 2026 (Peak monsoon / Habagat season)
date_range = pd.date_range(start='2026-07-01 00:00:00', end='2026-07-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad July Environmental Parameters:
# Temperatures: Cool and chilly, highs around 22°C to 25°C, lows around 16°C to 18°C
temperature = 20.0 + 3.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)

# Humidity: Near saturation point (85% to 95%), misty and damp
humidity = 90 + 4 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.2, n)
humidity = np.clip(humidity, 80, 99)

# Soil Moisture: Fully saturated / maximum capacity from prior weeks of rain
soil = 92 + 2 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.5, n)
soil = np.clip(soil, 80, 100)

# 4. Simulate Peak Monsoon Rain Events (Habagat downpours and multi-day typhoon weather)
rain = np.zeros(n)
for day in range(31):
    peak_hour = day * 24 + 14  # Afternoon peak
    daily_storm = np.random.uniform(1.0, 6.0) if day % 3 != 0 else np.random.uniform(0.5, 2.5)
    
    start_h = max(0, peak_hour - 3)
    end_h = min(n, peak_hour + 3)
    rain[start_h:end_h] = np.clip(np.random.normal(daily_storm, 0.4, end_h - start_h), 0, 10.0)

# Inject multi-day heavy storm/typhoon blocks typical of July
typhoon_start = 10 * 24
typhoon_end = 13 * 24
rain[typhoon_start:typhoon_end] += np.random.uniform(3.0, 9.0, typhoon_end - typhoon_start)
rain = np.clip(rain, 0, 14.0)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
ultrasonic = np.zeros(n)
flow = np.zeros(n)

base_water = 1.35  # Higher baseline due to sustained July river swelling
current_water = base_water

for i in range(n):
    runoff_effect = rain[i] * 0.18 + (soil[i] / 100) * 0.03
    current_water += runoff_effect - 0.02
    
    if rain[i] == 0:
        current_water = max(base_water, current_water - 0.025)
    
    current_water = min(current_water, 2.85)
    ultrasonic[i] = current_water + np.random.normal(0, 0.008)
    
    calculated_flow = 2.5 + (ultrasonic[i] - base_water) * 14.0 + (rain[i] * 0.9)
    flow[i] = np.clip(calculated_flow, 2.0, 14.0) + np.random.normal(0, 0.1)

# 6. Assemble into DataFrame for July
df_july = pd.DataFrame({
    'Timestamp_Sensor_Read': date_range,
    'ultrasonic': np.round(ultrasonic, 3),
    'rain': np.round(rain, 2),
    'temperature': np.round(temperature, 1),
    'humidity': np.round(humidity, 1),
    'soil': np.round(soil, 1),
    'flow': np.round(flow, 2)
})

# 7. Combine with existing data.csv (June) or create a new one
if os.path.exists('data.csv'):
    existing_df = pd.read_csv('data.csv')
    combined_df = pd.concat([existing_df, df_july], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended July data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_july.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of July data.")