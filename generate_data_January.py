import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for January 2026 (Peak Northeast Monsoon / Dry Season)
date_range = pd.date_range(start='2026-01-01 00:00:00', end='2026-01-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad January Environmental Parameters:
# Temperatures: Cool high-altitude Amihan season, averaging 13°C to 22°C
temperature = 17.5 + 4.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 12.5, 22.5)

# Humidity: Moderate moisture from early morning fog and mountain dew (75% to 85%)
humidity = 80.0 + 5.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 70.0, 88.0)

# Soil Moisture: Low or depleted along riverbanks due to weeks of minimal precipitation
soil = 45.0 + 3.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.5, n)
soil = np.clip(soil, 35.0, 55.0)

# 4. Simulate Rainfall: Extremely low to negligible (dry season)
# Mostly zeros with occasional very light morning mist/drizzle
rain = np.zeros(n)
for i in range(n):
    # 5% chance of a negligible trace drizzle (< 0.2 mm)
    if np.random.rand() < 0.05:
        rain[i] = np.random.uniform(0.0, 0.2)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Maintained low, stable baseline well below alert thresholds
base_water = 0.85  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    # Minimal runoff effect since rain is practically zero
    current_water = base_water + (rain[i] * 0.02)
    
    # Slight natural daily fluctuation and minor random noise
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.015 + np.random.normal(0, 0.003)
    ultrasonic[i] = max(0.80, ultrasonic[i]) # Keep realistic lower bound
    
    # Slow and calm water current volume
    calculated_flow = 1.0 + (ultrasonic[i] - base_water) * 4.0 + (rain[i] * 0.2)
    flow[i] = np.clip(calculated_flow, 0.6, 2.5) + np.random.normal(0, 0.05)

# 6. Assemble into DataFrame for January
df_january = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_january], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended January data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_january.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of January data.")