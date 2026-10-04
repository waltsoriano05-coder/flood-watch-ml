import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for June 2026 (La Trinidad core wet season onset)
date_range = pd.date_range(start='2026-06-01 00:00:00', end='2026-06-30 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad June Environmental Parameters:
# Temperatures: Mild and stable due to high altitude (lows ~16°C, highs ~23°C)
temperature = 19.5 + 3.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)

# Humidity: Consistently heavy and muggy, averaging above 85-90%
humidity = 88 + 5 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.5, n)
humidity = np.clip(humidity, 75, 99)

# Soil Moisture: Highly saturated across agricultural valley floors and slopes
soil = 82 + 3 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.8, n)
soil = np.clip(soil, 65, 100)

# 4. Simulate Monsoon Rain Events (Frequent afternoon downpours and mid-month storm)
rain = np.zeros(n)
for day in range(30):
    peak_hour = day * 24 + 15  # 3:00 PM peak rain hour
    storm_intensity = np.random.uniform(0.5, 4.5) if day % 2 == 0 or day > 12 else np.random.uniform(0, 1.0)
    
    start_h = max(0, peak_hour - 2)
    end_h = min(n, peak_hour + 2)
    rain[start_h:end_h] = np.clip(np.random.normal(storm_intensity, 0.3, end_h - start_h), 0, 8.0)

# Inject a severe multi-day storm event around mid-June
mid_storm_start = 13 * 24
mid_storm_end = 16 * 24
rain[mid_storm_start:mid_storm_end] += np.random.uniform(2.0, 7.0, mid_storm_end - mid_storm_start)
rain = np.clip(rain, 0, 12.0)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
ultrasonic = np.zeros(n)
flow = np.zeros(n)

base_water = 1.20
current_water = base_water

for i in range(n):
    runoff_effect = rain[i] * 0.15 + (soil[i] / 100) * 0.02
    current_water += runoff_effect - 0.03
    
    if rain[i] == 0:
        current_water = max(base_water, current_water - 0.04)
    
    current_water = min(current_water, 2.60)
    ultrasonic[i] = current_water + np.random.normal(0, 0.008)
    
    calculated_flow = 2.0 + (ultrasonic[i] - base_water) * 12.5 + (rain[i] * 0.8)
    flow[i] = np.clip(calculated_flow, 1.5, 12.0) + np.random.normal(0, 0.1)

# 6. Assemble into DataFrame for June
df_june = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_june], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended June data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_june.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of June data.")