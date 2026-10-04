import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for May 2026 (Transition/Onset of rainy season)
date_range = pd.date_range(start='2026-05-01 00:00:00', end='2026-05-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad May Environmental Parameters:
# Temperatures: Warm transition period leading into the monsoon, slightly higher than June
temperature = 21.0 + 3.2 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)

# Humidity: Rising towards late May as pre-monsoon showers begin
humidity = 82 + 6 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.5, n)
humidity = np.clip(humidity, 70, 95)

# Soil Moisture: Moderately high, absorbing the early seasonal rains
soil = 75 + 4 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.8, n)
soil = np.clip(soil, 55, 90)

# 4. Simulate Early Rain Events (Scattered pre-monsoon afternoon showers, picking up late in the month)
rain = np.zeros(n)
for day in range(31):
    peak_hour = day * 24 + 16  # Late afternoon showers
    # Rain becomes more frequent towards the end of May
    storm_intensity = np.random.uniform(0.2, 3.0) if day > 15 else np.random.uniform(0.0, 1.5)
    
    start_h = max(0, peak_hour - 2)
    end_h = min(n, peak_hour + 2)
    rain[start_h:end_h] = np.clip(np.random.normal(storm_intensity, 0.3, end_h - start_h), 0, 6.0)

# Inject an early onset squall line near the final days of May
late_may_storm = 27 * 24
rain[late_may_storm:late_may_storm + 48] += np.random.uniform(1.5, 5.0, 48)
rain = np.clip(rain, 0, 9.0)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
ultrasonic = np.zeros(n)
flow = np.zeros(n)

base_water = 1.10  # Lower baseline in May before heavy monsoon peaks
current_water = base_water

for i in range(n):
    runoff_effect = rain[i] * 0.12 + (soil[i] / 100) * 0.015
    current_water += runoff_effect - 0.035
    
    if rain[i] == 0:
        current_water = max(base_water, current_water - 0.045)
    
    current_water = min(current_water, 2.30)
    ultrasonic[i] = current_water + np.random.normal(0, 0.008)
    
    calculated_flow = 1.8 + (ultrasonic[i] - base_water) * 11.0 + (rain[i] * 0.7)
    flow[i] = np.clip(calculated_flow, 1.2, 10.0) + np.random.normal(0, 0.1)

# 6. Assemble into DataFrame for May
df_may = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_may], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended May data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_may.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of May data.")