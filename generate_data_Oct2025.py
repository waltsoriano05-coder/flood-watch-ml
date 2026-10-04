import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for October 2025 (Monsoon Transition)
date_range = pd.date_range(start='2025-10-01 00:00:00', end='2025-10-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)
day_progress = time_index / n # Tracks progression across the month

# 3. Simulate La Trinidad October Environmental Parameters:
# Temperature: Typical valley averages between 15°C to 23°C
temperature = 19.0 + 4.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)
temperature = np.clip(temperature, 14.5, 23.5)

# Humidity: Consistently high, hovering between 80% and 90%
humidity = 85.0 + 4.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 78.0, 92.0)

# Soil Moisture: High/saturated early in the month, gradually drying out
soil = 78.0 - (12.0 * day_progress) + 3.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.5, n)
soil = np.clip(soil, 55.0, 88.0)

# 4. Simulate Rainfall: Moderate accumulation with brief, intense afternoon/evening downpours
rain = np.zeros(n)
for day in range(31):
    if np.random.rand() < 0.35:  # Rain on roughly 35% of days (tail-end monsoon frequency)
        # Afternoon/evening downpour peak between 15:00 and 19:00
        peak_hour = day * 24 + np.random.randint(15, 20)
        start_h = max(0, peak_hour - 1)
        end_h = min(n, peak_hour + 3)
        rain[start_h:end_h] = np.random.uniform(0.5, 3.5, end_h - start_h)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Receding from monsoon peaks but volatile and responsive to localized downpours
base_water = 0.95  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    # Water level responds dynamically to rain and soil saturation runoff
    current_water = base_water + (rain[i] * 0.05) + (soil[i] * 0.001)
    
    # Volatile spikes during storms, combined with daily cycles and noise
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.015 + np.random.normal(0, 0.005)
    ultrasonic[i] = max(0.85, ultrasonic[i])
    
    # Flow velocity: Steady to strong discharge funnel from upstream tributaries
    calculated_flow = 1.3 + (ultrasonic[i] - base_water) * 4.5 + (rain[i] * 0.4)
    flow[i] = np.clip(calculated_flow, 0.9, 3.2) + np.random.normal(0, 0.05)

# 6. Assemble into DataFrame for October
df_october = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_october], ignore_index=True)
    
    # Ensure proper chronological sorting
    combined_df['Timestamp_Sensor_Read'] = pd.to_datetime(combined_df['Timestamp_Sensor_Read'])
    combined_df = combined_df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)
    
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully added October 2025 data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_october.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with October 2025 data ({n} rows).")