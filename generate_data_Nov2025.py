import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for November 2025 (Transition to Dry Season)
date_range = pd.date_range(start='2025-11-01 00:00:00', end='2025-11-30 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad November Environmental Parameters:
# Temperature: Daytime highs ~24°C, overnight lows ~18°C
temperature = 21.0 + 3.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 16.5, 25.5)

# Humidity: Declining from monsoon saturation, averaging around 75% to 85%
humidity = 80.0 + 5.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 72.0, 88.0)

# Soil Moisture: Steadily depleting from muddy conditions into firmer terrain
soil = 58.0 + 4.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)
soil = np.clip(soil, 45.0, 70.0)

# 4. Simulate Rainfall: Low intensity, limited to occasional light afternoon showers or drizzle
rain = np.zeros(n)
for day in range(30):
    if np.random.rand() < 0.15:  # Light drizzle on roughly 15% of days
        peak_hour = day * 24 + 16
        start_h = max(0, peak_hour - 1)
        end_h = min(n, peak_hour + 2)
        rain[start_h:end_h] = np.random.uniform(0.05, 0.5, end_h - start_h)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Water levels drop back to safe, baseline ranges with a calmer, steady stream
base_water = 0.83  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    current_water = base_water + (rain[i] * 0.025)
    
    # Stable daily baseline cycles with minor variance
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.009 + np.random.normal(0, 0.003)
    ultrasonic[i] = max(0.78, ultrasonic[i])
    
    # Flow velocity transitions to a calmer, steady stream
    calculated_flow = 1.1 + (ultrasonic[i] - base_water) * 3.8 + (rain[i] * 0.25)
    flow[i] = np.clip(calculated_flow, 0.8, 2.2) + np.random.normal(0, 0.04)

# 6. Assemble into DataFrame for November
df_november = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_november], ignore_index=True)
    
    # Ensure proper chronological sorting from November onwards
    combined_df['Timestamp_Sensor_Read'] = pd.to_datetime(combined_df['Timestamp_Sensor_Read'])
    combined_df = combined_df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)
    
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully added November 2025 data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_november.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with November 2025 data ({n} rows).")