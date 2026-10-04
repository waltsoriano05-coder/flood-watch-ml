import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for December 2025 (Dry Season Baseline)
date_range = pd.date_range(start='2025-12-01 00:00:00', end='2025-12-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad December Environmental Parameters:
# Temperature: Daytime highs ~24°C, overnight lows ~14°C
temperature = 19.0 + 5.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 13.5, 24.5)

# Humidity: Hovering around an average of 84% due to mountainous terrain and morning fog
humidity = 84.0 + 4.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 78.0, 90.0)

# Soil Moisture: Stable and relatively low during the dry season months
soil = 42.0 + 2.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)
soil = np.clip(soil, 32.0, 50.0)

# 4. Simulate Rainfall: Very low / negligible (averaging ~2.44 inches for the month)
rain = np.zeros(n)
for day in range(31):
    # Occasional very brief light mist or localized dry-season drizzle on about 15% of days
    if np.random.rand() < 0.15:
        peak_hour = day * 24 + 15
        start_h = max(0, peak_hour - 1)
        end_h = min(n, peak_hour + 2)
        rain[start_h:end_h] = np.random.uniform(0.05, 0.4, end_h - start_h)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Stable, shallow, and well within normal baseline thresholds
base_water = 0.81  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    current_water = base_water + (rain[i] * 0.02)
    
    # Stable daily baseline cycles with minimal sensor noise
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.008 + np.random.normal(0, 0.003)
    ultrasonic[i] = max(0.77, ultrasonic[i]) # Safe baseline lower bound
    
    # Slow, steady flow dominated by base flow / wastewater discharge rather than storm runoff
    calculated_flow = 1.0 + (ultrasonic[i] - base_water) * 3.5 + (rain[i] * 0.2)
    flow[i] = np.clip(calculated_flow, 0.7, 2.0) + np.random.normal(0, 0.04)

# 6. Assemble into DataFrame for December
df_december = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_december], ignore_index=True)
    
    # Ensure proper chronological sorting from December onwards
    combined_df['Timestamp_Sensor_Read'] = pd.to_datetime(combined_df['Timestamp_Sensor_Read'])
    combined_df = combined_df.sort_values('Timestamp_Sensor_Read').reset_index(drop=True)
    
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully added December 2025 data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_december.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with December 2025 data ({n} rows).")