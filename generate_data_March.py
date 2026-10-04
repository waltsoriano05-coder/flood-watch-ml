import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for March 2026 (Peak Dry Season)
date_range = pd.date_range(start='2026-03-01 00:00:00', end='2026-03-31 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad March Environmental Parameters:
# Temperatures: Daytime highs 23°C to 24°C, overnight lows ~14°C
temperature = 18.5 + 5.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 13.0, 24.5)

# Humidity: Averaging roughly 71% to 83%, peaking during overnight temperature drops
humidity = 77.0 + 6.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 68.0, 86.0)

# Soil Moisture: Highly depleted due to sustained high evaporation and minimal preceding rainfall
soil = 40.0 + 2.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.4, n)
soil = np.clip(soil, 30.0, 48.0)

# 4. Simulate Rainfall: Low to negligible, restricted to brief afternoon localized heat showers
rain = np.zeros(n)
for day in range(31):
    # Occasional brief afternoon heat showers on roughly 20% of the days
    if np.random.rand() < 0.20:
        peak_hour = day * 24 + 16  # Late afternoon (around 4 PM)
        start_h = max(0, peak_hour - 1)
        end_h = min(n, peak_hour + 2)
        rain[start_h:end_h] = np.random.uniform(0.1, 0.8, end_h - start_h)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Maintained at seasonal baseline lows
base_water = 0.82  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    # Minimal runoff effect from sparse afternoon heat showers
    current_water = base_water + (rain[i] * 0.03)
    
    # Slight daily fluctuations and sensor noise
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.01 + np.random.normal(0, 0.003)
    ultrasonic[i] = max(0.78, ultrasonic[i]) # Realistic lower bound
    
    # Slow to moderate flow driven heavily by domestic and commercial wastewater
    calculated_flow = 1.0 + (ultrasonic[i] - base_water) * 4.0 + (rain[i] * 0.3)
    flow[i] = np.clip(calculated_flow, 0.7, 2.5) + np.random.normal(0, 0.05)

# 6. Assemble into DataFrame for March
df_march = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_march], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended March data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_march.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of March data.")