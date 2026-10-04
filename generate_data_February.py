import pandas as pd
import numpy as np
import os

# 1. Generate hourly timestamps for February 2026 (Peak Dry Season / Cool Highland Conditions)
# Note: 2026 is not a leap year, so February has 28 days.
date_range = pd.date_range(start='2026-02-01 00:00:00', end='2026-02-28 23:00:00', freq='h')
n = len(date_range)

# 2. Setup randomness and time progression
np.random.seed(42)
time_index = np.arange(n)

# 3. Simulate La Trinidad February Environmental Parameters:
# Temperatures: Cool dry season, daytime highs ~23.3°C (74°F), overnight lows ~13.3°C (56°F)
temperature = 18.3 + 5.0 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.3, n)
temperature = np.clip(temperature, 12.5, 24.5)

# Humidity: Relatively high due to valley geography and morning mist, hovering around an average of 83%
humidity = 83.0 + 4.0 * np.sin(2 * np.pi * time_index / 24 + np.pi) + np.random.normal(0, 1.0, n)
humidity = np.clip(humidity, 73.0, 90.0)

# Soil Moisture: Dry to low due to the prolonged absence of sustained heavy downpours
soil = 42.0 + 2.5 * np.sin(2 * np.pi * time_index / 24) + np.random.normal(0, 0.5, n)
soil = np.clip(soil, 32.0, 52.0)

# 4. Simulate Rainfall: Very low to negligible (traditional dry season month)
rain = np.zeros(n)
for i in range(n):
    # Occasional very light trace morning mist or brief sparse drizzle (approx 4% chance)
    if np.random.rand() < 0.04:
        rain[i] = np.random.uniform(0.0, 0.15)

# 5. Simulate Balili River Height (Ultrasonic sensor in meters) & Flow Rate
# Maintained low, stable baseline well below alert thresholds
base_water = 0.83  
ultrasonic = np.zeros(n)
flow = np.zeros(n)

for i in range(n):
    # Negligible runoff effect from sparse dry-season conditions
    current_water = base_water + (rain[i] * 0.02)
    
    # Minimal daily fluctuations and minor random sensor noise
    ultrasonic[i] = current_water + np.sin(2 * np.pi * time_index[i] / 24) * 0.012 + np.random.normal(0, 0.003)
    ultrasonic[i] = max(0.78, ultrasonic[i]) # Keep realistic lower bound
    
    # Sluggish, low-velocity flow sustained primarily by urban wastewater and upstream creeks
    calculated_flow = 0.9 + (ultrasonic[i] - base_water) * 3.5 + (rain[i] * 0.15)
    flow[i] = np.clip(calculated_flow, 0.5, 2.0) + np.random.normal(0, 0.04)

# 6. Assemble into DataFrame for February
df_february = pd.DataFrame({
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
    combined_df = pd.concat([existing_df, df_february], ignore_index=True)
    combined_df.to_csv('data.csv', index=False)
    print(f"Successfully appended February data! Total rows in 'data.csv': {len(combined_df)}")
else:
    df_february.to_csv('data.csv', index=False)
    print(f"No existing file found. Created 'data.csv' with {n} rows of February data.")