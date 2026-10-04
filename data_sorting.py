import pandas as pd
import os

# 1. Check if data.csv exists
if not os.path.exists('data.csv'):
    print("Error: 'data.csv' not found in the current directory.")
else:
    # 2. Load the dataset
    df = pd.read_csv('data.csv')
    
    # 3. Convert the timestamp column to actual datetime objects for accurate sorting
    # Note: Replace 'Timestamp_Sensor_Read' with your exact column name if it differs
    df['Timestamp_Sensor_Read'] = pd.to_datetime(df['Timestamp_Sensor_Read'])
    
    # 4. Sort the DataFrame chronologically (ascending order: Jan -> Feb -> Mar -> Apr -> May -> June -> July)
    df_sorted = df.sort_values(by='Timestamp_Sensor_Read', ascending=True)
    
    # Optional: Filter strictly between January 1 and July 31 if you have other months mixed in
    start_date = '2026-01-01 00:00:00'
    end_date = '2026-07-31 23:59:59'
    df_sorted = df_sorted[(df_sorted['Timestamp_Sensor_Read'] >= start_date) & (df_sorted['Timestamp_Sensor_Read'] <= end_date)]
    
    # 5. Reset index for a clean row arrangement
    df_sorted = df_sorted.reset_index(drop=True)
    
    # 6. Save the sorted data back to data.csv (or a new file like 'sorted_data.csv')
    output_filename = 'data.csv'
    df_sorted.to_csv(output_filename, index=False)
    
    print(f"Dataset successfully sorted from January to July!")
    print(f"Total rows in sorted file: {len(df_sorted)}")
    print(f"Date range: {df_sorted['Timestamp_Sensor_Read'].min()} to {df_sorted['Timestamp_Sensor_Read'].max()}")