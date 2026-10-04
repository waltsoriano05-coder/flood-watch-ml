import pandas as pd

# 1. Read your existing data.csv file
df = pd.read_csv('data.csv')

# 2. Convert the Timestamp column to actual datetime objects so Python understands chronological order
df['Timestamp_Sensor_Read'] = pd.to_datetime(df['Timestamp_Sensor_Read'])

# 3. Sort the dataframe by the timestamp in ascending order (May -> June -> July)
df = df.sort_values(by='Timestamp_Sensor_Read', ascending=True)

# 4. Save the sorted data back to data.csv
df.to_csv('data.csv', index=False)

print("Successfully sorted 'data.csv' from May to July!")