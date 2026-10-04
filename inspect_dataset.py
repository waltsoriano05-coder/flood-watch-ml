import pandas as pd

df = pd.read_csv("data.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nNumber of rows:")
print(len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nTimestamp information:")

df["Timestamp_Sensor_Read"] = pd.to_datetime(df["Timestamp_Sensor_Read"])

print("First timestamp:")
print(df["Timestamp_Sensor_Read"].iloc[0])

print("Second timestamp:")
print(df["Timestamp_Sensor_Read"].iloc[1])

print("\nTime difference between first two records:")
print(
    df["Timestamp_Sensor_Read"].iloc[1]
    - df["Timestamp_Sensor_Read"].iloc[0]
)

print("=" * 60)