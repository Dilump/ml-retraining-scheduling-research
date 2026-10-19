
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"

input_file = DATA_DIR / "aws_spot_prices.csv"
output_file = DATA_DIR / "hourly_prices.csv"

# Load historical prices
df = pd.read_csv(input_file, parse_dates=["timestamp"])

print("Total raw records:", len(df))
print("\nAvailable zones:")
print(df["availability_zone"].value_counts())

# Select one Availability Zone
zone = "us-east-1a"

df = df[df["availability_zone"] == zone].copy()

if df.empty:
    raise ValueError(f"No records found for {zone}")

# Remove duplicates and sort by timestamp
df = (
    df.sort_values("timestamp")
      .drop_duplicates(subset=["timestamp"], keep="last")
)

# Create regular hourly time series
prices = (
    df.set_index("timestamp")["price"]
      .resample("1h")
      .last()
      .ffill()
      .dropna()
)

# Save the hourly price data
hourly_df = prices.reset_index()
hourly_df["availability_zone"] = zone

hourly_df.to_csv(output_file, index=False)

print("\nSelected zone:", zone)
print("Hourly observations:", len(hourly_df))
print("Date range:", hourly_df["timestamp"].min(),
      "to", hourly_df["timestamp"].max())

print("\nPrice statistics:")
print(hourly_df["price"].describe())

print("\nFirst 5 rows:")
print(hourly_df.head())

print("\nSaved:", output_file)
