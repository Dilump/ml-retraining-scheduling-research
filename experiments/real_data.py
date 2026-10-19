
from sklearn.datasets import fetch_openml
import pandas as pd
from pathlib import Path

print("Downloading Electricity dataset from OpenML...")

# OpenML Dataset ID 151 = Electricity (ELEC2)
data = fetch_openml(
    data_id=151,
    as_frame=True,
    parser="auto"
)

# Extract features and target
df = data.data.copy()

# Convert target UP/DOWN to 1/0
df["target"] = (
    data.target.astype(str)
    .str.upper()
    .map({"UP": 1, "DOWN": 0})
    .to_numpy()
)

# Keep chronological order
# Do not shuffle the dataset

# Create datasets folder
Path("datasets").mkdir(exist_ok=True)

# Save to CSV
df.to_csv("datasets/electricity.csv", index=False)

print("\nDataset downloaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget distribution:")
print(df["target"].value_counts())
print("\nSaved to datasets/electricity.csv")
