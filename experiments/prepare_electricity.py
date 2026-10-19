
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"

df = pd.read_csv(DATA_DIR / "electricity.csv")

# Basic validation
assert len(df) == 45312
assert "target" in df.columns

# Chronological splits
n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

train.to_csv(DATA_DIR / "electricity_train.csv",
             index=False)
validation.to_csv(DATA_DIR / "electricity_val.csv",
                  index=False)
test.to_csv(DATA_DIR / "electricity_test.csv",
            index=False)

print("\n=== CHRONOLOGICAL SPLITS ===")
print("Training:", train.shape)
print("Validation:", validation.shape)
print("Testing:", test.shape)

print("\nTarget distributions:")
for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(name, part["target"].value_counts(
        normalize=True
    ).round(3).to_dict())

print("\nAll files saved!")
