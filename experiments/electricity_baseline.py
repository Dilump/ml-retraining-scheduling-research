
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

train = pd.read_csv(
    DATA_DIR / "electricity_train.csv"
)
test = pd.read_csv(
    DATA_DIR / "electricity_test.csv"
)

# Exclude contemporaneous price values to avoid
# relying on potentially unavailable information.
features = [
    "day", "period",
    "nswdemand", "vicdemand", "transfer"
]

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

# Fit using only earlier observations
model.fit(X_train, y_train)

# Test in chronological batches
BATCH_SIZE = 336

records = []

for start in range(0, len(test), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(test))

    X_batch = X_test.iloc[start:end]
    y_batch = y_test.iloc[start:end]

    predictions = model.predict(X_batch)

    accuracy = np.mean(
        predictions == y_batch.to_numpy()
    )

    records.append({
        "batch": len(records) + 1,
        "accuracy": accuracy
    })

results = pd.DataFrame(records)

results.to_csv(
    OUTPUT / "electricity_baseline.csv",
    index=False
)

print("\n=== REAL-DATA BASELINE ===")
print("Mean batch accuracy:",
      results["accuracy"].mean())
print("Minimum batch accuracy:",
      results["accuracy"].min())
print("Maximum batch accuracy:",
      results["accuracy"].max())

plt.figure(figsize=(11, 5))
plt.plot(
    results["batch"],
    results["accuracy"],
    marker="o",
    markersize=3
)

plt.title("Fixed Model Accuracy on Electricity Data")
plt.xlabel("Chronological Test Batch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "electricity_baseline.png",
    dpi=300
)

plt.show()
