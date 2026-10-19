
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "day", "period",
    "nswdemand", "vicdemand", "transfer"
]

BATCH_SIZE = 336

history = pd.concat([
    pd.read_csv(DATA / "electricity_train.csv"),
    pd.read_csv(DATA / "electricity_val.csv")
], ignore_index=True)

test = pd.read_csv(DATA / "electricity_test.csv")


def fit_model(data):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
    model.fit(data[FEATURES], data["target"])
    return model


def evaluate(window, interval):
    known = history.copy()
    model = fit_model(known.iloc[:-672])

    total_correct = 0
    total_seen = 0
    retrainings = 0

    for i, start in enumerate(range(0, len(test), BATCH_SIZE)):
        batch = test.iloc[start:start + BATCH_SIZE]

        # Predict before observing labels
        predictions = model.predict(batch[FEATURES])
        total_correct += np.sum(
            predictions == batch["target"].to_numpy()
        )
        total_seen += len(batch)

        # Batch labels become available afterward
        known = pd.concat([known, batch], ignore_index=True)

        # Retrain for future batches only
        if (i + 1) % interval == 0:
            model = fit_model(known.iloc[-window:])
            retrainings += 1

    return total_correct / total_seen, retrainings


WINDOWS = [1000, 3000, 5000, 10000, 20000]
INTERVALS = [3, 6, 12]

rows = []

for window in WINDOWS:
    for interval in INTERVALS:
        acc, n = evaluate(window, interval)
        rows.append({
            "window": window,
            "interval": interval,
            "accuracy": acc,
            "retrainings": n
        })

results = pd.DataFrame(rows)
print("\n=== RETRAINING WINDOW ABLATION ===")
print(results.sort_values("accuracy", ascending=False).to_string(index=False))

results.to_csv(
    OUT / "electricity_window_ablation.csv",
    index=False
)
