
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

# ---------------- Configuration ----------------

BATCH_SIZE = 336
RETRAIN_WINDOW = 5000
CALIBRATION_SIZE = 672

GPU_HOURS = 8.0
V_ACC = 20.0

FEATURES = [
    "day", "period",
    "nswdemand", "vicdemand", "transfer"
]

THRESHOLDS = [
    0.01, 0.02, 0.03, 0.05,
    0.08, 0.12, 0.18, 0.25
]
ETAS = [0.0, 0.5, 1.0, 1.5, 2.0]
INTERVALS = [2, 3, 5, 8, 12, 20, 30, 50]

# ---------------- Load data ----------------

train = pd.read_csv(DATA / "electricity_train.csv")
val = pd.read_csv(DATA / "electricity_val.csv")
test = pd.read_csv(DATA / "electricity_test.csv")

price_df = pd.read_csv(
    DATA / "hourly_prices.csv",
    parse_dates=["timestamp"]
).sort_values("timestamp")

assert price_df["availability_zone"].nunique() == 1

prices = price_df["price"].to_numpy()

assert len(prices) >= 721

train_prices = prices[:241]
val_prices = prices[241:442]
test_prices = prices[520:721]

REFERENCE_PRICE = float(np.median(train_prices))

# ---------------- Model ----------------

def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )


def fit_model(data):
    model = build_model()
    model.fit(
        data[FEATURES],
        data["target"]
    )
    return model


def accuracy(model, batch):
    prediction = model.predict(batch[FEATURES])
    return float(np.mean(
        prediction == batch["target"].to_numpy()
    ))


def make_batches(data):
    return [
        data.iloc[i:i + BATCH_SIZE]
        for i in range(0, len(data), BATCH_SIZE)
    ]


# ---------------- Policies ----------------

def decide(policy, age, drop, price, cfg):
    if policy == "Never":
        return False

    if policy == "Fixed":
        return age >= cfg["k"]

    if policy == "Threshold":
        return drop > cfg["d"]

    if policy == "Price-aware":
        dynamic_d = cfg["d"] * (
            price / REFERENCE_PRICE
        ) ** cfg["eta"]
        return drop > dynamic_d

    raise ValueError("Unknown policy")


# ---------------- Replay experiment ----------------

def evaluate(policy, config, history, stream, price_path):
    """
    Prequential evaluation:
      1. Predict current batch
      2. Observe true labels and score
      3. Decide whether to retrain
      4. Retrain using data available so far
      5. Serve the next batch

    Price mapping is a historical replay, not a
    time-aligned deployment.
    """

    batches = make_batches(stream)

    # Hold out the most recent history for calibration
    fitting_history = history.iloc[:-CALIBRATION_SIZE]
    calibration = history.iloc[-CALIBRATION_SIZE:]

    model = fit_model(fitting_history)

    # Estimate initial reference accuracy out of sample
    reference_accuracy = accuracy(model, calibration)

    # Labels from calibration are now available
    known = history.copy()

    # Each batch receives a chronological price sample.
    # Each retraining job costs GPU_HOURS * starting price.
    price_indices = np.linspace(
        0, len(price_path) - 1,
        len(batches),
        dtype=int
    )

    total_cost = 0.0
    total_value = 0.0
    total_correct = 0
    total_samples = 0
    retrainings = 0
    age = 0
    records = []

    for i, batch in enumerate(batches):
        price = float(price_path[price_indices[i]])

        # Predictions happen BEFORE batch labels
        # are included in the retraining history.
        acc = accuracy(model, batch)
        correct = round(acc * len(batch))

        total_correct += correct
        total_samples += len(batch)

        # One accuracy value per batch:
        # same primary objective as earlier simulator.
        total_value += V_ACC * acc

        # Labels become available after prediction.
        drop = reference_accuracy - acc
        age += 1

        known = pd.concat(
            [known, batch],
            ignore_index=True
        )

        retrain = decide(
            policy, age, drop, price, config
        )

        cost = 0.0

        if retrain:
            recent = known.iloc[-RETRAIN_WINDOW:]
            model = fit_model(recent)

            cost = price * GPU_HOURS
            total_cost += cost
            retrainings += 1
            age = 0

            # Retain an externally established reference:
            # do not score the newly fitted model on
            # its own training examples.
            # Reference is fixed within this replay.

        records.append({
            "batch": i + 1,
            "accuracy": acc,
            "price": price,
            "drop": drop,
            "retrained": int(retrain),
            "cost": cost
        })

    utility = total_value - total_cost

    return {
        "utility": utility,
        "accuracy": total_correct / total_samples,
        "cost": total_cost,
        "retrainings": retrainings,
        "details": pd.DataFrame(records)
    }


# ---------------- Validation tuning ----------------

print("\nTuning real-data policies...")

configs = {}

fixed_scores = []
for k in INTERVALS:
    r = evaluate(
        "Fixed", {"k": k},
        train, val, val_prices
    )
    fixed_scores.append((r["utility"], k))

configs["Fixed"] = {
    "k": max(fixed_scores)[1]
}

threshold_scores = []
for d in THRESHOLDS:
    r = evaluate(
        "Threshold", {"d": d},
        train, val, val_prices
    )
    threshold_scores.append((r["utility"], d))

configs["Threshold"] = {
    "d": max(threshold_scores)[1]
}

price_scores = []
for d in THRESHOLDS:
    for eta in ETAS:
        cfg = {"d": d, "eta": eta}
        r = evaluate(
            "Price-aware", cfg,
            train, val, val_prices
        )
        price_scores.append(
            (r["utility"], d, eta)
        )

best = max(price_scores)
configs["Price-aware"] = {
    "d": best[1],
    "eta": best[2]
}

print("Selected configurations:")
print(configs)

# ---------------- Final held-out test ----------------

print("\nRunning real-world Electricity test...")

# Historical training/validation records are
# available when test time begins.
test_history = pd.concat(
    [train, val],
    ignore_index=True
)

policies = [
    "Never", "Fixed", "Threshold", "Price-aware"
]

summary_rows = []

for policy in policies:
    cfg = configs.get(policy, {})

    r = evaluate(
        policy, cfg,
        test_history, test, test_prices
    )

    summary_rows.append({
        "policy": policy,
        "utility": r["utility"],
        "accuracy": r["accuracy"],
        "cost": r["cost"],
        "retrainings": r["retrainings"]
    })

    r["details"].to_csv(
        OUTPUT / f"electricity_{policy.lower().replace('-', '_')}_batches.csv",
        index=False
    )

summary = pd.DataFrame(summary_rows)

print("\n=== REAL ELECTRICITY DATASET RESULTS ===")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT / "electricity_retraining_results.csv",
    index=False
)

print("\nResults saved successfully!")
