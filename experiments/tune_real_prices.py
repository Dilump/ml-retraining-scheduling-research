
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrain_sim as rs

DATA = ROOT / "datasets" / "hourly_prices.csv"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

# -----------------------------
# Experiment configuration
# -----------------------------

rs.GPU_HOURS = 8.0
rs.V_ACC = 20.0

THRESHOLDS = [
    0.01, 0.02, 0.03, 0.05,
    0.08, 0.12, 0.18, 0.25
]

ETAS = [0.0, 0.5, 1.0, 1.5, 2.0]

INTERVALS = [2, 3, 5, 8, 12, 20, 30, 50]

VALIDATION_SEEDS = 5
TEST_SEEDS = 10

# -----------------------------
# Load real AWS prices
# -----------------------------

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

assert df["availability_zone"].nunique() == 1

prices = df["price"].to_numpy(dtype=float)

assert len(prices) >= 721
assert np.all(prices > 0)

# 201 observations per episode:
# index 0 is initialization, then 200 decisions
validation_prices = prices[241:442]
test_prices = prices[520:721]

assert len(validation_prices) == rs.T + 1
assert len(test_prices) == rs.T + 1

# Reference price derived from validation only
REFERENCE_PRICE = float(np.median(validation_prices))

print("Reference price:", REFERENCE_PRICE)
print("Validation mean:", validation_prices.mean())
print("Test mean:", test_prices.mean())

# -----------------------------
# Episode generation
# -----------------------------

def create_episodes(price_path, n_seeds, seed_base):
    episodes = []

    for j, scenario in enumerate(rs.SCENARIOS):
        for i in range(n_seeds):

            seed = seed_base + j * 1000 + i

            ep = rs.make_episode(scenario, seed)

            # Replace synthetic prices
            ep["price"] = price_path.copy()

            ep["scenario"] = scenario
            ep["seed"] = seed

            episodes.append(ep)

    return episodes


validation = create_episodes(
    validation_prices, VALIDATION_SEEDS, 10000
)

test = create_episodes(
    test_prices, TEST_SEEDS, 900000
)

# -----------------------------
# Scheduling policies
# -----------------------------

def price_policy(d, eta):
    def decide(obs):

        dynamic_threshold = (
            d * (obs["price"] / REFERENCE_PRICE) ** eta
        )

        return int(obs["drop"] > dynamic_threshold)

    return decide


def mean_utility(policy, episodes):

    results = rs.evaluate(
        policy,
        episodes,
        pe=[0.27, 0.29]
    )

    return np.mean([
        r["utility"] for r in results
    ])

# -----------------------------
# Tune fixed-interval policy
# -----------------------------

print("\nTuning fixed-interval policy...")

fixed_scores = []

for k in INTERVALS:

    score = mean_utility(
        rs.fixed_policy(k),
        validation
    )

    fixed_scores.append((score, k))

best_fixed = max(fixed_scores)[1]

print("Best interval:", best_fixed)

# -----------------------------
# Tune price-blind threshold
# -----------------------------

print("\nTuning standard threshold...")

threshold_scores = []

for d in THRESHOLDS:

    score = mean_utility(
        rs.thresh_policy(d),
        validation
    )

    threshold_scores.append((score, d))

best_threshold = max(threshold_scores)[1]

print("Best threshold:", best_threshold)

# -----------------------------
# Tune price-aware threshold
# -----------------------------

print("\nTuning price-aware threshold...")

price_scores = []

for d in THRESHOLDS:
    for eta in ETAS:

        score = mean_utility(
            price_policy(d, eta),
            validation
        )

        price_scores.append((score, d, eta))

best_score, best_d, best_eta = max(price_scores)

print("Best price-aware d:", best_d)
print("Best price-aware eta:", best_eta)
print("Validation utility:", best_score)

# -----------------------------
# Final held-out evaluation
# -----------------------------

print("\nRunning held-out test...")

policies = {
    "Never": rs.never,
    "Fixed": rs.fixed_policy(best_fixed),
    "Threshold": rs.thresh_policy(best_threshold),
    "Price-aware": price_policy(best_d, best_eta)
}

rows = []
episode_rows = []

for name, policy in policies.items():

    outputs = rs.evaluate(
    policy,
    test,
    pe=[0.27, 0.29]
)

    for ep, result in zip(test, outputs):
        episode_rows.append({
            "policy": name,
            "scenario": ep["scenario"],
            "seed": ep["seed"],
            "utility": result["utility"],
            "accuracy": result["mean_acc"],
            "cost": result["cost"],
            "retrainings": result["n_ret"]
        })

    rows.append({
        "policy": name,
        "utility": np.mean([
            r["utility"] for r in outputs
        ]),
        "accuracy": np.mean([
            r["mean_acc"] for r in outputs
        ]),
        "cost": np.mean([
            r["cost"] for r in outputs
        ]),
        "retrainings": np.mean([
            r["n_ret"] for r in outputs
        ])
    })

results = pd.DataFrame(rows)

print("\n=== TUNED REAL AWS PRICE RESULTS ===")
print(results.to_string(index=False))

# -----------------------------
# Save results
# -----------------------------

results.to_csv(
    OUTPUT / "tuned_real_price_results.csv",
    index=False
)

pd.DataFrame(episode_rows).to_csv(
    OUTPUT / "tuned_real_price_episodes.csv",
    index=False
)

pd.DataFrame(
    price_scores,
    columns=["validation_utility", "d", "eta"]
).to_csv(
    OUTPUT / "price_aware_grid_search.csv",
    index=False
)

print("\nAll results saved successfully!")
