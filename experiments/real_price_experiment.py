
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrain_sim as rs

# Configuration
DATA_FILE = ROOT / "datasets" / "hourly_prices.csv"
GPU_HOURS = 8.0
N_TEST = 30

rs.GPU_HOURS = GPU_HOURS

# Load recorded AWS Spot prices
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

prices = df["price"].to_numpy(dtype=float)

if len(prices) < rs.T + 1:
    raise ValueError("Not enough historical observations")

# Generate an episode using actual AWS prices
def make_real_price_episode(scenario, seed, start):
    episode = rs.make_episode(scenario, seed)

    # Replace synthetic prices with recorded prices
    episode["price"] = prices[start:start + rs.T + 1].copy()

    return episode

# Create non-overlapping 200-step test episodes
episodes = []
scenario_names = rs.SCENARIOS

for i, scenario in enumerate(scenario_names):
    start = i * (rs.T + 1)

    episode = make_real_price_episode(
        scenario=scenario,
        seed=900000 + i,
        start=start
    )

    episodes.append(episode)

# Price-aware policy
def price_aware_policy(d, eta, reference_price):
    return lambda obs: int(
        obs["drop"] >
        d * (obs["price"] / reference_price) ** eta
    )

# Compare policies
reference_price = float(np.median(prices))

policies = {
    "Never": rs.never,
    "Fixed": rs.fixed_policy(30),
    "Threshold": rs.thresh_policy(0.05),
    "Price-aware": price_aware_policy(
        d=0.08,
        eta=1.0,
        reference_price=reference_price
    ),
}

# State price bins for Q-learning
price_edges = np.quantile(prices, [1/3, 2/3])

results = []

for name, policy in policies.items():
    outputs = rs.evaluate(policy, episodes, price_edges)

    results.append({
        "policy": name,
        "utility": np.mean([r["utility"] for r in outputs]),
        "accuracy": np.mean([r["mean_acc"] for r in outputs]),
        "cost": np.mean([r["cost"] for r in outputs]),
        "retrainings": np.mean([r["n_ret"] for r in outputs])
    })

results_df = pd.DataFrame(results)

print("\n=== HISTORICAL AWS PRICE EXPERIMENT ===")
print(results_df.to_string(index=False))

OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

results_df.to_csv(
    OUTPUT / "real_price_results.csv",
    index=False
)

print("\nResults saved successfully.")
