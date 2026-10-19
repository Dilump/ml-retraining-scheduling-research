
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrain_sim as rs
from cara_policy import CARATPolicy

DATA = ROOT / "datasets" / "hourly_prices.csv"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

rs.GPU_HOURS = 8.0
rs.V_ACC = 20.0

# Threshold candidates in normalized staleness units
TAUS = [
    0.001, 0.002, 0.005,
    0.01, 0.02, 0.03,
    0.05, 0.08, 0.12, 0.2
]

df = pd.read_csv(
    DATA,
    parse_dates=["timestamp"]
).sort_values("timestamp")

assert df["availability_zone"].nunique() == 1

prices = df["price"].to_numpy(dtype=float)
assert len(prices) >= 721

validation_prices = prices[241:442]
test_prices = prices[520:721]

price_edges = np.quantile(prices[:241], [1/3, 2/3])


def create_episodes(price_path, n, seed_base):
    items = []

    for j, scenario in enumerate(rs.SCENARIOS):
        for i in range(n):
            seed = seed_base + j * 1000 + i

            ep = rs.make_episode(scenario, seed)
            ep["price"] = price_path.copy()

            items.append((scenario, seed, ep))

    return items


validation = create_episodes(
    validation_prices, 5, 10000
)

test = create_episodes(
    test_prices, 10, 900000
)


def evaluate_cara(tau, episodes):
    rows = []

    for scenario, seed, ep in episodes:

        # New policy instance for every episode
        policy = CARATPolicy(
            ep,
            tau=tau,
            gamma=0.5
        )

        result = rs.run_episode(
            ep, policy, price_edges
        )

        rows.append({
            "scenario": scenario,
            "seed": seed,
            "utility": result["utility"],
            "accuracy": result["mean_acc"],
            "cost": result["cost"],
            "retrainings": result["n_ret"]
        })

    return pd.DataFrame(rows)


# --------------------------
# Validation grid search
# --------------------------

print("Tuning CARA-T...")

scores = []

for tau in TAUS:
    result = evaluate_cara(tau, validation)

    score = result["utility"].mean()
    scores.append({
        "tau": tau,
        "validation_utility": score
    })

    print(f"tau={tau:.3f} utility={score:.3f}")

grid = pd.DataFrame(scores)

# Break ties toward the smaller threshold
grid = grid.sort_values(
    ["validation_utility", "tau"],
    ascending=[False, True]
)

best_tau = float(grid.iloc[0]["tau"])

print("\nBest CARA-T threshold:", best_tau)

grid.to_csv(
    OUTPUT / "cara_grid_search.csv",
    index=False
)

# --------------------------
# Held-out evaluation
# --------------------------

results = evaluate_cara(best_tau, test)

summary = results[
    ["utility", "accuracy", "cost", "retrainings"]
].mean()

print("\n=== CARA-T RESULTS ===")
print(summary)

results.to_csv(
    OUTPUT / "cara_test_episodes.csv",
    index=False
)

# --------------------------
# Existing benchmarks
# --------------------------

heuristics = pd.read_csv(
    OUTPUT / "tuned_real_price_results.csv"
)

qlearning = pd.read_csv(
    OUTPUT / "real_price_rl_summary.csv"
)

dqn = pd.read_csv(
    OUTPUT / "real_price_dqn_summary.csv"
)

comparison = pd.DataFrame([
    {
        "policy": "Threshold",
        "utility": float(
            heuristics.loc[
                heuristics.policy == "Threshold",
                "utility"
            ].iloc[0]
        )
    },
    {
        "policy": "Price-aware",
        "utility": float(
            heuristics.loc[
                heuristics.policy == "Price-aware",
                "utility"
            ].iloc[0]
        )
    },
    {
        "policy": "Q-learning",
        "utility": qlearning.utility.mean()
    },
    {
        "policy": "DQN",
        "utility": dqn.utility.mean()
    },
    {
        "policy": "CARA-T sampled",
        "utility": summary["utility"]
    }
])

comparison.to_csv(
    OUTPUT / "cara_policy_comparison.csv",
    index=False
)

print("\n=== FINAL POLICY COMPARISON ===")
print(comparison.to_string(index=False))

print("\nCARA experiment completed!")
