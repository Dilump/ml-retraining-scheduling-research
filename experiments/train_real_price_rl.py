
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

# ---------------------------------
# Configuration
# ---------------------------------

rs.GPU_HOURS = 8.0
rs.V_ACC = 20.0

N_TRAIN = 2000
N_TEST = 10
TRAINING_SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------
# Load AWS historical prices
# ---------------------------------

df = pd.read_csv(
    DATA, parse_dates=["timestamp"]
).sort_values("timestamp")

assert df["availability_zone"].nunique() == 1

prices = df["price"].to_numpy(dtype=float)

assert len(prices) >= 721
assert np.all(prices > 0)

# Same chronology as previous experiment
train_prices = prices[:241]
validation_prices = prices[241:442]
test_prices = prices[520:721]

# Obtain state bins ONLY from training prices
price_edges = np.quantile(
    train_prices, [1 / 3, 2 / 3]
)

print("Training price bins:", price_edges)

# ---------------------------------
# Generate episodes
# ---------------------------------

def make_training_episode(scenario, seed, rng):
    ep = rs.make_episode(scenario, seed)

    # Sample a 201-point window from
    # the training period only
    max_start = len(train_prices) - (rs.T + 1)
    start = int(rng.integers(0, max_start + 1))

    ep["price"] = train_prices[
        start:start + rs.T + 1
    ].copy()

    return ep


def make_test_episodes():
    episodes = []

    for j, scenario in enumerate(rs.SCENARIOS):
        for i in range(N_TEST):
            seed = 900000 + j * 1000 + i

            ep = rs.make_episode(scenario, seed)
            ep["price"] = test_prices.copy()

            episodes.append({
                "episode": ep,
                "scenario": scenario,
                "seed": seed
            })

    return episodes


test_items = make_test_episodes()

# ---------------------------------
# Train multiple Q-learning agents
# ---------------------------------

all_rows = []
trained_agents = []

for training_seed in TRAINING_SEEDS:

    print(
        f"\nTraining RL agent "
        f"with seed {training_seed}"
    )

    agent = rs.QAgent(seed=training_seed)

    rng = np.random.default_rng(training_seed)

    for i in range(N_TRAIN):

        agent.eps = max(
            0.05,
            1.0 - i / (0.6 * N_TRAIN)
        )

        scenario = rs.SCENARIOS[i % 3]

        ep = make_training_episode(
            scenario,
            1000 + i,
            rng
        )

        rs.run_episode(
            ep,
            agent.act,
            price_edges,
            learn=agent,
            r_off=rs.A_MAX
        )

        if (i + 1) % 500 == 0:
            print(
                f"  Completed {i + 1}/{N_TRAIN}"
            )

    trained_agents.append(agent)

    # ---------------------------------
    # Evaluate on held-out test period
    # ---------------------------------

    print("Evaluating trained agent...")

    for item in test_items:
        result = rs.run_episode(
            item["episode"],
            agent.greedy,
            price_edges
        )

        all_rows.append({
            "policy": "Q-learning",
            "training_seed": training_seed,
            "scenario": item["scenario"],
            "seed": item["seed"],
            "utility": result["utility"],
            "accuracy": result["mean_acc"],
            "cost": result["cost"],
            "retrainings": result["n_ret"]
        })

# ---------------------------------
# Results
# ---------------------------------

results = pd.DataFrame(all_rows)

summary = results.groupby(
    "training_seed"
).agg(
    utility=("utility", "mean"),
    accuracy=("accuracy", "mean"),
    cost=("cost", "mean"),
    retrainings=("retrainings", "mean")
).reset_index()

print("\n=== RL RESULTS BY TRAINING SEED ===")
print(summary.to_string(index=False))

print("\n=== OVERALL RL RESULTS ===")

print(
    summary[
        [
            "utility",
            "accuracy",
            "cost",
            "retrainings"
        ]
    ].mean()
)

# ---------------------------------
# Compare with tuned heuristic
# ---------------------------------

heuristic = pd.read_csv(
    OUTPUT / "tuned_real_price_results.csv"
)

print("\n=== PREVIOUS HEURISTIC RESULTS ===")
print(heuristic.to_string(index=False))

rl_utility = summary["utility"].mean()

threshold_utility = heuristic.loc[
    heuristic["policy"] == "Threshold",
    "utility"
].iloc[0]

price_utility = heuristic.loc[
    heuristic["policy"] == "Price-aware",
    "utility"
].iloc[0]

print("\n=== COMPARISON ===")

print(
    "RL mean utility:",
    round(rl_utility, 2)
)

print(
    "Standard threshold:",
    round(threshold_utility, 2)
)

print(
    "Price-aware threshold:",
    round(price_utility, 2)
)

print(
    "RL improvement over price-aware:",
    round(rl_utility - price_utility, 2)
)

# ---------------------------------
# Save outputs
# ---------------------------------

results.to_csv(
    OUTPUT / "real_price_rl_episodes.csv",
    index=False
)

summary.to_csv(
    OUTPUT / "real_price_rl_summary.csv",
    index=False
)

# Save trained Q-tables
for seed, agent in zip(
    TRAINING_SEEDS, trained_agents
):
    np.save(
        OUTPUT / f"real_price_qtable_seed{seed}.npy",
        agent.Q
    )

print("\nAll RL results saved successfully!")
