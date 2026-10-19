
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from real_price_dqn_env import RetrainingDQNEnv

import retrain_sim as rs

rs.GPU_HOURS = 8.0
rs.V_ACC = 20.0

assert rs.GPU_HOURS == 8.0
assert rs.V_ACC == 20.0

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "datasets" / "hourly_prices.csv"
OUTPUT = ROOT / "results"
MODELS = OUTPUT / "dqn_models"

OUTPUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Configuration
TRAINING_SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 100000
N_TEST = 10

df = pd.read_csv(
    DATA, parse_dates=["timestamp"]
).sort_values("timestamp")

prices = df["price"].to_numpy(dtype=float)

assert len(prices) >= 721
assert df["availability_zone"].nunique() == 1

train_prices = prices[:241]
validation_prices = prices[241:442]
test_prices = prices[520:721]

REFERENCE_PRICE = float(np.median(train_prices))

print("Reference price:", REFERENCE_PRICE)

# Verify the environment first
check_environment = RetrainingDQNEnv(
    train_prices,
    REFERENCE_PRICE
)

check_env(check_environment)

print("Environment check passed!")

results = []

for training_seed in TRAINING_SEEDS:

    print(f"\n=== TRAINING SEED {training_seed} ===")

    env = RetrainingDQNEnv(
        price_history=train_prices,
        reference_price=REFERENCE_PRICE,
        seed_base=training_seed
    )

    env.reset(seed=training_seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.6,
        exploration_final_eps=0.05,
        policy_kwargs={
            "net_arch": [64, 64]
        },
        verbose=1,
        seed=training_seed,
        device="cpu"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=False
    )

    model.save(
        MODELS / f"dqn_seed_{training_seed}"
    )

    print("Training completed. Evaluating...")

    for j, scenario in enumerate(
        ["sudden", "gradual", "recurring"]
    ):
        for i in range(N_TEST):

            episode_seed = 900000 + j * 1000 + i

            test_env = RetrainingDQNEnv(
                price_history=test_prices,
                reference_price=REFERENCE_PRICE,
                mode="test"
            )

            obs, _ = test_env.reset(
                seed=episode_seed,
                options={
                    "scenario": scenario,
                    "episode_seed": episode_seed,
                    "price_path": test_prices
                }
            )

            done = False

            while not done:
                action, _ = model.predict(
                    obs, deterministic=True
                )

                obs, reward, terminated, truncated, info = (
                    test_env.step(action)
                )

                done = terminated or truncated

            results.append({
                "policy": "DQN",
                "training_seed": training_seed,
                "scenario": scenario,
                "seed": episode_seed,
                "utility": info["utility"],
                "accuracy": info["accuracy"],
                "cost": info["cost"],
                "retrainings": info["retrainings"]
            })

    env.close()

results = pd.DataFrame(results)

results.to_csv(
    OUTPUT / "real_price_dqn_episodes.csv",
    index=False
)

summary = results.groupby(
    "training_seed"
).agg(
    utility=("utility", "mean"),
    accuracy=("accuracy", "mean"),
    cost=("cost", "mean"),
    retrainings=("retrainings", "mean")
).reset_index()

summary.to_csv(
    OUTPUT / "real_price_dqn_summary.csv",
    index=False
)

print("\n=== DQN RESULTS BY TRAINING SEED ===")
print(summary.to_string(index=False))

print("\n=== OVERALL DQN RESULTS ===")
print(
    summary[
        ["utility", "accuracy", "cost", "retrainings"]
    ].mean()
)

# Comparison with previous experiments
heuristics = pd.read_csv(
    OUTPUT / "tuned_real_price_results.csv"
)

qlearning = pd.read_csv(
    OUTPUT / "real_price_rl_summary.csv"
)

comparison = pd.DataFrame([
    {
        "policy": "Threshold",
        "utility": heuristics.loc[
            heuristics["policy"] == "Threshold",
            "utility"
        ].iloc[0]
    },
    {
        "policy": "Price-aware",
        "utility": heuristics.loc[
            heuristics["policy"] == "Price-aware",
            "utility"
        ].iloc[0]
    },
    {
        "policy": "Q-learning",
        "utility": qlearning["utility"].mean()
    },
    {
        "policy": "DQN",
        "utility": summary["utility"].mean()
    }
])

comparison.to_csv(
    OUTPUT / "dqn_policy_comparison.csv",
    index=False
)

print("\n=== FINAL COMPARISON ===")
print(comparison.to_string(index=False))

print("\nAll results saved successfully!")
