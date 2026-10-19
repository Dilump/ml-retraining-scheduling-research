
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retrain_sim as rs


class RetrainingDQNEnv(gym.Env):
    """
    DQN environment for cost-aware model retraining.

    Observation:
      [accuracy_drop, normalized_age, relative_price]

    Actions:
      0 = keep current model
      1 = retrain
    """

    metadata = {"render_modes": []}

    def __init__(self, price_history, reference_price,
                 mode="train", seed_base=1000):

        super().__init__()

        self.price_history = np.asarray(
            price_history, dtype=float
        )
        self.reference_price = float(reference_price)
        self.mode = mode
        self.seed_base = seed_base

        if self.reference_price <= 0:
            raise ValueError("Invalid reference price")

        if len(self.price_history) < rs.T + 1:
            raise ValueError("Insufficient price history")

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        self.episode_counter = 0

    def _observation(self):
        if self.t > rs.T:
            return np.zeros(3, dtype=np.float32)

        self.pre_accuracy = rs.accuracy(
            self.episode, self.w, self.t
        )

        self.drop = self.a_ref - self.pre_accuracy
        self.price = float(self.episode["price"][self.t])

        return np.array([
            self.drop,
            min(self.age / rs.T, 1.0),
            self.price / self.reference_price
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}

        if "scenario" in options:
            scenario = options["scenario"]
        else:
            scenario = rs.SCENARIOS[
                int(self.np_random.integers(
                    len(rs.SCENARIOS)
                ))
            ]

        episode_seed = options.get(
            "episode_seed",
            int(self.np_random.integers(1, 2**31))
        )

        self.episode = rs.make_episode(
            scenario, episode_seed
        )

        if "price_path" in options:
            price_path = np.asarray(
                options["price_path"], dtype=float
            )
        else:
            max_start = (
                len(self.price_history) - (rs.T + 1)
            )

            start = int(self.np_random.integers(
                0, max_start + 1
            ))

            price_path = self.price_history[
                start:start + rs.T + 1
            ]

        if len(price_path) != rs.T + 1:
            raise ValueError("Incorrect price path length")

        self.episode["price"] = price_path.copy()

        self.w = rs.model_at(self.episode, 0)
        self.a_ref = rs.accuracy(
            self.episode, self.w, 0
        )

        self.age = 0
        self.t = 1

        self.total_accuracy = 0.0
        self.total_cost = 0.0
        self.n_retrain = 0

        self.episode_counter += 1

        return self._observation(), {}

    def step(self, action):
        action = int(action)

        if action not in (0, 1):
            raise ValueError("Invalid action")

        cost = 0.0

        if action == 1:
            self.w = rs.model_at(
                self.episode, self.t
            )

            self.a_ref = rs.accuracy(
                self.episode, self.w, self.t
            )

            self.age = 0
            cost = self.price * rs.GPU_HOURS
            self.n_retrain += 1

        else:
            self.age += 1

        accuracy = rs.accuracy(
            self.episode, self.w, self.t
        )

        # Same centred reward used by tabular Q-learning
        reward = (
            rs.V_ACC * (accuracy - rs.A_MAX)
            - cost
        )

        self.total_accuracy += accuracy
        self.total_cost += cost

        self.t += 1

        terminated = self.t > rs.T
        truncated = False

        obs = self._observation()

        info = {}

        if terminated:
            info = {
                "utility": (
                    rs.V_ACC * self.total_accuracy
                    - self.total_cost
                ),
                "accuracy": (
                    self.total_accuracy / rs.T
                ),
                "cost": self.total_cost,
                "retrainings": self.n_retrain
            }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info
        )
