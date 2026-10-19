"""Leakage-safe, prequential Electricity retraining environment.
Decision at batch t sees only accuracy on batches < t. No same-batch fitting.
Prices are independent historical AWS replay, not time-aligned market observations.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["day", "period", "nswdemand", "vicdemand", "transfer"]
BATCH_SIZE = 336
WINDOW = 1000
GPU_HOURS = 8.0
VALUE_PER_BATCH_ACCURACY = 20.0


def load_data(root=ROOT):
    d = Path(root) / "datasets"
    train = pd.read_csv(d / "electricity_train.csv")
    val = pd.read_csv(d / "electricity_val.csv")
    test = pd.read_csv(d / "electricity_test.csv")
    prices = pd.read_csv(d / "hourly_prices.csv", parse_dates=["timestamp"]).sort_values("timestamp")
    if prices["availability_zone"].nunique() != 1 or len(prices) < 721:
        raise ValueError("Require 721+ prices from one availability zone")
    for part in (train, val, test):
        if part[FEATURES + ["target"]].isna().any().any():
            raise ValueError("Missing features or labels")
    return train, val, test, prices["price"].to_numpy(dtype=float)


class ElectricityRetrainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, history, stream, price_path, reference_price, batch_size=BATCH_SIZE,
                 window=WINDOW, value=VALUE_PER_BATCH_ACCURACY, gpu_hours=GPU_HOURS):
        super().__init__()
        self.history = history.reset_index(drop=True)
        self.stream = stream.reset_index(drop=True)
        self.price_path = np.asarray(price_path, dtype=float)
        self.reference_price = float(reference_price)
        self.batch_size, self.window = int(batch_size), int(window)
        self.value, self.gpu_hours = float(value), float(gpu_hours)
        if len(self.history) < self.window + self.batch_size:
            raise ValueError("History too short; need window plus calibration batch")
        if len(self.stream) == 0 or len(self.price_path) < 2:
            raise ValueError("Need nonempty evaluation stream and prices")
        self.action_space = spaces.Discrete(2)
        # drop, model age (scaled), price relative to TRAINING reference
        self.observation_space = spaces.Box(
            low=np.array([-1., 0., 0.], dtype=np.float32),
            high=np.array([1., 100., 100.], dtype=np.float32), dtype=np.float32)
        self.batches = [self.stream.iloc[i:i+self.batch_size] for i in range(0,len(self.stream),self.batch_size)]
        self.mapped_prices = self.price_path[np.linspace(0,len(self.price_path)-1,len(self.batches),dtype=int)]

    def _fit(self):
        rows = pd.DataFrame(self.known[-self.window:], columns=FEATURES + ["target"])
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
        model.fit(rows[FEATURES], rows["target"].astype(int))
        return model

    def _obs(self):
        if self.t >= len(self.batches):
            return np.zeros(3, dtype=np.float32)
        return np.array([np.clip(self.reference_accuracy-self.last_accuracy,-1,1),
                         float(self.age)/20.0,
                         float(self.mapped_prices[self.t])/self.reference_price], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cols = FEATURES + ["target"]
        h = self.history[cols].to_numpy(dtype=float)
        self.known = list(h[:-self.batch_size])
        self.model = self._fit()
        calibration = self.history.iloc[-self.batch_size:]
        preds = self.model.predict(calibration[FEATURES])
        self.reference_accuracy = float(np.mean(preds == calibration["target"].to_numpy()))
        self.last_accuracy = self.reference_accuracy
        self.known.extend(list(h[-self.batch_size:]))
        self.t, self.age = 0, 0
        self.total_cost, self.total_value, self.total_correct, self.total_samples, self.n_ret = 0.,0.,0,0,0
        self.records = []
        return self._obs(), {}

    def step(self, action):
        if self.t >= len(self.batches):
            raise RuntimeError("Episode terminated; call reset")
        action = int(action)
        if action not in (0,1):
            raise ValueError("Action must be 0 or 1")
        price = float(self.mapped_prices[self.t])
        cost = 0.
        # Retrain from PRIOR labelled examples, before predicting current batch
        if action == 1:
            self.model = self._fit()
            cost = self.gpu_hours * price
            self.age = 0
            self.n_ret += 1
        batch = self.batches[self.t]
        pred = self.model.predict(batch[FEATURES])
        y = batch["target"].to_numpy(dtype=int)
        correct = int(np.sum(pred == y))
        acc = correct / len(batch)
        self.total_correct += correct
        self.total_samples += len(batch)
        self.total_value += self.value * acc
        self.total_cost += cost
        reward = self.value * (acc - 1.0) - cost
        self.records.append(dict(batch=self.t+1,accuracy=acc,price=price,retrained=action,cost=cost))
        # Labels released only AFTER model prediction
        self.known.extend(list(batch[FEATURES+["target"]].to_numpy(dtype=float)))
        self.last_accuracy = acc
        self.age += 1
        self.t += 1
        ended = self.t >= len(self.batches)
        info = {}
        if ended:
            info = dict(utility=self.total_value-self.total_cost,
                        accuracy=self.total_correct/self.total_samples,
                        batch_accuracy=self.total_value/(self.value*len(self.batches)),
                        cost=self.total_cost,retrainings=self.n_ret)
        return self._obs(), float(reward), ended, False, info
