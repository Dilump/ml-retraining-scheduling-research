"""Walk-forward comparison of heuristics, tabular Q-learning and DQN.
Place next to electricity_rl_env.py. Run from project root:
  python experiments/electricity_walkforward_rl.py
For a faster smoke test: WF_SEEDS=1 WF_STEPS=1500 WF_Q_EPISODES=30 python experiments/electricity_walkforward_rl.py
Exploratory: Electricity test blocks have already been inspected in prior analyses.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from electricity_rl_env import ElectricityRetrainingEnv, load_data, BATCH_SIZE
from gymnasium import Env

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
train, val, test, prices = load_data(ROOT)
data = pd.concat([train, val, test], ignore_index=True)
FOLDS = [(18000, 23000, 28000), (23000, 28000, 33000),
         (28000, 33000, 38000), (33000, 38000, len(data))]
REFERENCE = float(np.median(prices[:241]))
P_TRAIN = prices[:241]
P_VAL, P_TEST = prices[241:442], prices[520:721]
SEEDS = list(range(int(os.getenv('WF_SEEDS', '5'))))
DQN_STEPS = int(os.getenv('WF_STEPS', '12000'))
Q_EPISODES = int(os.getenv('WF_Q_EPISODES', '400'))
DQN_LRS = [1e-4, 3e-4]
CANDIDATES = {
    'Never': [{}],
    'Fixed': [{'k': k} for k in [3, 6, 12, 20]],
    'Threshold': [{'d': d} for d in [0.05, 0.1, 0.18, 0.25]],
    'Price-aware': [{'d': d, 'eta': eta} for d in [0.05, 0.1, 0.18, 0.25]
                    for eta in [0.0, 0.5, 1.0, 2.0]],
}

def env_for(history, stream, price_path):
    return ElectricityRetrainingEnv(history, stream, price_path, REFERENCE, window=1000)

def evaluate(policy, history, stream, price_path):
    env = env_for(history, stream, price_path)
    obs, _ = env.reset()
    while True:
        action = int(policy(obs, env))
        obs, _, done, _, info = env.step(action)
        if done:
            return info

def heuristic(name, cfg):
    def decide(obs, env):
        drop, age, price_ratio = map(float, obs)
        if name == 'Never': return 0
        if name == 'Fixed': return int(env.age >= cfg['k'])
        if name == 'Threshold': return int(drop > cfg['d'])
        return int(drop > cfg['d'] * price_ratio ** cfg['eta'])
    return decide

def make_sampler(prefix, seed):
    """Sample training segments entirely before this fold's validation boundary."""
    rng = np.random.default_rng(seed)
    segment = 12 * BATCH_SIZE
    if len(prefix) < 2000 + segment:
        raise ValueError('Not enough preceding training records')
    def sample():
        start = int(rng.integers(2000, len(prefix) - segment + 1))
        history = prefix.iloc[start-2000:start]
        stream = prefix.iloc[start:start+segment]
        p_start = int(rng.integers(0, len(P_TRAIN)-201+1))
        return env_for(history, stream, P_TRAIN[p_start:p_start+201])
    return sample

class EpisodeSampler(Env):
    metadata = {'render_modes': []}
    def __init__(self, sample):
        super().__init__()
        self.sample = sample
        self.current = sample()
        self.action_space = self.current.action_space
        self.observation_space = self.current.observation_space
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current = self.sample()
        return self.current.reset(seed=seed, options=options)
    def step(self, action):
        return self.current.step(action)

def state(obs):
    drop, age, price = map(float, obs)
    d = int(np.digitize(drop, [0.02, 0.05, 0.10, 0.20]))
    a = int(np.digitize(age, [0.25, 0.75, 1.5]))
    p = int(np.digitize(price, [0.97, 1.03]))
    return (d * 4 + a) * 3 + p

def train_q(prefix, seed):
    sample = make_sampler(prefix, seed + 10000)
    rng = np.random.default_rng(seed)
    Q = np.zeros((60, 2))
    for episode in range(Q_EPISODES):
        env = sample()
        obs, _ = env.reset()
        eps = max(0.05, 1.0 - episode / max(1, 0.6 * Q_EPISODES))
        while True:
            s = state(obs)
            action = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[s]))
            next_obs, reward, done, _, _ = env.step(action)
            target = reward if done else reward + 0.95 * float(np.max(Q[state(next_obs)]))
            Q[s, action] += 0.1 * (target - Q[s, action])
            obs = next_obs
            if done: break
    return lambda obs, env: int(np.argmax(Q[state(obs)]))

rows = []
for fold, (vstart, vend, tend) in enumerate(FOLDS, 1):
    prefix = data.iloc[:vstart]
    vh, vs = prefix, data.iloc[vstart:vend]
    eh, es = data.iloc[:vend], data.iloc[vend:tend]
    print(f'\nFold {fold}: train < {vstart}, validate [{vstart}:{vend}), test [{vend}:{tend})', flush=True)

    # All heuristic tuning happens on THIS fold's validation stream.
    for name, configs in CANDIDATES.items():
        scores = [evaluate(heuristic(name, cfg), vh, vs, P_VAL)['utility'] for cfg in configs]
        cfg = configs[int(np.argmax(scores))]
        result = evaluate(heuristic(name, cfg), eh, es, P_TEST)
        rows.append(dict(fold=fold, policy=name, seed=-1, validation_utility=max(scores),
                         config=str(cfg), evaluation_batches=int(np.ceil(len(es)/BATCH_SIZE)), **result))

    for seed in SEEDS:
        print(f'  RL seed {seed}', flush=True)
        q_policy = train_q(prefix, seed + 1000*fold)
        q_val = evaluate(q_policy, vh, vs, P_VAL)
        q_test = evaluate(q_policy, eh, es, P_TEST)
        rows.append(dict(fold=fold, policy='Q-learning', seed=seed,
                         validation_utility=q_val['utility'], config='tabular',
                         evaluation_batches=int(np.ceil(len(es)/BATCH_SIZE)), **q_test))

        # Train all DQN candidates only on the prefix; choose learning rate on validation.
        options = []
        for lr in DQN_LRS:
            sampler = EpisodeSampler(make_sampler(prefix, seed + 10000*fold + int(lr*1e7)))
            model = DQN('MlpPolicy', sampler, learning_rate=lr, buffer_size=10000,
                        learning_starts=250, batch_size=64, gamma=0.95,
                        train_freq=4, exploration_fraction=0.6,
                        exploration_final_eps=0.05, policy_kwargs={'net_arch':[64, 64]},
                        seed=seed, verbose=0, device='cpu')
            model.learn(total_timesteps=DQN_STEPS)
            policy = lambda obs, env, m=model: int(m.predict(obs, deterministic=True)[0])
            val_result = evaluate(policy, vh, vs, P_VAL)
            options.append((val_result['utility'], lr, model))
            sampler.close()
        best_val, best_lr, best_model = max(options, key=lambda x: x[0])
        policy = lambda obs, env, m=best_model: int(m.predict(obs, deterministic=True)[0])
        result = evaluate(policy, eh, es, P_TEST)
        rows.append(dict(fold=fold, policy='DQN', seed=seed,
                         validation_utility=best_val, config=f'lr={best_lr}',
                         evaluation_batches=int(np.ceil(len(es)/BATCH_SIZE)), **result))
        print(f'    DQN validation={best_val:.3f}, test={result["utility"]:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'electricity_walkforward_rl_folds.csv', index=False)

results = pd.DataFrame(rows)
# Normalize utility to account for last fold having a different number of batches.
results['utility_per_batch'] = results['utility'] / results['evaluation_batches']
results['cost_per_batch'] = results['cost'] / results['evaluation_batches']
results.to_csv(OUT / 'electricity_walkforward_rl_folds.csv', index=False)
print('\n=== WALK-FORWARD RL SUMMARY ===')
print(results.groupby('policy')[['utility_per_batch','accuracy','cost_per_batch','retrainings']].mean().round(4).to_string())
print('\nNOTE: Five seeds are not independent data folds. Compare per fold and report seed variability.')
print('Saved:', OUT / 'electricity_walkforward_rl_folds.csv')
