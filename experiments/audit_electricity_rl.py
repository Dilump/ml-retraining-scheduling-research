"""Audit saved Electricity DQN models against the original evaluation protocol.
Run from project root: python experiments/audit_electricity_rl.py
Requires experiments/electricity_rl_env.py, results/electricity_dqn_seed*.zip,
results/electricity_rl_final_episodes.csv and local datasets.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'experiments'))
from electricity_rl_env import ElectricityRetrainingEnv, load_data

OUT = ROOT / 'results'
train, val, test, prices = load_data(ROOT)
reference = float(np.median(prices[:241]))
val_prices = prices[241:442]
test_prices = prices[520:721]
test_history = pd.concat([train, val], ignore_index=True)
assert len(test_prices) == 201
assert len(val_prices) == 201


def rollout(policy, history, stream, ppath):
    env = ElectricityRetrainingEnv(history, stream, ppath, reference, window=1000)
    obs, _ = env.reset()
    while True:
        action = int(policy(obs, env))
        obs, _, done, _, info = env.step(action)
        if done:
            return info, pd.DataFrame(env.records)

base, base_records = rollout(lambda obs, e: 0, test_history, test, test_prices)
rows = []
batches = []
for seed in range(5):
    path = OUT / f'electricity_dqn_seed{seed}.zip'
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved model: {path}. Re-run train_electricity_rl.py to save models.')
    model = DQN.load(str(path), device='cpu')
    decision = lambda obs, e, m=model: int(m.predict(obs, deterministic=True)[0])
    outcome, rec = rollout(decision, test_history, test, test_prices)
    cost_check = np.isclose(rec['cost'].sum(), outcome['cost'], atol=1e-8)
    count_check = int(rec['retrained'].sum()) == outcome['retrainings']
    utility_check = np.isclose(20.0 * rec['accuracy'].sum() - rec['cost'].sum(), outcome['utility'], atol=1e-7)
    if not all((cost_check, count_check, utility_check)):
        raise AssertionError(f'Internal accounting mismatch for seed {seed}')
    rec['seed'] = seed
    rec['never_batch_accuracy'] = base_records['accuracy'].values
    rec['accuracy_gain'] = rec['accuracy'] - rec['never_batch_accuracy']
    rec['utility_gain'] = 20.0 * rec['accuracy_gain'] - rec['cost']
    rec['cumulative_utility_gain'] = rec['utility_gain'].cumsum()
    batches.append(rec)
    rows.append({'seed': seed, **outcome,
                 'difference_vs_never': outcome['utility']-base['utility'],
                 'accuracy_difference_vs_never': outcome['accuracy']-base['accuracy'],
                 'sum_batch_utility_differences': rec['utility_gain'].sum(),
                 'number_of_retraining_steps': int(rec['retrained'].sum())})
summary = pd.DataFrame(rows)
batch_data = pd.concat(batches, ignore_index=True)
summary.to_csv(OUT / 'electricity_dqn_seed_audit.csv', index=False)
batch_data.to_csv(OUT / 'electricity_dqn_batch_audit.csv', index=False)
print('\n=== AUDITED DQN BY SEED ===')
print(summary[['seed','utility','accuracy','cost','retrainings','difference_vs_never']].to_string(index=False))
print('\nNever utility:', round(base['utility'],6))
print('DQN mean gain:', round(summary['difference_vs_never'].mean(),6))
print('DQN gain range:', (round(summary['difference_vs_never'].min(),6),round(summary['difference_vs_never'].max(),6)))
print('Seeds above Never:', int((summary['difference_vs_never']>0).sum()),'/5')
print('Retraining steps by seed:')
for seed, g in batch_data.groupby('seed'):
    print(seed, g.loc[g.retrained==1,'batch'].tolist())
# Verify each seed can be reconstructed using saved model and original run output.
original_path = OUT/'electricity_rl_final_episodes.csv'
if original_path.exists():
    original = pd.read_csv(original_path)
    for _, row in summary.iterrows():
        prior = original[(original.policy=='DQN') & (original.seed==row.seed)]
        if len(prior)!=1:
            raise AssertionError(f'Expected one original DQN row for seed {row.seed}; found {len(prior)}')
        for k in ('utility','accuracy','cost','retrainings'):
            if not np.isclose(float(prior.iloc[0][k]), float(row[k]), atol=1e-6):
                raise AssertionError(f'Mismatch seed={row.seed}, column={k}; original={prior.iloc[0][k]}, audit={row[k]}')
    print('\nPASS: saved models reproduce DQN rows in original CSV.')
else:
    print('\nNOTE: original CSV unavailable, model-vs-reported-results verification skipped.')
print('PASS: accounting checks (per-batch cost, retraining events and utility).')
print('NOTE: same test stream is reused for all five seeds; do not use five seeds as five independent drift datasets.')
print('NOTE: final test period already explored; results are exploratory, not a fresh confirmatory test.')
print('\nSaved:', OUT/'electricity_dqn_seed_audit.csv', 'and',OUT/'electricity_dqn_batch_audit.csv')
