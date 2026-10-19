"""Walk-forward Electricity evaluation, leakage-safe heuristics.
Place alongside electricity_rl_env.py inside experiments/.
Run from project root: python experiments/electricity_walkforward.py
Validation tunes parameters; evaluation is a later, non-overlapping period.
Exploratory only: dataset was used in prior development.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from electricity_rl_env import ElectricityRetrainingEnv, load_data, FEATURES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
train, val, test, prices = load_data(ROOT)
data = pd.concat([train, val, test], ignore_index=True)
# Sliding chronological evaluation windows, each preceded by a validation window.
# Disjoint evaluation windows of 5,000 rows; the final two include already explored test data.
folds = [(18000, 23000, 28000), (23000, 28000, 33000), (28000, 33000, 38000), (33000, 38000, len(data))]
intervals = [3, 6, 12, 20]
thresholds = [0.05, 0.10, 0.18, 0.25]
etas = [0.0, 0.5, 1.0, 2.0]

# A separate AWS price reference is computed from the early price segment only.
reference = float(np.median(prices[:241]))
# Fixed chronological validation and evaluation price windows; reused across folds.
# This is a price-replay sensitivity study, not independent AWS market periods.
p_val, p_test = prices[241:442], prices[520:721]


def evaluate(history, stream, price_path, name, cfg):
    env = ElectricityRetrainingEnv(history, stream, price_path, reference, window=1000)
    obs, _ = env.reset()
    while True:
        drop, age_scaled, price_ratio = map(float, obs)
        # age_scaled reflects age / 20 in this environment.
        if name == 'Never':
            action = 0
        elif name == 'Fixed':
            action = int(round(age_scaled * 20) >= cfg['k'])
        elif name == 'Threshold':
            action = int(drop > cfg['d'])
        else:
            action = int(drop > cfg['d'] * price_ratio ** cfg['eta'])
        obs, _, done, _, info = env.step(action)
        if done:
            return info


candidates = {
    'Never': [{}],
    'Fixed': [{'k': k} for k in intervals],
    'Threshold': [{'d': d} for d in thresholds],
    'Price-aware': [{'d': d, 'eta': eta} for d in thresholds for eta in etas],
}
rows, selections = [], []
for fold, (vstart, vend, tend) in enumerate(folds, 1):
    # Fit on data before validation, tune on validation, evaluate on later records.
    vhist, vstream = data.iloc[:vstart], data.iloc[vstart:vend]
    thist, tstream = data.iloc[:vend], data.iloc[vend:tend]
    print(f'Fold {fold}: validation [{vstart}:{vend}), eval [{vend}:{tend})', flush=True)
    for name, grid in candidates.items():
        scored = [(evaluate(vhist, vstream, p_val, name, cfg)['utility'], cfg) for cfg in grid]
        # Stable tie-break: first config in predeclared grid.
        best_idx = int(np.argmax([x[0] for x in scored]))
        vutil, cfg = scored[best_idx]
        metric = evaluate(thist, tstream, p_test, name, cfg)
        row = dict(fold=fold, policy=name, validation_utility=vutil,
                   utility=metric['utility'], accuracy=metric['accuracy'],
                   cost=metric['cost'], retrainings=metric['retrainings'],
                   evaluation_rows=len(tstream), configuration=str(cfg))
        rows.append(row)
        selections.append(dict(fold=fold, policy=name, config=str(cfg), validation_utility=vutil))
        print(f'  {name:12s} U={metric["utility"]:.3f} acc={metric["accuracy"]:.4f} updates={metric["retrainings"]}', flush=True)

result = pd.DataFrame(rows)
result.to_csv(OUT / 'electricity_walkforward_folds.csv', index=False)
pd.DataFrame(selections).to_csv(OUT / 'electricity_walkforward_choices.csv', index=False)
print('\n=== WALK-FORWARD POLICY SUMMARY ===')
print(result.groupby('policy')[['utility','accuracy','cost','retrainings']].mean().round(4).to_string())
print('\nNote: Different evaluation fold lengths; compare paired per-fold utility and report normalized utility if needed.')
print('Saved results/electricity_walkforward_folds.csv')
