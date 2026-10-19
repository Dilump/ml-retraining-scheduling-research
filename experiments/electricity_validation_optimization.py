"""Validation-only tuning for Electricity retraining policy experiments.

Run from project root:
    python experiments/electricity_validation_optimization.py

Outputs are written to results/. This is an exploratory assessment because
an earlier test-set window ablation was already performed.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'datasets'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

FEATURES = ['day', 'period', 'nswdemand', 'vicdemand', 'transfer']
BATCH_SIZE = 336
CALIBRATION_SIZE = 672
V_ACC = 20.0
GPU_HOURS = 8.0  # hypothetical instance-hours, not measured training duration
WINDOWS = [1000, 3000, 5000, 10000, 20000]
INTERVALS = [3, 6, 12, 20]
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]
ETAS = [0.0, 0.5, 1.0, 2.0]

train = pd.read_csv(DATA / 'electricity_train.csv')
val = pd.read_csv(DATA / 'electricity_val.csv')
test = pd.read_csv(DATA / 'electricity_test.csv')
price_frame = pd.read_csv(DATA / 'hourly_prices.csv', parse_dates=['timestamp'])
price_frame = price_frame.sort_values('timestamp').reset_index(drop=True)
if price_frame['availability_zone'].nunique() != 1:
    raise ValueError('Use one Availability Zone only')
prices = price_frame['price'].to_numpy(float)
if len(prices) < 721 or not np.all(np.isfinite(prices)) or np.any(prices <= 0):
    raise ValueError('Need 721 positive hourly prices')
train_prices, val_prices, test_prices = prices[:241], prices[241:442], prices[520:721]
ref_price = float(np.median(train_prices))  # no test leakage

for part in [train, val, test]:
    missing = set(FEATURES + ['target']) - set(part.columns)
    if missing:
        raise ValueError(f'Missing columns: {missing}')


def fit(df):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(df[FEATURES], df['target'])
    return pipeline


def run_policy(name, window, config, past, stream, path):
    """Predict each batch before labels are available; update after scoring."""
    if len(past) <= CALIBRATION_SIZE:
        raise ValueError('Not enough initial historical records')
    model = fit(past.iloc[:-CALIBRATION_SIZE])
    cal = past.iloc[-CALIBRATION_SIZE:]
    reference = float(np.mean(model.predict(cal[FEATURES]) == cal['target'].to_numpy()))
    known = past.copy().reset_index(drop=True)
    batches = [stream.iloc[i:i+BATCH_SIZE] for i in range(0, len(stream), BATCH_SIZE)]
    # Prices are replayed as one hourly observation per data batch, spread over window.
    price_indices = np.linspace(0, len(path)-1, len(batches), dtype=int)
    acc_values = []
    sample_correct = sample_total = 0
    spend = 0.0
    age = 0
    n_retrain = 0
    details = []
    for idx, batch in enumerate(batches):
        pred = model.predict(batch[FEATURES])
        correct = int(np.sum(pred == batch['target'].to_numpy()))
        accuracy = correct / len(batch)
        sample_correct += correct
        sample_total += len(batch)
        acc_values.append(accuracy)
        age += 1
        price = float(path[price_indices[idx]])
        drop = reference - accuracy
        # labels only join historical data after batch prediction
        known = pd.concat([known, batch], ignore_index=True)
        if name == 'Never':
            trigger = False
        elif name == 'Fixed':
            trigger = age >= config['k']
        elif name == 'Threshold':
            trigger = drop > config['d']
        elif name == 'Price-aware':
            trigger = drop > config['d'] * (price/ref_price) ** config['eta']
        else:
            raise ValueError(name)
        cost = 0.0
        if trigger:
            model = fit(known.iloc[-window:])
            age = 0
            cost = price * GPU_HOURS
            spend += cost
            n_retrain += 1
        details.append(dict(batch=idx+1, accuracy=accuracy, drop=drop, price=price,
                            retrained=int(trigger), cost=cost, n_samples=len(batch)))
    return dict(utility=V_ACC*sum(acc_values)-spend,
                accuracy=sample_correct/sample_total, batch_accuracy=np.mean(acc_values),
                cost=spend, retrainings=n_retrain, details=pd.DataFrame(details))


# All selection/tuning below uses chronological validation only.
candidates = [('Never', 5000, {})]
for w in WINDOWS:
    candidates += [('Fixed', w, {'k':k}) for k in INTERVALS]
    candidates += [('Threshold', w, {'d':d}) for d in THRESHOLDS]
    candidates += [('Price-aware', w, {'d':d, 'eta':eta})
                   for d,eta in itertools.product(THRESHOLDS, ETAS)]

records = []
print(f'Tuning {len(candidates)} configurations on validation only...', flush=True)
for i, (name, window, config) in enumerate(candidates, 1):
    r = run_policy(name, window, config, train, val, val_prices)
    records.append(dict(policy=name, window=window, **config,
                        utility=r['utility'], accuracy=r['accuracy'],
                        batch_accuracy=r['batch_accuracy'],
                        cost=r['cost'], retrainings=r['retrainings']))
    if i % 50 == 0:
        print(f'  {i}/{len(candidates)} complete', flush=True)

search = pd.DataFrame(records)
search.to_csv(OUT / 'electricity_validation_grid.csv', index=False)

# Choose highest validation utility; deterministic ties prefer fewer retrains,
# then smaller number of settings and shorter window.
chosen = []
for name in ['Never', 'Fixed', 'Threshold', 'Price-aware']:
    pool = search.loc[search.policy == name].copy()
    pool = pool.sort_values(['utility','retrainings','window'], ascending=[False, True, True])
    best = pool.iloc[0]
    config = ({'k':int(best['k'])} if name == 'Fixed' else
              {'d':float(best['d'])} if name == 'Threshold' else
              {'d':float(best['d']), 'eta':float(best['eta'])} if name == 'Price-aware' else {})
    chosen.append((name, int(best['window']), config, float(best['utility'])))

print('\n=== VALIDATION-SELECTED CONFIGURATIONS ===')
for name, w, config, utility in chosen:
    print(f'{name:12s} window={w:5d} config={config} validation_utility={utility:.3f}')

# Preserve previous experiment's test initialization with all historical train+val data.
past_test = pd.concat([train, val], ignore_index=True)
result_rows = []
for name, window, config, val_utility in chosen:
    r = run_policy(name, window, config, past_test, test, test_prices)
    result_rows.append(dict(policy=name, window=window, config=str(config),
                            validation_utility=val_utility, utility=r['utility'],
                            accuracy=r['accuracy'], batch_accuracy=r['batch_accuracy'],
                            cost=r['cost'], retrainings=r['retrainings']))
    r['details'].to_csv(OUT / f'electricity_selected_{name.lower().replace("-", "_")}_batches.csv', index=False)

summary = pd.DataFrame(result_rows)
summary.to_csv(OUT / 'electricity_validation_selected_test.csv', index=False)
print('\n=== VALIDATION-SELECTED TEST RESULTS (EXPLORATORY) ===')
print(summary.to_string(index=False))
print('\nSaved CSV outputs under results/.')
