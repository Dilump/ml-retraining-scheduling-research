"""
Cost-aware retraining scheduler: simulation + tabular Q-learning vs baselines.
Everything is reproducible from the seeds below.
"""
import numpy as np, json, time, sys
from scipy.stats import wilcoxon

# ---------------- configuration ----------------
T = 200            # decision steps per episode
N = 500            # samples per batch
MU, SIGMA = 1.5, 1.0
BASE_PRICE = 3.0   # $/GPU-hour (ASSUMPTION, replace with verified price)
GPU_HOURS = 2.0    # GPU-hours per retraining job (ASSUMPTION)
V_ACC = 20.0       # $ value of one unit of accuracy per step (ASSUMPTION)
A_MAX = 0.9332     # Phi(MU/SIGMA): best attainable accuracy, used to centre the learning reward
PRICE_SD = 0.25
SCENARIOS = ["sudden", "gradual", "recurring"]

# ---------------- environment ----------------
def make_episode(scenario, seed, price_mult=1.0):
    rng = np.random.default_rng(seed)
    th = np.zeros(T + 1)
    if scenario == "sudden":
        t = 0
        cur = rng.uniform(0, np.pi)
        while t <= T:
            seg = int(rng.integers(25, 60))
            th[t:t + seg] = cur
            cur += rng.choice([-1, 1]) * rng.uniform(np.radians(50), np.radians(90))
            t += seg
    elif scenario == "gradual":
        w = np.radians(rng.uniform(0.8, 1.6))
        th = rng.uniform(0, np.pi) + w * np.arange(T + 1)
    else:  # recurring
        per = int(rng.integers(20, 35))
        a = rng.uniform(0, np.pi)
        d = np.radians(rng.uniform(55, 85))
        th = a + d * (((np.arange(T + 1) // per) % 2).astype(float))
    U = np.stack([np.cos(th), np.sin(th)], -1)[:, None, :]            # (T+1,1,2)
    def gen():
        Y = rng.choice([-1.0, 1.0], size=(T + 1, N))
        X = Y[..., None] * MU * U + SIGMA * rng.standard_normal((T + 1, N, 2))
        return X, Y
    X, Y = gen()          # serving batches
    Xr, Yr = gen()        # labelled batches available for retraining
    # AR(1) log-price process
    x = np.zeros(T + 1)
    for t in range(1, T + 1):
        x[t] = 0.9 * x[t - 1] + PRICE_SD * rng.standard_normal()
    price = BASE_PRICE * price_mult * np.exp(x - 0.5 * np.var(x))
    return dict(X=X, Y=Y, Xr=Xr, Yr=Yr, price=price, cache={})

def fit_lr(X, Y, iters=8, l2=1e-3):
    A = np.hstack([X, np.ones((len(X), 1))])
    y01 = (Y > 0).astype(float)
    w = np.zeros(3)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y01) / len(A) + l2 * w
        H = (A * (p * (1 - p))[:, None]).T @ A / len(A) + l2 * np.eye(3)
        w -= np.linalg.solve(H, g)
    return w

def model_at(ep, m):
    if m not in ep["cache"]:
        ep["cache"][m] = fit_lr(ep["Xr"][m], ep["Yr"][m])
    return ep["cache"][m]

def accuracy(ep, w, t):
    s = ep["X"][t] @ w[:2] + w[2]
    return float(np.mean(np.sign(s) == ep["Y"][t]))

# ---------------- policies ----------------
DROP_EDGES = [0.02, 0.05, 0.10, 0.20]
AGE_EDGES = [5, 15, 30]

def state_of(drop, age, price, pr_edges):
    return (int(np.digitize(drop, DROP_EDGES)) * 4 + int(np.digitize(age, AGE_EDGES))) * 3 \
        + int(np.digitize(price, pr_edges))
N_STATES = 5 * 4 * 3

def run_episode(ep, policy, price_edges, learn=None, r_off=0.0):
    """policy(obs)->0/1 ; obs=dict(drop,age,price,t,state). Returns metrics."""
    w = model_at(ep, 0)
    a_ref = accuracy(ep, w, 0)
    age = 0
    tot_acc = tot_cost = 0.0
    n_ret = 0
    accs, rets = [], []
    prev = None
    for t in range(1, T + 1):
        a_pre = accuracy(ep, w, t)
        price = ep["price"][t]
        drop = a_ref - a_pre
        s = state_of(drop, age, price, price_edges)
        if learn is not None and prev is not None:
            learn.update(prev[0], prev[1], prev[2], s)
        act = policy(dict(drop=drop, age=age, price=price, t=t, state=s))
        if act == 1:
            w = model_at(ep, t)
            a_ref = accuracy(ep, w, t)
            age = 0
            cost = price * GPU_HOURS
            n_ret += 1
        else:
            cost = 0.0
            age += 1
        a = accuracy(ep, w, t)
        r = V_ACC * (a - r_off) - cost   # r_off = 0 -> raw reward; A_MAX -> centred reward
        if learn is not None:
            prev = (s, act, r)
        tot_acc += a; tot_cost += cost
        accs.append(a); rets.append(act)
    if learn is not None and prev is not None:
        learn.update(prev[0], prev[1], prev[2], None)
    return dict(mean_acc=tot_acc / T, cost=tot_cost, n_ret=n_ret,
                utility=V_ACC * tot_acc - tot_cost, accs=accs, rets=rets)

class QAgent:
    def __init__(self, seed=0, alpha=0.1, gamma=0.95):
        self.Q = np.zeros((N_STATES, 2)); self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma, self.eps = alpha, gamma, 1.0
    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(2))
        q = self.Q[obs["state"]]
        return int(np.argmax(q)) if q[0] != q[1] else int(self.rng.integers(2))
    def update(self, s, a, r, s2):
        tgt = r if s2 is None else r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (tgt - self.Q[s, a])
    def greedy(self, obs):
        q = self.Q[obs["state"]]
        return int(q[1] > q[0])

def fixed_policy(k):  return lambda o: int(o["age"] + 1 >= k)
def thresh_policy(d): return lambda o: int(o["drop"] > d)
never  = lambda o: 0
always = lambda o: 1

def price_edges_for(price_mult=1.0):
    # tercile edges of the stationary price distribution (estimated once, fixed seed)
    ps = np.concatenate([make_episode("gradual", 900000 + i, price_mult)["price"] for i in range(5)])
    return list(np.quantile(ps, [1 / 3, 2 / 3]))

# ---------------- experiments ----------------
def evaluate(policy, eps, pe):
    out = [run_episode(e, policy, pe) for e in eps]
    return out

def summarize(res, never_res):
    u = np.array([r["utility"] for r in res]); acc = np.array([r["mean_acc"] for r in res])
    c = np.array([r["cost"] for r in res]); n = np.array([r["n_ret"] for r in res])
    gain = acc - np.array([r["mean_acc"] for r in never_res])
    apd = np.where(c > 0, gain / np.maximum(c, 1e-9), 0.0)
    return dict(utility=(u.mean(), u.std()), acc=(acc.mean(), acc.std()),
                cost=(c.mean(), c.std()), n_ret=(n.mean(), n.std()),
                apd=(apd.mean(), apd.std())), u, apd

def run_all(price_mult=1.0, n_train=1500, n_tune=15, n_test=30, seed0=0, log=print):
    pe = price_edges_for(price_mult)
    tune = {s: [make_episode(s, 50000 + 1000 * j + i, price_mult) for i in range(n_tune)]
            for j, s in enumerate(SCENARIOS)}
    tune_all = [e for s in SCENARIOS for e in tune[s]]
    # tune baselines on the pooled tuning set (one setting for all scenarios)
    best_k = max([2, 3, 5, 8, 12, 20, 30, 50], key=lambda k: np.mean(
        [r["utility"] for r in evaluate(fixed_policy(k), tune_all, pe)]))
    best_d = max([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25], key=lambda d: np.mean(
        [r["utility"] for r in evaluate(thresh_policy(d), tune_all, pe)]))
    log(f"tuned fixed k={best_k}, threshold d={best_d}")
    # train Q agents on mixed scenarios: raw reward and centred reward
    agents = {"RL raw reward": (QAgent(seed=seed0), 0.0), "RL centred reward": (QAgent(seed=seed0), A_MAX)}
    curves = {k: [] for k in agents}
    t0 = time.time()
    for name, (agent, off) in agents.items():
        for i in range(n_train):
            agent.eps = max(0.05, 1.0 - i / (0.6 * n_train))
            sc = SCENARIOS[i % 3]
            ep = make_episode(sc, 1000 + i, price_mult)
            r = run_episode(ep, agent.act, pe, learn=agent, r_off=off)
            curves[name].append(r["utility"])
        log(f"trained {name} in {time.time()-t0:.0f}s")
    curve = curves
    agent = agents["RL centred reward"][0]
    # test
    test = {s: [make_episode(s, 900000 + 1000 * j + i + 10, price_mult) for i in range(n_test)]
            for j, s in enumerate(SCENARIOS)}
    pols = {"Never": never, "Always": always, f"Fixed (k={best_k})": fixed_policy(best_k),
            f"Threshold ({best_d})": thresh_policy(best_d),
            "RL raw reward": agents["RL raw reward"][0].greedy,
            "RL centred reward": agents["RL centred reward"][0].greedy}
    results, raw = {}, {}
    for s in SCENARIOS + ["all"]:
        eps = test[s] if s != "all" else [e for q in SCENARIOS for e in test[q]]
        rs = {n: evaluate(p, eps, pe) for n, p in pols.items()}
        results[s] = {}; raw[s] = {}
        for n in pols:
            summ, u, apd = summarize(rs[n], rs["Never"])
            results[s][n] = summ; raw[s][n] = (u, apd)
    return dict(results=results, raw=raw, curve=curve, agent=agent, agents={k: v[0] for k, v in agents.items()}, pols=pols, pe=pe,
                test=test, best_k=best_k, best_d=best_d)

if __name__ == "__main__":
    t0 = time.time()
    out = run_all(n_train=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    print(time.time() - t0)
    for s in out["results"]:
        print("==", s)
        for n, m in out["results"][s].items():
            print(f"{n:22s} U={m['utility'][0]:7.1f}±{m['utility'][1]:5.1f} acc={m['acc'][0]:.3f} "
                  f"cost={m['cost'][0]:6.1f} n={m['n_ret'][0]:5.1f} apd={m['apd'][0]*1000:.2f}")
