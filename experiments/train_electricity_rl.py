"""Run from project root: python experiments/train_electricity_rl.py
Train on earlier Electricity data, select RL hyperparams on validation, report exploratory test.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from electricity_rl_env import ElectricityRetrainingEnv, FEATURES, load_data, BATCH_SIZE

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
train,val,test,prices=load_data(ROOT)
train_prices=prices[:241]; val_prices=prices[241:442]; test_prices=prices[520:721]
reference=float(np.median(train_prices))
WINDOW=1000
SEEDS=[0,1,2,3,4]
# train episodes from chronological training portion, with 1000+336 records of prior history
# NO training/validation/test leakage and no reuse of validation labels for RL gradient updates
RNG=np.random.default_rng(500)


def make_train_env(seed):
    r=np.random.default_rng(seed)
    def init():
        # pick a 12-batch segment, with 2k prior records for fitting
        n_batches=12; segment=n_batches*BATCH_SIZE
        start=int(r.integers(2000,len(train)-segment+1))
        hist=train.iloc[start-2000:start].copy()
        stream=train.iloc[start:start+segment].copy()
        pstart=int(r.integers(0,len(train_prices)-201+1))
        return ElectricityRetrainingEnv(hist,stream,train_prices[pstart:pstart+201],reference,window=WINDOW)
    return init

# Gymnasium instances have fixed streams; SB3 training can use a wrapper to vary episodes.
import gymnasium as gym
class TrainingSampler(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,seed):
        super().__init__(); self.make=make_train_env(seed); self.current=self.make()
        self.action_space=self.current.action_space
        self.observation_space=self.current.observation_space
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.current=self.make()
        return self.current.reset(seed=seed,options=options)
    def step(self,action):
        return self.current.step(action)


def eval_model(policy, history, stream, ppath):
    env=ElectricityRetrainingEnv(history,stream,ppath,reference,window=WINDOW)
    obs,_=env.reset()
    while True:
        action=policy(obs,env)
        obs,_,done,_,info=env.step(action)
        if done:
            return info


def train_tabular(seed,episodes=400):
    # 5 drop x 4 age x 3 price = 60 states; bins determined without test information
    r=np.random.default_rng(seed); Q=np.zeros((60,2))
    def state(obs):
        drop,scaled_age,relative_price=obs
        # age values in these short training episodes scaled by episode count
        db=np.digitize(drop,[0.02,0.05,0.10,0.20]); ab=np.digitize(scaled_age,[.25,.75,1.5]); pb=np.digitize(relative_price,[.97,1.03])
        return (int(db)*4+int(ab))*3+int(pb)
    sample_env=make_train_env(seed+10000)
    for i in range(episodes):
        env=sample_env(); obs,_=env.reset()
        epsilon=max(.05,1-i/(.6*episodes))
        while True:
            s=state(obs)
            a=int(r.integers(2)) if r.random()<epsilon else int(np.argmax(Q[s]))
            obs2,reward,done,_,_=env.step(a)
            target=reward if done else reward+.95*max(Q[state(obs2)])
            Q[s,a]+=.1*(target-Q[s,a])
            obs=obs2
            if done: break
    return lambda obs,env:int(np.argmax(Q[state(obs)]))


def heuristic_policy(name, config):
    def policy(obs, env):
        drop,_,_=obs
        if name=='Never': return 0
        if name=='Fixed': return int(env.age>=config['k'])
        if name=='Threshold': return int(drop>config['d'])
        if name=='Price-aware':
            return int(drop>config['d']*(env.mapped_prices[env.t]/reference)**config['eta'])
        raise ValueError(name)
    return policy


# Note: previous heuristic benchmark used another decision timing and reference definition.
# Recompute heuristics for THIS environment instead of comparing unmatched numbers.
CANDIDATES=[('Never',{})]
CANDIDATES += [('Fixed',{'k':k}) for k in [3,6,12,20]]
CANDIDATES += [('Threshold',{'d':d}) for d in [0.02,0.05,0.1,0.18,0.25]]
CANDIDATES += [('Price-aware',{'d':d,'eta':e}) for d in [0.02,0.05,0.1,0.18,0.25] for e in [0,1,2]]

print('Tuning heuristics against new leakage-safe environment...',flush=True)
valid=[]
for name,cfg in CANDIDATES:
    m=eval_model(heuristic_policy(name,cfg),train,val,val_prices)
    valid.append(dict(policy=name,config=str(cfg),utility=m['utility'],cfg=cfg))
selected={}
for item in valid:
    if item['policy'] not in selected or item['utility']>selected[item['policy']]['utility']:
        selected[item['policy']]=item
print('Selected heuristics:',{k:(v['cfg'],round(v['utility'],2)) for k,v in selected.items()},flush=True)

# Train independently, choose learning rate ONLY on validation; report all five seeds
rows=[]
for seed in SEEDS:
    print('Training seed',seed,flush=True)
    q= train_tabular(seed)
    v=eval_model(q,train,val,val_prices)
    t=eval_model(q,pd.concat([train,val],ignore_index=True),test,test_prices)
    rows.append(dict(policy='Q-learning',seed=seed,validation_utility=v['utility'],**t))
    env=TrainingSampler(seed)
    if seed==0: check_env(env,warn=True)
    candidates=[]
    for lr in [1e-4,3e-4]:
        agent=DQN('MlpPolicy',env,learning_rate=lr,buffer_size=10000,
                  learning_starts=250,batch_size=64,gamma=.95,train_freq=4,
                  exploration_fraction=.6,exploration_final_eps=.05,
                  policy_kwargs={'net_arch':[64,64]},seed=seed,verbose=0,device='cpu')
        agent.learn(total_timesteps=12000)
        decision=lambda obs,e,model=agent:int(model.predict(obs,deterministic=True)[0])
        result=eval_model(decision,train,val,val_prices)
        candidates.append((result['utility'],lr,agent))
    best=max(candidates,key=lambda x:x[0])
    decision=lambda obs,e,model=best[2]:int(model.predict(obs,deterministic=True)[0])
    test_result=eval_model(decision,pd.concat([train,val],ignore_index=True),test,test_prices)
    rows.append(dict(policy='DQN',seed=seed,selected_lr=best[1],validation_utility=best[0],**test_result))
    best[2].save(str(OUT/f'electricity_dqn_seed{seed}.zip'))

for name,item in selected.items():
    m=eval_model(heuristic_policy(name,item['cfg']),pd.concat([train,val],ignore_index=True),test,test_prices)
    rows.append(dict(policy=name,seed=-1,validation_utility=item['utility'],**m))

results=pd.DataFrame(rows)
results.to_csv(OUT/'electricity_rl_final_episodes.csv',index=False)
print('\n=== EXPLORATORY ELECTRICITY POLICY RESULTS ===')
print(results.groupby('policy')[['utility','accuracy','cost','retrainings']].mean().to_string())
print('\nSaved',OUT/'electricity_rl_final_episodes.csv')
