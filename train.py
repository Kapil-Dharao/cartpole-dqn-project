"""Train and evaluate a CartPole DQN. Run `python train.py --help`."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from cartpole import CartPole
from dqn import Network, Replay


def environment(use_gym=False):
    if use_gym:
        import gymnasium as gym
        return gym.make('CartPole-v1')
    return CartPole()


def evaluate(policy, seeds, use_gym=False):
    env=environment(use_gym)
    returns=[]
    for seed in seeds:
        s,_=env.reset(seed=int(seed)); total=0
        for _ in range(500):
            a=int(policy(s)); s,r,term,trunc,_=env.step(a)
            total+=r
            if term or trunc: break
        returns.append(total)
    env.close()
    return np.array(returns,dtype=float)


def train(seed, steps, out, use_gym=False, eval_every=5000, double=False, learning_rate=.001):
    rng=np.random.default_rng(seed)
    env=environment(use_gym)
    online=Network(seed,learning_rate=learning_rate); target=Network(seed+10000); target.copy_from(online)
    replay=Replay()
    s,_=env.reset(seed=seed)
    eval_seeds=np.arange(10000,10020)
    history=[]; episode_returns=[]; episode_return=0
    started=time.perf_counter()
    for step in range(1,steps+1):
        epsilon=max(.05,1-.95*step/20000)
        a=int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(online.predict(s[None,:])))
        ns,r,terminated,truncated,_=env.step(a)
        replay.add(s,a,r,ns,terminated)
        episode_return+=r
        s=ns
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return=0
            s,_=env.reset()
        if step>1000 and step%4==0:
            bs,ba,br,bns,bdone=replay.sample(rng,64)
            future=(target.predict(bns)[np.arange(len(ba)),np.argmax(online.predict(bns),axis=1)]
                    if double else np.max(target.predict(bns),axis=1))
            y=br+.99*(1-bdone)*future
            online.update(bs,ba,y)
        if step%1000==0: target.copy_from(online)
        if step%eval_every==0 or step==steps:
            vals=evaluate(lambda state: np.argmax(online.predict(state[None,:])),eval_seeds,use_gym)
            row={'seed':seed,'step':step,'mean':round(float(vals.mean()),2),
                 'std':round(float(vals.std(ddof=1)),2),'success_rate':round(float(np.mean(vals==500)),3)}
            history.append(row)
            print(row,flush=True)
    env.close()
    online.save(out/f'dqn_seed_{seed}.npz')
    return history,episode_returns,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33,44,55])
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--gym',action='store_true',help='Use Gymnasium CartPole-v1 instead of the bundled compatible physics')
    p.add_argument('--double',action='store_true',help='Use Double DQN targets')
    p.add_argument('--learning-rate',type=float,default=.001)
    args,_=p.parse_known_args()  # Also works in notebooks that pass -f.
    args.output.mkdir(parents=True,exist_ok=True)
    eval_seeds=np.arange(10000,10020)
    # Use one reproducible action RNG per evaluation episode.
    env=environment(args.gym); random_scores=[]
    for sd in eval_seeds:
        rng=np.random.default_rng(int(sd)+90000)
        s,_=env.reset(seed=int(sd)); total=0
        for _ in range(500):
            s,r,t,tr,_=env.step(int(rng.integers(2))); total+=r
            if t or tr: break
        random_scores.append(total)
    env.close()
    random_scores=np.array(random_scores)
    heuristic_scores=evaluate(lambda s: int(s[2]>0),eval_seeds,args.gym)
    histories=[]; times={}; training=[]
    for seed in args.seeds:
        h,e,elapsed=train(seed,args.steps,args.output,args.gym,double=args.double,learning_rate=args.learning_rate)
        histories+=h; times[str(seed)]=round(elapsed,2)
        training.extend([{'seed':seed,'episode':i+1,'return':x} for i,x in enumerate(e)])
    for name,rows in [('evaluations.csv',histories),('training_episodes.csv',training)]:
        with (args.output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    final=[h[-1] for h in [list(filter(lambda row:row['seed']==seed,histories)) for seed in args.seeds]]
    summary={'environment':'Gymnasium CartPole-v1' if args.gym else 'bundled CartPole-v1 physics',
             'algorithm':'Double DQN' if args.double else 'DQN',
             'learning_rate':args.learning_rate,
             'seeds':args.seeds,'steps_per_seed':args.steps,'evaluation_episodes':20,
             'random_baseline_mean':float(np.mean(random_scores)),
             'heuristic_baseline_mean':float(np.mean(heuristic_scores)),
             'final_by_seed':final,'median_dqn_final_mean':float(np.median([x['mean'] for x in final])),
             'elapsed_seconds_by_seed':times,
             'success_criterion_met':sum(x['mean']>=400 for x in final)>=4 and
              np.median([x['mean'] for x in final])>max(np.median(random_scores),np.median(heuristic_scores))}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
