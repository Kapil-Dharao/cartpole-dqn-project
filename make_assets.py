"""Create result figures and an actual policy playback from saved checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cartpole import CartPole
from dqn import Network


def plot(out):
    with (out/'evaluations.csv').open() as f: rows=list(csv.DictReader(f))
    summary=json.loads((out/'summary.json').read_text())
    fig,ax=plt.subplots(figsize=(9,4.5),dpi=160)
    for seed in summary['seeds']:
        rr=[r for r in rows if int(r['seed'])==seed]
        ax.plot([int(r['step']) for r in rr],[float(r['mean']) for r in rr],alpha=.72,lw=1.8,label=f'Seed {seed}')
    ax.axhline(400,color='#b24531',ls='--',lw=1.5,label='Part 1 target (400)')
    ax.axhline(summary['random_baseline_mean'],color='#777',ls=':',label='Random baseline')
    ax.axhline(summary['heuristic_baseline_mean'],color='#395b8a',ls=':',label='Angle rule')
    ax.set(xlabel='Environment steps',ylabel='Mean return over 20 episodes',ylim=(0,520))
    ax.grid(alpha=.18); ax.legend(ncol=2,fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(out/'learning_curves.png'); plt.close(fig)
    labels=[str(x['seed']) for x in summary['final_by_seed']]
    scores=[x['mean'] for x in summary['final_by_seed']]
    fig,ax=plt.subplots(figsize=(8,3.7),dpi=160)
    bars=ax.bar(labels,scores,color='#337f8a')
    ax.axhline(summary['random_baseline_mean'],color='#777',ls=':',label='Random')
    ax.axhline(summary['heuristic_baseline_mean'],color='#395b8a',ls=':',label='Angle rule')
    ax.axhline(400,color='#b24531',ls='--',label='Target')
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set(xlabel='Training seed',ylabel='Final mean return',ylim=(0,520)); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(out/'final_comparison.png'); plt.close(fig)


def render_frame(state,step,seed):
    w,h=960,540
    im=Image.new('RGB',(w,h),'#f8fafb'); d=ImageDraw.Draw(im)
    ground=385; center=480; scale=155
    d.line((0,ground+32,w,ground+32),fill='#2b4452',width=4)
    for tick in range(-2,3):
        tx=center+tick*scale
        d.line((tx,ground+20,tx,ground+40),fill='#a4b5bb',width=2)
    x,_,theta,_=state; cx=int(center+x*scale)
    d.rounded_rectangle((cx-55,ground-24,cx+55,ground+20),radius=7,fill='#237b87',outline='#14525b',width=3)
    tipx=int(cx+175*math.sin(theta)); tipy=int(ground-175*math.cos(theta))
    d.line((cx,ground-21,tipx,tipy),fill='#d75d40',width=15)
    d.ellipse((cx-9,ground-30,cx+9,ground-12),fill='#1f3540')
    d.ellipse((cx-39,ground+11,cx-20,ground+30),fill='#172b35')
    d.ellipse((cx+20,ground+11,cx+39,ground+30),fill='#172b35')
    d.text((45,34),f'DQN agent   |   seed {seed}',fill='#142b36')
    d.text((45,70),f'Step {step} / 500',fill='#365563')
    d.text((45,495),'Bundled CartPole physics  |  greedy policy  |  real checkpoint rollout',fill='#516b75')
    return im


def demo(out):
    summary=json.loads((out/'summary.json').read_text())
    selected=max(summary['final_by_seed'],key=lambda x:x['mean'])['seed']
    net=Network(selected); net.load(out/f'dqn_seed_{selected}.npz')
    env=CartPole(); state,_=env.reset(seed=10000)
    frames=[]; score=0
    for step in range(1,501):
        action=int(np.argmax(net.predict(state[None,:])))
        state,reward,term,trunc,_=env.step(action)
        score+=reward
        if step==1 or step%2==0: frames.append(render_frame(state,step,selected))
        if term or trunc: break
    frames[0].save(out/'agent_demo.gif',save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=True)
    frames[len(frames)//2].save(out/'demo_still.png')
    (out/'demo_metadata.json').write_text(json.dumps({'checkpoint_seed':selected,'episode_seed':10000,'episode_return':score,'frames':len(frames)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('results_full'))
    args=p.parse_args(); plot(args.output); demo(args.output)
