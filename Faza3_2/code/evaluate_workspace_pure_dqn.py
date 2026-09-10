import argparse,json
from pathlib import Path
import numpy as np
import torch
from precision_dqn import load_model
from workspace_benchmark import WorkspaceSamplingEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-steps',type=int,default=600);a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a fresh output')
    torch.set_num_threads(1);net,cfg,metadata=load_model(a.model);cfg.max_steps=a.max_steps
    env=WorkspaceSamplingEnv(cfg);episodes=[]
    for i,t in enumerate(json.loads(a.tasks.read_text())):
        s=env.reset(dict(start_q=t['start_q'],goal=t['goal']));positions=[env.ee().tolist()];qs=[env.data.qpos[env.ids.qpos_ids].tolist()]
        for k in range(a.max_steps):
            s,_,done,info=env.step(net.action(s));positions.append(env.ee().tolist());qs.append(env.data.qpos[env.ids.qpos_ids].tolist())
            if done:break
        episodes.append(dict(task=t,**info,positions=positions,qpos=qs))
        if (i+1)%24==0:print(i+1,'success',sum(e['success'] for e in episodes),flush=True)
    d=np.array([e['distance'] for e in episodes]);summary=dict(controller='DQN without global planner',model=str(a.model.resolve()),metadata=metadata,
        episodes=len(episodes),success_count=sum(e['success'] for e in episodes),success_rate=float(np.mean([e['success'] for e in episodes])),
        mean_error_mm=float(d.mean()*1000),p95_error_mm=float(np.percentile(d,95)*1000),collisions=sum(e['collision'] for e in episodes),max_steps=a.max_steps,control_dt=env.dt)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(summary=summary,episodes=episodes)))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
