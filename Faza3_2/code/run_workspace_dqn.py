"""Explicit global planner + learned joint-action DQN; evaluator never supplies IK witnesses."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from precision_environment import PrecisionConfig
from joint_waypoint_dqn import WorkspaceEnv,load_joint,joint_observation,joint_commands,apply_joint_action
from workspace_planner import WorkspacePlanner


def densify(path,max_delta=.12):
    result=[]
    for a,b in zip(path[:-1],path[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/max_delta)))
        result.extend(a+(b-a)*i/n for i in range(1,n+1))
    return result


def run_episode(env,net,start_q,goal,seed=0,viewer=None):
    # Only XYZ and starting measurements enter controller. No witness argument.
    env.reset(dict(start_q=start_q,goal=goal));planner=WorkspacePlanner(env,seed)
    q=env.data.qpos[env.ids.qpos_ids].copy()
    path,planning=planner.plan(q,np.asarray(goal))
    positions=[env.ee().tolist()];qs=[q.tolist()]
    info=dict(distance=env.distance,success=False,collision=False,steps=0,tcp_speed=0.,terminated=False,truncated=False)
    if path is None:
        return dict(**info,planning=planning,status=planning['status'],positions=positions,qpos=qs,shield_interventions=0)
    waypoints=densify(path);index=0;interventions=0;status='timeout'
    for _ in range(env.config.max_steps):
        begin=time.monotonic()
        if viewer is not None and not viewer.is_running():
            status='viewer_closed';break
        q=env.data.qpos[env.ids.qpos_ids].copy()
        while index<len(waypoints)-1 and np.max(np.abs(q-waypoints[index]))<.012:
            index+=1
        target=waypoints[index]
        obs=joint_observation(env,target)
        with torch.no_grad():order=net(torch.from_numpy(obs)).argsort(descending=True).numpy()
        commands=joint_commands(env,target)
        action=None
        for candidate in order:
            # Static collision shield only. It ranks no actions analytically and
            # does not simulate actions. Actual dynamics are checked by env.step.
            if planner.edge(q,commands[candidate],resolution=.015):
                action=int(candidate);break
        if action is None:status='shield_blocked';break
        interventions+=int(action!=order[0])
        _,_,done,info=apply_joint_action(env,target,action)
        positions.append(env.ee().tolist());qs.append(env.data.qpos[env.ids.qpos_ids].tolist())
        if viewer is not None:
            viewer.sync();time.sleep(max(0,env.dt-(time.monotonic()-begin)))
        if done:
            status='success' if info['success'] else 'collision' if info['collision'] else 'timeout';break
    return dict(**info,status=status,planning=planning,positions=positions,qpos=qs,
                shield_interventions=interventions,waypoint_count=len(waypoints),waypoints=[p.tolist() for p in waypoints])


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'results/workspace/joint_run_001/double_dqn.pt')
    p.add_argument('--tasks',type=Path);p.add_argument('--output',type=Path);p.add_argument('--goal',type=float,nargs=3)
    p.add_argument('--viewer',action='store_true');p.add_argument('--max-steps',type=int,default=600)
    a=p.parse_args();torch.set_num_threads(1);net,meta=load_joint(a.model)
    env=WorkspaceEnv(PrecisionConfig(max_steps=a.max_steps))
    if a.tasks:tasks=json.loads(a.tasks.read_text())
    elif a.goal:tasks=[dict(start_q=env.home.tolist(),goal=a.goal,group='custom')]
    else:raise ValueError('Provide --tasks or --goal X Y Z')
    if a.output and a.output.exists():raise ValueError('Use a fresh results path')
    from contextlib import nullcontext
    if a.viewer:import mujoco.viewer
    context=mujoco.viewer.launch_passive(env.model,env.data) if a.viewer else nullcontext(None)
    episodes=[]
    with context as viewer:
        for i,t in enumerate(tasks):
            r=run_episode(env,net,t['start_q'],t['goal'],seed=8100+i,viewer=viewer)
            episodes.append(dict(task=t,**r))
            print(i,t['group'],r['status'],round(r['distance']*1000,3),'mm',r['steps'],'steps',flush=True)
            if r['status']=='viewer_closed':break
    distances=np.array([r['distance'] for r in episodes]);successes=sum(r['success'] for r in episodes)
    summary=dict(controller='Global IK + RRT-Connect + joint-action Double DQN + collision shield',episodes=len(episodes),
                 success_count=successes,success_rate=successes/len(episodes),mean_error_mm=float(distances.mean()*1000),
                 p95_error_mm=float(np.percentile(distances,95)*1000),collisions=sum(r['collision'] for r in episodes),
                 tolerance_mm=1.,hold_seconds=.3,max_tcp_speed_mm_s=5.,control_dt=env.dt,max_steps=a.max_steps,
                 model=str(a.model.resolve()),planning_failures=sum(r['status'] in ('ik_failed','rrt_failed','invalid_start') for r in episodes))
    result=dict(summary=summary,episodes=episodes)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
