import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from precision_environment import PrecisionConfig
from joint_waypoint_dqn import WorkspaceEnv,JointActionDQN,JointTeacher,joint_observation,apply_joint_action,save_joint
from workspace_planner import WorkspacePlanner


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=160)
    p.add_argument('--seed',type=int,default=411);a=p.parse_args()
    if a.output.exists():raise ValueError('Choose new output directory')
    a.output.mkdir(parents=True);torch.set_num_threads(1);np.random.seed(a.seed);random.seed(a.seed);torch.manual_seed(a.seed)
    cfg=PrecisionConfig(max_steps=600);env=WorkspaceEnv(cfg,a.seed);planner=WorkspacePlanner(env,a.seed)
    teacher=JointTeacher(env);states=[];scores=[];actions=[];rewards=[];nextstates=[];terminals=[];begin=time.time()
    for ep in range(a.episodes):
        for _ in range(3000):
            q=env.rng.uniform(env.low,env.high)
            if env.valid_pose(q) is None:continue
            target=np.clip(q+env.rng.uniform(-.18,.18,7),env.low,env.high);g=env.valid_pose(target)
            if g is not None and planner.edge(q,target):break
        else:raise RuntimeError('No valid local training task')
        env.reset(dict(start_q=q,goal=g));s=joint_observation(env,target)
        for k in range(100):
            label=teacher.scores(target);act=int(label.argmax()) if env.rng.random()>.08 else int(env.rng.integers(15))
            before=np.linalg.norm(target-env.data.qpos[env.ids.qpos_ids])
            _,_,done,info=apply_joint_action(env,target,act)
            after=np.linalg.norm(target-env.data.qpos[env.ids.qpos_ids]);ns=joint_observation(env,target)
            reward=4*(before-after)/max(before,.001)-.04+(3 if info['success'] else 0)-(5 if info['collision'] else 0)
            states.append(s);scores.append(label);actions.append(act);rewards.append(reward);nextstates.append(ns);terminals.append(info['terminated']);s=ns
            if done:break
        if (ep+1)%20==0:print('episodes',ep+1,'samples',len(states),'seconds',round(time.time()-begin),flush=True)
    arrays=dict(states=np.array(states),scores=np.array(scores),actions=np.array(actions),rewards=np.array(rewards,np.float32),nextstates=np.array(nextstates),terminals=np.array(terminals,np.float32))
    np.savez_compressed(a.output/'replay.npz',**arrays);d={k:torch.as_tensor(v) for k,v in arrays.items()};n=len(states)
    net=JointActionDQN();optim=torch.optim.Adam(net.parameters(),lr=3e-4)
    for u in range(2500):
        ix=torch.randint(n,(256,));q=net(d['states'][ix]);y=d['scores'][ix]
        loss=nn.functional.smooth_l1_loss(q,y)+.05*nn.functional.cross_entropy(q/.1,y.argmax(1))
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
    save_joint(a.output/'pretrain.pt',net,'teacher_pretrain',cfg,seed=a.seed,transitions=n)
    target=JointActionDQN();target.load_state_dict(net.state_dict())
    for group in optim.param_groups:group['lr']=1e-4
    for u in range(2500):
        ix=torch.randint(n,(256,));q=net(d['states'][ix]);chosen=q.gather(1,d['actions'][ix,None]).squeeze(1)
        with torch.no_grad():
            ns=d['nextstates'][ix];best=net(ns).argmax(1,keepdim=True)
            bellman=d['rewards'][ix]+.95*(1-d['terminals'][ix])*target(ns).gather(1,best).squeeze(1)
        expert=d['scores'][ix].argmax(1);margin=torch.full_like(q,.15);margin.scatter_(1,expert[:,None],0)
        loss=nn.functional.smooth_l1_loss(chosen,bellman)+((q+margin).max(1).values-q.gather(1,expert[:,None]).squeeze(1)).mean()
        optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step()
        if (u+1)%200==0:target.load_state_dict(net.state_dict())
    save_joint(a.output/'double_dqn.pt',net,'double_dqn',cfg,seed=a.seed,transitions=n,pretrain_updates=2500,dqn_updates=2500)
    (a.output/'training.json').write_text(json.dumps(dict(seed=a.seed,episodes=a.episodes,transitions=n,seconds=time.time()-begin),indent=2))
    print('Saved Double DQN',a.output,'elapsed',time.time()-begin,flush=True)
if __name__=='__main__':main()
