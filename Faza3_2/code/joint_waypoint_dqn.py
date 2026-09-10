"""Learned joint-action servo for explicit global joint waypoints.

Global XYZ routing belongs to WorkspacePlanner; this network chooses each of the
15 discrete physical control actions. There is no analytic argmax at inference.
"""
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
import torch
from torch import nn
from precision_environment import PrecisionEnv,PrecisionConfig


class WorkspaceEnv(PrecisionEnv):
    def valid_pose(self,q):
        q=np.asarray(q)
        if q.shape!=(7,) or not np.isfinite(q).all() or np.any(q<self.low) or np.any(q>self.high):return None
        mujoco.mj_resetDataKeyframe(self.model,self.scratch,self.ids.home_key_id)
        self.scratch.qpos[self.ids.qpos_ids]=q
        mujoco.mj_forward(self.model,self.scratch)
        p=self.scratch.site_xpos[self.ids.ee_site_id].copy()
        return p if p[2]>=.03 and not self.collision(self.scratch) else None


class JointActionDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)
    def action(self,x):
        with torch.no_grad():return int(self(torch.as_tensor(x)).argmax())


def joint_commands(env,target_q):
    q=env.data.qpos[env.ids.qpos_ids];ctrl=env.data.ctrl[env.ids.actuator_ids]
    error=target_q-q
    step=np.clip(.65*np.abs(error),.0003,.035)
    commands=np.repeat(ctrl[None,:],15,axis=0)
    for j in range(7):
        commands[1+2*j,j]+=step[j];commands[2+2*j,j]-=step[j]
    return np.clip(commands,env.low,env.high)


def joint_observation(env,target_q):
    q=env.data.qpos[env.ids.qpos_ids];ctrl=env.data.ctrl[env.ids.actuator_ids]
    vel=env.data.qvel[env.ids.dof_ids];error=target_q-q
    distance=np.linalg.norm(error);scale=max(distance,.001)
    lag=ctrl-q;commands=joint_commands(env,target_q)
    result=[]
    for a in range(15):
        j=(a-1)//2 if a else 0
        delta=commands[a,j]-ctrl[j] if a else 0
        margin=min(commands[a,j]-env.low[j],env.high[j]-commands[a,j])
        result.append([(2*error[j]*delta-delta*delta)/scale**2,
                       delta/scale,error[j]/scale,lag[j]/scale,vel[j]*env.dt/scale,
                       np.dot(error,lag)/scale**2,np.linalg.norm(lag)/scale,
                       np.linalg.norm(vel)*env.dt/scale,distance,np.log10(max(distance,1e-6))/6,
                       float(a==0),margin])
    return np.clip(np.array(result,dtype=np.float32),-10,10)


class JointTeacher:
    def __init__(self,env):self.env=env;self.data=mujoco.MjData(env.model)
    def scores(self,target):
        e=self.env;before=np.linalg.norm(target-e.data.qpos[e.ids.qpos_ids]);scores=[]
        for cmd in joint_commands(e,target):
            mujoco.mj_copyData(self.data,e.model,e.data);self.data.ctrl[e.ids.actuator_ids]=cmd
            collision=False
            for k in range(75):
                mujoco.mj_step(e.model,self.data)
                if k%25==0:collision |= e.collision(self.data)
            after=np.linalg.norm(target-self.data.qpos[e.ids.qpos_ids])
            scores.append(4*(before-after)/max(before,.001)-(5 if collision else 0))
        return np.array(scores,dtype=np.float32)


def apply_joint_action(env,target,action):
    """Advance the original physical env with a selected joint command."""
    commands=joint_commands(env,target)
    # Parent step(0) holds this exact command while physics, collision and final
    # Cartesian hold checks run unchanged. No qpos teleportation is performed.
    env.data.ctrl[env.ids.actuator_ids]=commands[action]
    return env.step(0)


def save_joint(path,net,stage,config,**metadata):
    torch.save(dict(version='joint_waypoint_v1',state_dict=net.state_dict(),stage=stage,
                    algorithm='Offline demonstration-augmented Double DQN',env_config=asdict(config),metadata=metadata),path)


def load_joint(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['version']!='joint_waypoint_v1':raise ValueError('Wrong joint policy checkpoint')
    net=JointActionDQN();net.load_state_dict(data['state_dict']);net.eval()
    return net,data
