"""XYZ -> collision-checked joint waypoints. Explicit planning, not learned DQN.

No test witness configuration is accepted by this API. IK uses XYZ and robot
kinematics, then RRT-Connect searches joint space if a direct path is blocked.
"""
import time
import numpy as np
import mujoco


class WorkspacePlanner:
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(env.model)
        self.jac = np.zeros((3, env.model.nv))
        self.checks = 0

    def fk(self, q):
        self.data.qpos[self.env.ids.qpos_ids] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.data)
        return self.data.site_xpos[self.env.ids.ee_site_id].copy()

    def valid(self, q):
        self.checks += 1
        if np.any(q < self.env.low) or np.any(q > self.env.high):
            return False
        p = self.fk(q)
        return p[2] >= .03 and not self.env.collision(self.data)

    def edge(self, a, b, resolution=.025):
        n = max(1,int(np.ceil(np.max(np.abs(b-a))/resolution)))
        return all(self.valid(a+(b-a)*i/n) for i in range(1,n+1))

    def inverse_candidates(self, start, goal, attempts=35):
        solutions=[]
        for attempt in range(attempts):
            if attempt == 0:q=start.copy()
            elif attempt == 1:q=self.env.home.copy()
            else:q=self.rng.uniform(self.env.low,self.env.high)
            for k in range(160):
                p=self.fk(q);error=goal-p
                if np.linalg.norm(error)<.00015:
                    if self.valid(q):solutions.append(q.copy())
                    break
                mujoco.mj_jacSite(self.env.model,self.data,self.jac,None,self.env.ids.ee_site_id)
                j=self.jac[:,self.env.ids.dof_ids]
                dq=j.T@np.linalg.solve(j@j.T+.002**2*np.eye(3),error)
                dq*=min(1,.16/max(np.max(np.abs(dq)),1e-9))
                q=np.clip(q+dq,self.env.low,self.env.high)
            if len(solutions)>=6:break
        return sorted(solutions,key=lambda q:np.linalg.norm(q-start))

    def rrt_connect(self,start,goal,iterations=3500):
        trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
        flipped=False
        def extend(tree,target):
            nodes,parents=tree
            nearest=int(np.argmin(np.linalg.norm(np.array(nodes)-target,axis=1)))
            d=target-nodes[nearest];length=np.linalg.norm(d)
            if length < 1e-9:return nearest,True
            q=nodes[nearest]+d*min(1,.22/length)
            if not self.edge(nodes[nearest],q):return None,False
            nodes.append(q);parents.append(nearest)
            return len(nodes)-1,length<=.22
        def trace(tree,i):
            result=[]
            while i!=-1:result.append(tree[0][i]);i=tree[1][i]
            return result[::-1]
        for _ in range(iterations):
            sample=self.rng.uniform(self.env.low,self.env.high)
            if self.rng.random()<.15:sample=trees[1][0][0]
            ai,_=extend(trees[0],sample)
            if ai is not None:
                target=trees[0][0][ai]
                for _ in range(100):
                    bi,reached=extend(trees[1],target)
                    if bi is None:break
                    if reached:
                        left=trace(trees[0],ai);right=trace(trees[1],bi)
                        path=left+right[-2::-1]
                        return path[::-1] if flipped else path
            trees=trees[::-1];flipped=not flipped
        return None

    def plan(self,start,goal):
        start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
        if start.shape!=(7,) or goal.shape!=(3,) or not np.isfinite(goal).all():
            raise ValueError('Finite start[7] and goal[3] required')
        begin=time.perf_counter();self.checks=0
        if not self.valid(start):
            return None,dict(status='invalid_start',planning_seconds=time.perf_counter()-begin)
        candidates=self.inverse_candidates(start,goal)
        for q in candidates:
            if self.edge(start,q):
                return [start.copy(),q],dict(status='direct',planning_seconds=time.perf_counter()-begin,checks=self.checks,ik_solutions=len(candidates))
        for q in candidates[:3]:
            path=self.rrt_connect(start,q)
            if path is not None:
                # Greedy line-of-sight simplification, preserving collision tests.
                smooth=[path[0]];i=0
                while i<len(path)-1:
                    for j in range(len(path)-1,i,-1):
                        if self.edge(path[i],path[j]):break
                    smooth.append(path[j]);i=j
                return smooth,dict(status='rrt',planning_seconds=time.perf_counter()-begin,checks=self.checks,ik_solutions=len(candidates))
        return None,dict(status='ik_failed' if not candidates else 'rrt_failed',planning_seconds=time.perf_counter()-begin,checks=self.checks,ik_solutions=len(candidates))
