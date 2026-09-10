import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from joint_waypoint_dqn import WorkspaceEnv,JointActionDQN,joint_commands,joint_observation
from workspace_planner import WorkspacePlanner
from run_workspace_dqn import densify

class PlannerTests(unittest.TestCase):
    def test_plans_xyz_without_changing_live_state(self):
        env=WorkspaceEnv(seed=21);task=env.sample_task(.6);env.reset(task)
        q=env.data.qpos[env.ids.qpos_ids].copy();live=env.data.qpos.copy();vel=env.data.qvel.copy()
        planner=WorkspacePlanner(env,31);path,info=planner.plan(q,np.array(task['goal']))
        self.assertIsNotNone(path);self.assertLess(np.linalg.norm(planner.fk(path[-1])-task['goal']),.0002)
        self.assertTrue(all(planner.edge(a,b) for a,b in zip(path[:-1],path[1:])))
        np.testing.assert_array_equal(live,env.data.qpos);np.testing.assert_array_equal(vel,env.data.qvel)
        p2,_=WorkspacePlanner(env,31).plan(q,np.array(task['goal']))
        np.testing.assert_allclose(path,p2)

    def test_joint_policy_contract_and_bounds(self):
        env=WorkspaceEnv();target=env.home+.02
        commands=joint_commands(env,target)
        self.assertEqual(commands.shape,(15,7));self.assertTrue((commands>=env.low).all() and (commands<=env.high).all())
        obs=joint_observation(env,target);self.assertEqual(obs.shape,(15,12));self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(0<=JointActionDQN().action(obs)<15)

    def test_densify_preserves_checked_polyline(self):
        path=[np.zeros(7),np.ones(7),np.array([1,2,1,1,1,1,1])]
        dense=densify(path,.12)
        np.testing.assert_allclose(dense[-1],path[-1])
        for a,b in zip([path[0]]+dense,dense):self.assertLessEqual(np.max(np.abs(a-b)),.120001)

    def test_invalid_start_rejected_and_unreachable_target_fails(self):
        env=WorkspaceEnv();p=WorkspacePlanner(env)
        path,info=p.plan(env.high+1,np.zeros(3));self.assertIsNone(path);self.assertEqual(info['status'],'invalid_start')
        self.assertEqual(p.inverse_candidates(env.home,np.array([20.,20.,20.]),attempts=2),[])

    def test_rrt_connect_routes_around_blocked_joint_region(self):
        env=WorkspaceEnv();p=WorkspacePlanner(env,17)
        a=env.home.copy();b=env.home.copy();a[0]=-.6;b[0]=.6
        p.valid=lambda q: bool(np.all(q>=env.low) and np.all(q<=env.high)
                              and not (-.15<q[0]<.15 and -.2<q[1]<.2))
        self.assertFalse(p.edge(a,b))
        path=p.rrt_connect(a,b)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0],a);np.testing.assert_allclose(path[-1],b)
        self.assertTrue(all(p.edge(x,y) for x,y in zip(path[:-1],path[1:])))

if __name__=='__main__':unittest.main()
