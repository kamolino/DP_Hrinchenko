import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from precision_environment import PrecisionConfig,PrecisionEnv
from precision_dqn import ActionDQN,PhysicsTeacher,save_model,load_model,evaluate


class PrecisionTests(unittest.TestCase):
    def test_reachable_task_witness_and_determinism(self):
        a=PrecisionEnv(seed=91);b=PrecisionEnv(seed=91)
        for _ in range(12):
            ta=a.sample_task(.6,True);tb=b.sample_task(.6,True)
            self.assertEqual(ta,tb)
            np.testing.assert_allclose(a.valid_pose(np.array(ta['witness_q'])),ta['goal'],atol=1e-12)
            self.assertIsNotNone(a.valid_pose(np.array(ta['start_q'])))

    def test_physical_hold_and_episode_lifecycle(self):
        env=PrecisionEnv()
        env.reset()
        for _ in range(100):
            mujoco.mj_step(env.model,env.data)
        env.goal=env.ee();env.distance=0.
        for k in range(env.hold_steps):
            s,r,done,info=env.step(0)
            self.assertEqual(done,k==env.hold_steps-1)
        self.assertTrue(info['success']);self.assertFalse(info['truncated'])
        self.assertLessEqual(info['tcp_speed'],env.config.max_tcp_speed)
        with self.assertRaises(RuntimeError):env.step(0)

    def test_timeout_is_not_terminal(self):
        env=PrecisionEnv(PrecisionConfig(max_steps=1));env.reset()
        _,_,done,info=env.step(0)
        self.assertTrue(done);self.assertTrue(info['truncated']);self.assertFalse(info['terminated'])

    def test_teacher_does_not_modify_live_state(self):
        env=PrecisionEnv();env.reset()
        q=env.data.qpos.copy();v=env.data.qvel.copy();ctrl=env.data.ctrl.copy();t=env.data.time
        scores=PhysicsTeacher(env).scores()
        self.assertEqual(scores.shape,(15,));self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(q,env.data.qpos);np.testing.assert_array_equal(v,env.data.qvel)
        np.testing.assert_array_equal(ctrl,env.data.ctrl);self.assertEqual(t,env.data.time)

    def test_commands_bounded_and_network_contract(self):
        env=PrecisionEnv();s=env.reset()
        self.assertEqual(s.shape,(15,19));self.assertTrue(np.isfinite(s).all())
        cmd,_,_=env.commands()
        self.assertTrue((cmd>=env.low).all() and (cmd<=env.high).all())
        self.assertEqual(ActionDQN()(torch.from_numpy(s)).shape,(15,))
        for a in [-1,15,1.2]:
            with self.assertRaises(ValueError):env.step(a)

    def test_tcp_velocity_prevents_false_success(self):
        env=PrecisionEnv();env.reset();env.goal=env.ee();env.distance=0.
        env.data.qvel[env.ids.dof_ids[0]]=1.
        _,_,_,info=env.step(0)
        self.assertFalse(info['success']);self.assertEqual(info['hold_count'],0)

    def test_checkpoint_and_teacher_free_evaluation(self):
        net=ActionDQN();cfg=PrecisionConfig(max_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.pt';save_model(path,net,cfg,stage='test')
            loaded,config,metadata=load_model(path)
            self.assertEqual(config,cfg);self.assertEqual(metadata['stage'],'test')
            env=PrecisionEnv();task=env.sample_task();task['group']='test'
            with patch.object(PhysicsTeacher,'scores',side_effect=AssertionError('Teacher used at evaluation')):
                summary,rows,_=evaluate(loaded,config,[task])
            self.assertEqual(len(rows),1)

if __name__=='__main__':unittest.main()
