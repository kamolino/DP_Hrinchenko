import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from workspace_benchmark import WorkspaceSamplingEnv, generate_suite, point_bins, validate_suite


class WorkspaceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.coverage = generate_suite(190011, count=36, pool_size=1800)

    def test_deterministic_independent_splits(self):
        same, _ = generate_suite(190011, count=36, pool_size=1800)
        other, _ = generate_suite(190012, count=36, pool_size=1800)
        self.assertEqual(self.tasks, same)
        goals = {tuple(t['goal']) for t in self.tasks}
        self.assertFalse(goals.intersection(tuple(t['goal']) for t in other))

    def test_full_azimuth_and_diverse_starts(self):
        self.assertEqual(self.coverage['goal_azimuth_counts'], [3]*12)
        for group in ('global_home','boundary_home','random_start'):
            bins = [t['bin']['azimuth'] for t in self.tasks if t['group'] == group]
            self.assertEqual(sorted(bins), list(range(12)))
        starts = np.array([t['start_q'] for t in self.tasks if t['group'] == 'random_start'])
        self.assertTrue(np.all(np.ptp(starts, axis=0) > 1.))
        self.assertGreater(sum(c > 0 for c in self.coverage['goal_radius_counts']), 3)
        self.assertGreater(sum(c > 0 for c in self.coverage['goal_height_counts']), 4)

    def test_witnesses_feasible_and_corruption_is_error(self):
        self.assertEqual(validate_suite(self.tasks)['validated_tasks'], 36)
        corrupted = copy.deepcopy(self.tasks)
        corrupted[3]['goal'][0] += .01
        with self.assertRaisesRegex(ValueError, 'witness disagree'):
            validate_suite(corrupted)

    def test_local_validation_keeps_live_state(self):
        env = WorkspaceSamplingEnv()
        q, v, t = env.data.qpos.copy(), env.data.qvel.copy(), env.data.time
        for task in self.tasks:
            self.assertIsNotNone(env.valid_pose(task['witness_q']))
        np.testing.assert_array_equal(q, env.data.qpos)
        np.testing.assert_array_equal(v, env.data.qvel)
        self.assertEqual(t, env.data.time)
        self.assertIsNone(env.valid_pose([float('nan')]*7))
        self.assertIsNone(env.valid_pose(env.high+.01))

    def test_bins_wrap_negative_x_and_invalid_counts_raise(self):
        bins = point_bins([[-.5,0,.03],[-.5,-0.,.03],[.5,0,1.2]])
        self.assertEqual(bins[0,0], bins[1,0])
        self.assertEqual(bins[2,0], 6)
        with self.assertRaises(ValueError):
            generate_suite(1, count=35)


if __name__ == '__main__':
    unittest.main()
