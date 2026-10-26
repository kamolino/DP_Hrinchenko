"""One letter on the table: the stage-1 one-policy environment with the letter instead of the block."""
import os

import mujoco
import numpy as np

from common import settings as cfg
from common.block_sim import BlockSim
from common.controller import TCPController
from common.geometry import wrap
from stage1_cube.cube_env import CubeEnv
from stage2_letters import letters as L

DEV_SEEDS = {"A": 512000, "R": 522000, "M": 532000}
TEST_SEEDS = {"A": 961000, "R": 971000, "M": 981000}
TRAIN_SEEDS = {"A": 712000, "R": 722000, "M": 732000}
START_LOW, START_HIGH = np.array([0.36, -0.17]), np.array([0.64, 0.17])
GOAL_LOW, GOAL_HIGH = np.array([0.38, -0.15]), np.array([0.60, 0.15])


def make_tasks(letter, count, seed):
    """Half of the tasks end in the letter's ARM slot, half at a random pose; move 0-15 cm, any turn."""
    rng, tasks = np.random.default_rng(seed), []
    arm_goal = np.array(L.GOALS[letter][:2])
    while len(tasks) < count:
        if rng.random() < 0.5:
            goal, goal_yaw = arm_goal.copy(), float(L.GOALS[letter][2])
            angle, length = rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.15)
            obj = goal + length * np.array([np.cos(angle), np.sin(angle)])
            yaw = float(rng.uniform(-np.pi, np.pi))
            if not (np.all(obj > START_LOW) and np.all(obj < START_HIGH)):
                continue
            kind = "arm"
        else:
            obj = rng.uniform(START_LOW, START_HIGH)
            yaw = float(rng.uniform(-np.pi, np.pi))
            angle, length = rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.15)
            goal = obj + length * np.array([np.cos(angle), np.sin(angle)])
            goal_yaw = wrap(yaw + rng.choice([-1, 1]) * rng.uniform(0, np.pi))
            if not (np.all(goal > GOAL_LOW) and np.all(goal < GOAL_HIGH)):
                continue
            kind = "random"
        if np.linalg.norm(goal - obj) < 0.02 and abs(wrap(goal_yaw - yaw)) < np.deg2rad(10):
            continue
        if L.on_table(letter, (*obj, yaw)) and L.on_table(letter, (*goal, goal_yaw)):
            tasks.append(dict(id=len(tasks), letter=letter, kind=kind, object=obj.tolist(), goal=goal.tolist(),
                              yaw=float(yaw), goal_yaw=float(goal_yaw)))
    return tasks


def letter_scene(letter):
    volume = sum(8 * hx * hy * L.HALF_HEIGHT for _, _, hx, hy, _ in L.LETTERS[letter])
    density = L.MASSES[letter] / volume
    geoms, ghost = [], []
    for i, (cx, cy, hx, hy, a) in enumerate(L.LETTERS[letter]):
        geoms.append(f'      <geom name="letter_{i}" type="box" pos="{cx} {cy} 0" size="{hx} {hy} {L.HALF_HEIGHT}" '
                     f'euler="0 0 {a}" density="{density:.1f}" friction="{L.FRICTION}" rgba="{L.COLORS[letter]}"/>')
        ghost.append(f'      <geom type="box" pos="{cx} {cy} 0" size="{hx} {hy} 0.001" euler="0 0 {a}" '
                     f'contype="0" conaffinity="0" rgba="0.1 0.9 0.35 0.45"/>')
    xml = f"""<mujoco model="FR3 letter {letter}">
  <include file="{cfg.FRANKA / 'fr3.xml'}"/>
  <compiler angle="radian" meshdir="{cfg.FRANKA / 'assets'}"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1200" offheight="800"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
  </visual>
  <worldbody>
    <light pos="0.5 -0.5 1.6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.02" rgba="0.14 0.18 0.23 1"/>
    <geom name="table" type="box" pos="0.50 0 0.15" size="0.29 0.28 0.03" friction="{L.FRICTION}" rgba="0.65 0.69 0.73 1"/>
    <body name="object" pos="0.48 0 {L.TABLE_TOP + L.HALF_HEIGHT}">
      <freejoint name="object_joint"/>
{chr(10).join(geoms)}
    </body>
    <body name="goal" mocap="true" pos="0.52 0 {L.TABLE_TOP + 0.001}">
{chr(10).join(ghost)}
    </body>
  </worldbody>
</mujoco>
"""
    path = L.GENERATED / f"letter_{letter}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != xml:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, path)
    return path


class LetterSim(BlockSim):
    """Same physics and rules as the block; contacts are checked for every box of the letter.

    During an episode the zone test is the block's rectangle (as the letter policies were trained);
    the reset and the word manager use the full letter outline.
    """

    def __init__(self, letter):
        self.letter = letter
        self.model = mujoco.MjModel.from_xml_path(str(letter_scene(letter)))
        self.data = mujoco.MjData(self.model)
        self.controller = TCPController(self.model)
        self.object_body = self.model.body("object").id
        self.object_qpos = self.model.joint("object_joint").qposadr[0]
        self.object_dof = self.model.joint("object_joint").dofadr[0]
        self.goal_mocap = self.model.body("goal").mocapid[0]
        self.table_geom = self.model.geom("table").id
        body_name = lambda g: self.model.body(self.model.geom_bodyid[g]).name  # noqa: E731
        self.is_letter = np.array([body_name(g) == "object" for g in range(self.model.ngeom)])
        self.finger_geoms = {g for g in range(self.model.ngeom)
                             if body_name(g) in ("left_finger", "right_finger") and self.model.geom_contype[g] != 0}
        self.is_finger = np.zeros(self.model.ngeom, dtype=bool)
        self.is_finger[list(self.finger_geoms)] = True
        self.goal, self.goal_yaw = np.array([0.52, 0.0]), 0.0
        self.applied_xy, self.last_action = np.zeros(2), np.zeros(4)
        self.steps = self.hold_steps = 0
        self.renderer = None
        self.camera = mujoco.MjvCamera()

    def keypoint_error(self):
        return 0.0

    def bad_push_contact(self):
        contacts = self.data.contact
        first, second = contacts.geom1, contacts.geom2
        touching = (self.is_letter[first] & self.is_finger[second]) | (self.is_letter[second] & self.is_finger[first])
        for index in np.flatnonzero(touching & (contacts.dist < 0)):
            if abs(contacts.frame[index, 2]) > 0.7:
                return "top_contact"
            if self.tcp[2] > cfg.TCP_HEIGHT + 0.006:
                return "high_contact"
        return ""

    def forbidden_contact(self):
        for contact in self.data.contact:
            if contact.dist >= -0.001:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if self.is_letter[a] and (b == self.table_geom or self.is_finger[b]):
                continue
            if self.is_letter[b] and (a == self.table_geom or self.is_finger[a]):
                continue
            return True
        return False

    def reset_failure(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "invalid_state"
        z = self.data.qpos[self.object_qpos + 2]
        if z < cfg.TABLE_TOP - 0.01:
            return "object_fell"
        if not L.on_table(self.letter, (*self.object_xy, self.yaw)):
            return "object_outside"
        if z > cfg.TABLE_TOP + 0.08:
            return "object_lifted"
        if self.data.xmat[self.object_body].reshape(3, 3)[2, 2] < np.cos(np.deg2rad(20)):
            return "object_tilted"
        reason = self.bad_push_contact()
        if reason:
            return reason
        if self.forbidden_contact():
            return "forbidden_contact"
        q = self.data.qpos[self.controller.qpos]
        limits = self.model.jnt_range[self.controller.joints]
        if np.any(q < limits[:, 0] - 0.01) or np.any(q > limits[:, 1] + 0.01):
            return "joint_limit"
        if not cfg.TCP_Z_LOW - 0.025 < self.tcp[2] < cfg.TCP_Z_HIGH + 0.025:
            return "tool_height"
        return ""


class LetterEnv(CubeEnv):
    def __init__(self, letter, task_seed=None):
        super().__init__(task_seed, sim=LetterSim(letter))
        self.letter = letter

    def new_task(self):
        return make_tasks(self.letter, 1, int(self.task_rng.integers(1 << 30)))[0]
