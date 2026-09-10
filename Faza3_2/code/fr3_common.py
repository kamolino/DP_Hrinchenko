from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'mujoco' / 'franka_fr3' / 'scene.xml'
JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]
ACTUATOR_NAMES = JOINT_NAMES.copy()
EE_SITE_NAME = 'attachment_site'
HOME_KEY_NAME = 'home'
TEACHER_TOLERANCE = 0.03
PRECISE_TOLERANCE = 0.01

def require_id(model, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id == -1:
        raise RuntimeError(f'MuJoCo object not found: {name}')
    return object_id

@dataclass
class FR3Indices:
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    actuator_ids: np.ndarray
    ee_site_id: int
    home_key_id: int

def load_model_and_data() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model file does not exist: {MODEL_PATH}')
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model)

def resolve_indices(model: mujoco.MjModel) -> FR3Indices:
    joint_ids=[]; qpos_ids=[]; dof_ids=[]; actuator_ids=[]
    for name in JOINT_NAMES:
        jid=require_id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        joint_ids.append(jid); qpos_ids.append(int(model.jnt_qposadr[jid])); dof_ids.append(int(model.jnt_dofadr[jid]))
    for name in ACTUATOR_NAMES:
        actuator_ids.append(require_id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name))
    return FR3Indices(np.asarray(joint_ids),np.asarray(qpos_ids),np.asarray(dof_ids),np.asarray(actuator_ids),require_id(model,mujoco.mjtObj.mjOBJ_SITE,EE_SITE_NAME),require_id(model,mujoco.mjtObj.mjOBJ_KEY,HOME_KEY_NAME))
