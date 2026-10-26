"""Letters A, R, M (6 x 6 cm, 5 cm tall) built from boxes, the ARM goal layout and random starts."""
import numpy as np

from common import settings as cfg

TABLE_TOP = 0.18
HALF_HEIGHT = 0.025
MASS = 0.08
FRICTION = "0.25 0.005 0.0001"
NAMES = ("A", "R", "M")
COLORS = {"A": "1 0.4 0.08 1", "R": "0.15 0.55 0.95 1", "M": "0.75 0.25 0.85 1"}
GENERATED = cfg.MUJOCO / "generated"


def bar(p, q, half_width):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    return (*(p + q) / 2, np.linalg.norm(d) / 2 + half_width, half_width, float(np.arctan2(d[1], d[0])))


def rect(cx, cy, hx, hy):
    return (cx, cy, hx, hy, 0.0)


def shapes(H, T, m_extra):
    """Boxes (cx, cy, hx, hy, angle) of each letter; H = half outline, T = half stroke width."""
    return {
        "A": [
            bar((-0.825 * H, -H + T), (-0.1 * H, H - T), T),
            bar((0.825 * H, -H + T), (0.1 * H, H - T), T),
            rect(0.0, H - T, 0.3 * H, T),
            rect(0.0, -0.2 * H, 0.55 * H, T),
        ],
        "R": [
            rect(-0.825 * H, 0.0, T, H),
            rect(-0.15 * H, H - T, 0.675 * H, T),
            rect(0.5 * H, 0.5 * H, T, 0.5 * H),
            rect(-0.15 * H, 0.0, 0.675 * H, T),
            bar((-0.1 * H, 0.0), (0.825 * H, -H + T), T),
        ],
        "M": [
            rect(-0.825 * H, 0.0, T, H),
            rect(0.775 * H, 0.0, T + m_extra, H),       # thicker right stem: M is not 180 deg symmetric
            bar((-0.825 * H, H - T), (0.0, -0.4 * H), T),
            bar((0.775 * H, H - T), (0.0, -0.4 * H), T),
        ],
    }


LETTERS = shapes(0.03, 0.0055, 0.0015)
_REFERENCE = shapes(0.04, 0.007, 0.002)


def _area(boxes):
    return sum(4 * hx * hy for _, _, hx, hy, _ in boxes)


# mass proportional to area; an 8 x 8 cm letter weighs 0.08 kg
MASSES = {n: MASS * _area(LETTERS[n]) / _area(_REFERENCE[n]) for n in NAMES}

# letters upright as read from the robot, in a row along y, gaps >= 4 cm
GOAL_YAW = -np.pi / 2
GOALS = {"A": (0.50, 0.105, GOAL_YAW), "R": (0.50, 0.0, GOAL_YAW), "M": (0.50, -0.105, GOAL_YAW)}

START_LOW, START_HIGH = np.array([0.36, -0.17]), np.array([0.64, 0.17])
WORKSPACE_LOW, WORKSPACE_HIGH = np.array([0.24, -0.25]), np.array([0.76, 0.25])
TABLE_LOW, TABLE_HIGH = np.array([0.21, -0.28]), np.array([0.79, 0.28])
MIN_GAP = 0.03
RADIUS = 0.06


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def polygons(name, x=0.0, y=0.0, yaw=0.0):
    """World corners (4 x 2) of every box of a letter at pose (x, y, yaw)."""
    out = []
    for cx, cy, hx, hy, a in LETTERS[name]:
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rot(a).T + [cx, cy]
        out.append(local @ rot(yaw).T + [x, y])
    return out


def _separated(p, q):
    for poly in (p, q):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            if (p @ normal).max() < (q @ normal).min() or (q @ normal).max() < (p @ normal).min():
                return True
    return False


def _point_segment(p, a, b):
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return np.linalg.norm(p - (a + t * (b - a)))


def box_distance(p, q):
    if not _separated(p, q):
        return 0.0
    return min(_point_segment(v, s[i], s[(i + 1) % 4]) for v_set, s in ((p, q), (q, p)) for v in v_set for i in range(4))


def letter_distance(a, pose_a, b, pose_b):
    return min(box_distance(p, q) for p in polygons(a, *pose_a) for q in polygons(b, *pose_b))


def on_table(name, pose, margin=0.0):
    corners = np.vstack(polygons(name, *pose))
    return bool(np.all(corners > WORKSPACE_LOW + margin) and np.all(corners < WORKSPACE_HIGH - margin))


def away_from_edge(name, pose, edge):
    corners = np.vstack(polygons(name, *pose))
    return bool(np.all(corners > TABLE_LOW + edge) and np.all(corners < TABLE_HIGH - edge))


def closer_than(a, pose_a, b, pose_b, gap):
    if np.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1]) > 2 * RADIUS + gap:
        return False
    return letter_distance(a, pose_a, b, pose_b) < gap


def sample_starts(rng, tries=20000, edge=None):
    """Three letters, random yaw, >= 3 cm from each other and from the other letters' goals."""
    for _ in range(tries):
        poses = {}
        for name in NAMES:
            pose = (*rng.uniform(START_LOW, START_HIGH), float(rng.uniform(-np.pi, np.pi)))
            if not on_table(name, pose, 0.01):
                break
            if edge is not None and not away_from_edge(name, pose, edge):
                break
            if any(closer_than(name, pose, other, p, MIN_GAP) for other, p in poses.items()):
                break
            if any(closer_than(name, pose, other, GOALS[other], MIN_GAP) for other in NAMES if other != name):
                break
            poses[name] = pose
        if len(poses) == 3:
            return poses
    raise RuntimeError("Could not sample letter starts.")


def letter_body(name, pose):
    x, y, yaw = pose
    volume = sum(8 * hx * hy * HALF_HEIGHT for _, _, hx, hy, _ in LETTERS[name])
    density = MASSES[name] / volume
    lines = [f'    <body name="{name}" pos="{x} {y} {TABLE_TOP + HALF_HEIGHT}" '
             f'quat="{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}">',
             f'      <freejoint name="{name}_joint"/>']
    for i, (cx, cy, hx, hy, a) in enumerate(LETTERS[name]):
        lines.append(f'      <geom name="{name}_{i}" type="box" pos="{cx} {cy} 0" '
                     f'size="{hx} {hy} {HALF_HEIGHT}" euler="0 0 {a}" density="{density:.1f}" '
                     f'friction="{FRICTION}" rgba="{COLORS[name]}"/>')
    lines.append("    </body>")
    return lines


def goal_body(name, pose):
    x, y, yaw = pose
    lines = [f'    <body name="goal_{name}" mocap="true" pos="{x} {y} {TABLE_TOP + 0.001}" '
             f'quat="{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}">']
    for cx, cy, hx, hy, a in LETTERS[name]:
        lines.append(f'      <geom type="box" pos="{cx} {cy} 0" size="{hx} {hy} 0.001" euler="0 0 {a}" '
                     f'contype="0" conaffinity="0" rgba="0.1 0.9 0.35 0.45"/>')
    lines.append("    </body>")
    return lines


def word_scene(path, starts=None):
    """Robot, table, the three letters and their translucent goals."""
    starts = starts or {"A": (0.40, 0.12, 0.0), "R": (0.40, -0.12, 0.6), "M": (0.62, 0.0, 2.5)}
    body = []
    for name in NAMES:
        body += letter_body(name, starts[name])
    for name in NAMES:
        body += goal_body(name, GOALS[name])
    xml = f"""<mujoco model="FR3 letters ARM">
  <include file="{cfg.FRANKA / 'fr3.xml'}"/>
  <compiler angle="radian" meshdir="{cfg.FRANKA / 'assets'}"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <statistic center="0.35 0 0.3" extent="0.9"/>
  <visual>
    <global offwidth="1200" offheight="800" azimuth="135" elevation="-30"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
  </visual>
  <worldbody>
    <light pos="0.5 -0.5 1.6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.02" rgba="0.14 0.18 0.23 1"/>
    <geom name="table" type="box" pos="0.50 0 0.15" size="0.29 0.28 0.03"
          friction="{FRICTION}" rgba="0.65 0.69 0.73 1"/>
{chr(10).join(body)}
  </worldbody>
</mujoco>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)
    return path
