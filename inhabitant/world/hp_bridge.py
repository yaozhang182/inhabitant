"""Bridge to HumanCLAW's half-physics humanoid (HalfPhysicsEnv, imported from the HumanCLAW clone at
$HUMANCLAW_ROOT; see inhabitant/paths.py).

  executed_joints(body_state, rest_J)  realised SMPL-X state -> (T,22,3) joints (Z-up)
  action_joints(action, rest_J)        the same for an action dict (transl, global_orient, body_pose)
  terrain_offsets(J, ground_fn)        vertical correction of a placed motion on uneven ground (exactly 0 on a flat floor)
  floor_point, heading, half_physics_env
How HalfPhysicsEnv executes an action: per frame, the root gets the frame-to-frame translation of `transl` as a velocity
command (plus a gravity bias; furniture and walls stop it), the root orientation is set from `global_orient`, and the joints
track `body_pose`; absolute positions are used only when the body is reset.
HalfPhysicsEnv frames: habitat (Y-up) = W @ motion (Z-up), W = [[-1,0,0],[0,0,1],[0,1,0]]; it places the URDF base at
W (shift[0] + transl), shift[0] = pelvis of the neutral SMPL-X rest pose, so transl = pelvis - shift[0]."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..motion.smpl_fk import fk22
from ..paths import HUMANCLAW_ROOT

HC = HUMANCLAW_ROOT
SHIFT0 = np.load(HC / "resources/agent/neutral_beta0_handmerged/shift.npy")[0]
W = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
FEET = [7, 8, 10, 11]


def half_physics_env(**kw):
    if str(HC / "src") not in sys.path:
        sys.path.insert(0, str(HC / "src"))
    from humanclaw_bench.envs.half_physics_env import HalfPhysicsEnv
    return HalfPhysicsEnv(**kw)


def floor_point(sim, nav_point, up=0.5):
    """Navmesh points sit a few cm above the floor (a body placed on a navmesh point drops ~9.5 cm): cast a ray down
    from `up` m above the point and return the first hit (habitat Y-up), or the point itself if nothing is hit."""
    import magnum as mn
    import habitat_sim
    p = np.asarray(nav_point, float)
    ray = habitat_sim.geo.Ray(mn.Vector3(p[0], p[1] + up, p[2]), mn.Vector3(0, -1, 0))
    hits = sim.cast_ray(ray, max_distance=up + 1.0)
    if hits.has_hits():
        hp = hits.hits[0].point; return np.array([hp[0], hp[1], hp[2]])
    return p


def heading(orient):
    f = R.from_rotvec(orient).apply([0, 0, 1.0]); return float(np.arctan2(f[1], f[0]))


def executed_joints(body_state, rest_J=None):
    """HalfPhysicsEnv info['body_state'] -> (T,22,3) joints, Z-up, pelvis = transl + shift[0].
    rest_J: skeleton (default SMPL neutral; smpl_fk.sim_rest() gives the simulated body's real joint positions)."""
    t = np.asarray(body_state["transl"], float) + SHIFT0
    return fk22(t, np.asarray(body_state["global_orient"], float), np.asarray(body_state["body_pose"], float), rest_J)


def action_joints(action, rest_J=None):
    return fk22(np.asarray(action["transl"], float) + SHIFT0, action["global_orient"], action["body_pose"], rest_J)


def terrain_offsets(J, ground_fn, fps=30.0, vmax=0.5, dead=0.05, smooth_s=0.3):
    """Vertical correction of a placed reference for uneven ground. J (T,22,3) placed joints, Z-up world, standing on the
    ground at frame 0. ground_fn((N,2) xy) -> (N,) ground heights (any fixed offset is fine). At stance frames (lowest foot joint slower than
    vmax m/s) the body is raised by how much the real ground under that foot rose since the start minus how much the decoded
    foot itself rose (a decoded stair climb on real stairs gets no correction, a flat walk onto a step is lifted); held
    between stance frames and smoothed over smooth_s. Ground changes below `dead` m count as flat: on a flat floor the result is
    exactly zero, i.e. placement is unchanged. -> (T,) metres to add to the pelvis height."""
    J = np.asarray(J, float); T = len(J)
    fz = J[np.arange(T), np.array(FEET)[np.argmin(J[:, FEET, 2], 1)]]
    G = np.asarray(ground_fn(fz[:, :2]), float); rise = G - G[0]; rise[np.abs(rise) < dead] = 0.0
    if not rise.any():
        return np.zeros(T)
    v = np.r_[0.0, np.linalg.norm(np.diff(fz, axis=0), axis=1) * fps]
    d = np.full(T, np.nan); st = v < vmax; d[st] = rise[st] - (fz[st, 2] - fz[0, 2]); d[0] = 0.0 if np.isnan(d[0]) else d[0]
    for t in range(1, T):
        if np.isnan(d[t]):
            d[t] = d[t - 1]
    k = max(1, int(round(smooth_s * fps))); pad = np.r_[np.full(k, d[0]), d, np.full(k, d[-1])]
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
