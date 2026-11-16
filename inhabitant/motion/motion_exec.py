"""From a generated motion to the simulated body.

Frames. Segment frame: x forward, y left, z up (m), origin on the floor below the pelvis at the start of the segment, x = the
facing at the start. Kimodo's frame of a segment is the same frame with other axis names: X left, Y up, Z forward, so
kimodo (X, Y, Z) = segment (y, z, x) (`seg_to_kim`, `kim_to_seg`).

Bodies. The generator's body (Kimodo's SMPL-X neutral) is larger than the simulated one (HumanCLAW's SMPL-X neutral,
smpl_fk.sim_rest(): thigh 0.42 vs 0.38 m, shin 0.44 vs 0.41 m). `to_sim` gives the generated pose the simulated body's bones
(every bone keeps its direction) and moves it up or down, frame by frame, so that the lowest foot joint keeps its height above
the floor (KIM_FEET -> SIM_FEET: the two skeletons' foot joints stand at different heights).

Execution: action_from_local (IK on the simulated body, placed exactly as given) -> HalfPhysicsEnv action."""
import math
import numpy as np

JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
          "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
FEET = [7, 8, 10, 11]                       # ankles and toes
FPS, MIN_T, MAX_T = 30.0, 15, 241
SIM_FEET = 0.042                            # lowest foot joint of the simulated body standing on the floor (executed, m)
KIM_FEET = 0.007                            # the same for Kimodo's body (its post-processing keeps the lowest joint 0.007 m up)


# ----------------------------------------------------------------- frames
def seg_to_kim(p):
    p = np.asarray(p, float); return np.stack([p[..., 1], p[..., 2], p[..., 0]], -1)


def kim_to_seg(j):
    j = np.asarray(j, float); return np.stack([j[..., 2], j[..., 0], j[..., 1]], -1)


def hml_to_local(j):
    """HumanML3D / HumanCLAW frame (X left, Y up, Z forward) -> segment frame (= kim_to_seg)."""
    return kim_to_seg(j)


def local_to_hml(a):
    return seg_to_kim(a)


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def to_local(J, yaw, origin):
    """Z-up world points -> segment frame whose x axis has world yaw `yaw` and whose origin is `origin` (Z-up)."""
    return (np.asarray(J, float) - origin) @ _rz(yaw)


def to_world(A, yaw, origin):
    return np.asarray(A, float) @ _rz(yaw).T + origin


def facing_yaw(A):
    """(T,22,3) segment-frame joints -> facing angle per frame (rad, + = left), from the hip and shoulder lines."""
    a = (A[:, 1] - A[:, 2]) + (A[:, 16] - A[:, 17])            # pointing left
    return np.unwrap(np.arctan2(-a[:, 0], a[:, 1]))


def local_root_motion(A):
    """Pelvis displacement (forward, left) along the facing at frame 0 and heading change (deg) of segment-frame joints."""
    y = facing_yaw(A); d = A[-1, 0, :2] - A[0, 0, :2]; f = np.array([math.cos(y[0]), math.sin(y[0])]); l = np.array([-f[1], f[0]])
    return float(d @ f), float(d @ l), float(np.degrees(y[-1] - y[0]))


# ----------------------------------------------------------------- bodies
def retarget(A, bone):
    """Joints (...,22,3) -> the same pose with bone lengths `bone` (22,): every bone keeps its direction, the pelvis stays."""
    A = np.asarray(A, float); out = A.copy()
    for i in range(1, 22):
        p = PARENTS[i]; d = A[..., i, :] - A[..., p, :]
        out[..., i, :] = out[..., p, :] + d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9) * bone[i]
    return out


def bones_of(J):
    return np.array([0.0] + [float(np.linalg.norm(J[i] - J[PARENTS[i]])) for i in range(1, 22)])


def sim_bones():
    """Bone lengths of the half-physics body (HumanCLAW's SMPL-X neutral)."""
    from .smpl_fk import sim_rest
    return bones_of(sim_rest())


def to_sim(A, bone=None, feet_from=KIM_FEET, feet_to=SIM_FEET):
    """Segment-frame joints of the generator's body (T,22,3) -> the simulated body: its bones, and per frame moved vertically so
    the lowest foot joint is as high above its standing level as before (feet on the floor stay on the floor)."""
    A = np.asarray(A, float); B = retarget(A, sim_bones() if bone is None else bone)
    dz = (A[:, FEET, 2].min(1) - feet_from) - (B[:, FEET, 2].min(1) - feet_to)
    B[..., 2] += dz[:, None]; return B


def palm_offset():
    """Distance from the wrist to the palm centre the grasp rule uses (midpoint of the wrist and the middle finger's base link)."""
    from .smpl_fk import sim_link
    return float(np.linalg.norm(0.5 * (sim_link("right_wrist") + sim_link("right_middle1")) - sim_link("right_wrist")))


def body_text():
    """The 'Your body' line of the prompt: standing heights of the simulated body."""
    from .smpl_fk import sim_rest
    P = sim_rest(); h = P[:, 1] - P[FEET, 1].min() + SIM_FEET                  # rest pose is Y-up, pelvis at 0
    return (f"Your body (the generated motion is fitted to it; the generator's own body is a little taller): standing heights "
            f"above the floor: pelvis {h[0]:.2f} m, knees {h[4:6].mean():.2f} m, shoulders {h[16:18].mean():.2f} m, head {h[15]:.2f} m; "
            f"the palm centre, which the grasp rule uses, is {palm_offset():.3f} m beyond the wrist.")


def action_from_local(A, yaw, origin):
    """Segment-frame joints of the simulated body (T,22,3, 30 fps) -> HalfPhysicsEnv action at their exact place in the world:
    IK on the HumanML3D-frame joints (its Z-up output is the segment frame turned by +90 deg), then turned by yaw - 90 deg and
    moved to origin."""
    from scipy.spatial.transform import Rotation as R
    from .ik import ik_hml
    from ..world.hp_bridge import SHIFT0
    from .smpl_fk import sim_rest
    m = ik_hml(local_to_hml(A), rest_J=sim_rest()); rz = R.from_euler("z", yaw - math.pi / 2)
    pel = rz.apply(m["transl"]) + origin; go = (rz * R.from_rotvec(m["global_orient"])).as_rotvec()
    bp = np.zeros((len(pel), 54, 3), np.float32); bp[:, :21] = m["body_pose"][:, :21]
    return dict(transl=(pel - SHIFT0).astype(np.float32), global_orient=go.astype(np.float32), body_pose=bp, fps=FPS), m["fit_mm"]


def kimodo_prev(jr, go, bp, yaw, origin):
    """The executed end of the last segment as Kimodo's input for continuing from it: per frame the root rotation (axis-angle,
    Kimodo frame of the new segment), the 21 local body rotations (SMPL-X, as Kimodo's), the pelvis (Kimodo frame) and the
    lowest foot joint's height above the simulated body's standing level (the server puts Kimodo's body on the floor with it).
    jr (n,22,3) Z-up world joints, go (n,3), bp (n,>=21,3) executed SMPL-X axis-angles."""
    from scipy.spatial.transform import Rotation as R
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.]])                    # segment -> Kimodo axes
    Rr = P @ _rz(-yaw) @ R.from_rotvec(np.asarray(go, float)).as_matrix()
    L = to_local(jr, yaw, origin)
    return dict(root_orient=R.from_matrix(Rr).as_rotvec().tolist(), body_pose=np.asarray(bp, float)[:, :21].tolist(),
                pelvis=seg_to_kim(L[:, 0]).tolist(), feet=(L[:, FEET, 2].min(1) - SIM_FEET).tolist())
