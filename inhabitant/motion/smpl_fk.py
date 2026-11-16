"""Minimal forward kinematics for the 22 body joints (neutral shape, beta = 0).
rest(): SMPL neutral joint centres (pelvis-relative) from HumanCLAW's resources/metrics/smpl_neutral_body22.json (J_regressor @
v_template of SMPL_neutral). The pelvis is placed at `transl`.
sim_rest(): the same 22 joints of the simulator's own body (SMPL-X neutral, beta = 0: the joint origins of HumanCLAW's URDF, all
without rotation, same frame as SMPL: X left, Y up, Z forward). With it, fk22 gives the real joint positions of the simulated
body; its bones differ from SMPL's by 1-3 cm (shoulder width 0.32 vs 0.35 m)."""
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

from ..paths import HUMANCLAW_ROOT

HC = HUMANCLAW_ROOT / "resources"
BODY22 = HC / "metrics/smpl_neutral_body22.json"
URDF = HC / "agent/neutral_beta0_handmerged/neutral_beta0_handmerged.urdf"
NAMES = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
         "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_REST = None
_SIM = {}


def rest():
    global _REST
    if _REST is None:
        _REST = np.asarray(json.load(open(BODY22))["positions_m"], np.float64)
    return _REST


def _urdf_links():
    if not _SIM:
        for jt in ET.parse(URDF).getroot().findall("joint"):
            _SIM[jt.find("child").get("link")] = (jt.find("parent").get("link"), np.array([float(v) for v in jt.find("origin").get("xyz").split()]))
    return _SIM


def sim_link(name):
    """Rest position of any URDF link (pelvis = 0)."""
    L = _urdf_links()
    return np.zeros(3) if name not in L else sim_link(L[name][0]) + L[name][1]


def sim_rest():
    """(22,3) rest joint centres of the simulator's SMPL-X neutral body, pelvis at 0."""
    return np.array([sim_link(n) for n in NAMES])


def fk22(transl, global_orient, body_pose, rest_J=None):
    """transl (T,3), global_orient (T,3) axis-angle, body_pose (T,>=21,3) axis-angle (SMPL / SMPL-X body order) -> (T,22,3).
    rest_J: rest joint centres (default SMPL neutral; sim_rest() for the simulator's body)."""
    T = len(transl); J = rest() if rest_J is None else rest_J
    off = np.array([J[i] - (J[PARENTS[i]] if PARENTS[i] >= 0 else 0) for i in range(22)])
    rots = np.concatenate([np.asarray(global_orient)[:, None], np.asarray(body_pose)[:, :21]], 1)    # (T,22,3)
    Rl = R.from_rotvec(rots.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    G = np.zeros((T, 22, 3, 3)); P = np.zeros((T, 22, 3))
    for i in range(22):
        p = PARENTS[i]
        if p < 0:
            G[:, i] = Rl[:, i]; P[:, i] = transl
        else:
            G[:, i] = G[:, p] @ Rl[:, i]; P[:, i] = P[:, p] + np.einsum("tij,j->ti", G[:, p], off[i])
    return P
