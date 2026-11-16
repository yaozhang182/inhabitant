"""Joints -> SMPL rotations (optimisation IK), so joint positions can drive the half-physics
SMPL-X humanoid.

The SMPL canonical frame (Y up, Z forward, X left) equals the HumanML3D frame, so IK runs on HumanML3D-frame joints with a rest
skeleton: SMPL neutral by default (smpl_fk.rest), the simulated SMPL-X body with rest_J=smpl_fk.sim_rest() (what the agent
pipeline uses). Output is converted to the AMASS / half-physics Z-up frame (X right, Y forward, Z up):
p_zup = W p_hml, global_orient_zup = W * R_hml, with W = [[-1,0,0],[0,0,1],[0,1,0]] (a proper rotation).
Fit: Adam on per-frame axis-angles of all 22 joints + pelvis translation; loss = joint MSE + small pose prior + temporal smoothing.
Twist of end segments (wrists, head, feet) is unobserved by 22 joints and stays near zero."""
import numpy as np
import torch

from .smpl_fk import PARENTS, rest

W = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)


def _aa_to_mat(aa):
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8); k = aa / th
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + s * K + (1 - c) * K @ K


def fk_torch(aa, transl, off):
    """aa (T,22,3), transl (T,3), off (22,3) parent offsets -> (T,22,3)."""
    Rl = _aa_to_mat(aa); G, P = [None] * 22, [None] * 22
    for i in range(22):
        p = PARENTS[i]
        if p < 0:
            G[i], P[i] = Rl[:, i], transl
        else:
            G[i] = G[p] @ Rl[:, i]; P[i] = P[p] + (G[p] @ off[i])
    return torch.stack(P, 1)


def ik_hml(joints, iters=400, lr=0.03, w_prior=1e-3, w_smooth=1e-1, rest_J=None):
    """joints (T,22,3) HumanML3D frame (metres) -> dict(transl, global_orient (T,3), body_pose (T,21,3)) in the Z-up frame,
    plus fit_mm (mean joint error after the fit, mm). rest_J: the skeleton to fit (default SMPL neutral; smpl_fk.sim_rest() for
    the simulator's SMPL-X body)."""
    J = torch.tensor(np.asarray(joints, np.float64)); T = len(J)
    Jr = rest() if rest_J is None else np.asarray(rest_J, np.float64); off = torch.tensor(np.array([Jr[i] - (Jr[PARENTS[i]] if PARENTS[i] >= 0 else 0) for i in range(22)]))
    # initial global orientation from the hip line and the spine (facing = left_hip - right_hip x up)
    across = (J[:, 1] - J[:, 2]); up = (J[:, 9] - J[:, 0]); fwd = torch.linalg.cross(across, up)
    x = across / across.norm(dim=-1, keepdim=True); z = fwd / fwd.norm(dim=-1, keepdim=True); y = torch.linalg.cross(z, x)
    from scipy.spatial.transform import Rotation as R
    go0 = R.from_matrix(torch.stack([x, y, z], -1).numpy()).as_rotvec()
    aa = torch.zeros(T, 22, 3, dtype=torch.float64); aa[:, 0] = torch.tensor(go0)
    aa.requires_grad_(True); tr = J[:, 0].clone().requires_grad_(True)
    opt = torch.optim.Adam([aa, tr], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        P = fk_torch(aa, tr, off)
        loss = ((P - J) ** 2).sum(-1).mean() + w_prior * (aa[:, 1:] ** 2).sum(-1).mean() * 1e-2
        if T > 1:
            loss = loss + w_smooth * ((aa[1:] - aa[:-1]) ** 2).sum(-1).mean() * 1e-2
        loss.backward(); opt.step()
    with torch.no_grad():
        P = fk_torch(aa, tr, off); fit = float((P - J).norm(dim=-1).mean() * 1000)
    a = aa.detach().numpy(); t = tr.detach().numpy()
    go = (R.from_matrix(W) * R.from_rotvec(a[:, 0])).as_rotvec()
    return dict(transl=(t @ W.T).astype(np.float32), global_orient=go.astype(np.float32), body_pose=a[:, 1:].astype(np.float32),
                fit_mm=fit)
