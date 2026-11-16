"""Kimodo (Rempe et al., NVIDIA, arXiv 2603.15546, github.com/nv-tlabs/kimodo, code Apache-2.0) behind gen_server.py, with its
native inputs: the agent reads Kimodo's own documentation and gives Kimodo's own inputs; nothing is filled in for it.

Model: Kimodo-SMPLX-RP-v1 (NVIDIA R&D model licence, research use only) from $KIMODO_WEIGHTS (default <data>/kimodo); text
encoder LLM2Vec from the Hugging Face cache $KIMODO_WEIGHTS/hf_hub (offline). Kimodo's repository is used unmodified
($KIMODO_ROOT on PYTHONPATH; see inhabitant/generator/__init__.py and scripts/run_generator.sh).

Request (generate):
  text: str, or a list of str (Kimodo's multi-prompt generation: one after another, each continuing from the one before)
  seconds: float, or a list (one per text); the segment is their sum (+ 1 frame), 30 fps
  constraints: Kimodo's constraint JSON list (types root2d, fullbody, left-hand, right-hand, left-foot, right-foot, end-effector),
               in Kimodo's frame of this segment (origin on the floor below the pelvis at the start, +Z = the facing at the start,
               Y up, X left), frame_indices counted from the start of this segment
  seed (default 0), post (Kimodo's post-processing, default True), cfg ([text, constraint] weights, default [2, 2])
  prev: the executed end of the last segment (motion_exec.kimodo_prev; None before the first): its n frames become the first n
        frames of the generation as a full-body constraint (as Kimodo's own multi-prompt continuation does with the last frames
        of the previous prompt), first_heading_angle is their heading, and the n - 1 frames before the segment start are cut off
        again, so frame 0 of the result is the actual pose.
Output: joints (T,22,3) Kimodo's body in the segment frame (x forward, y left, z up), and the native motion of the segment
(rot: (T,22,3) local axis-angles as Kimodo's constraint JSON; root: (T,3) pelvis, Kimodo frame) for inspect pose."""
import os, time
import numpy as np

from ..paths import KIMODO_WEIGHTS as DATA

MODEL_NAME = "Kimodo-SMPLX-RP-v1"
FPS, MAX_PROMPT_S = 30, 10.0
KIM_FEET = 0.007                                     # = inhabitant.motion.motion_exec.KIM_FEET
FEET = [7, 8, 10, 11]


def kim_to_seg(j):
    j = np.asarray(j, float); return np.stack([j[..., 2], j[..., 0], j[..., 1]], -1)


def _env():
    os.environ.setdefault("CHECKPOINT_DIR", str(DATA))
    os.environ.setdefault("HF_HUB_CACHE", str(DATA / "hf_hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TEXT_ENCODER_MODE", "local")


def _shift_xz(c, g):
    """A constraint dict moved horizontally by -g (Kimodo x, z) and later by k frames (done by the caller)."""
    c = dict(c)
    for key in ("smooth_root_2d",):
        if key in c:
            c[key] = (np.asarray(c[key], float) - g).tolist()
    if "root_positions" in c:
        r = np.asarray(c["root_positions"], float); r[..., [0, 2]] -= g; c["root_positions"] = r.tolist()
    return c


class Kimodo:
    def __init__(self, device="cuda", text_device=None, steps=100, cfg=(2.0, 2.0)):
        _env()
        import torch
        if text_device is None:                                # LLM2Vec 8B in bf16 needs ~15 GB; on small cards on the CPU
            text_device = os.environ.get("TEXT_ENCODER_DEVICE")
            if text_device is None and device != "cpu" and torch.cuda.is_available():
                text_device = "cuda" if torch.cuda.get_device_properties(0).total_memory > 30e9 else "cpu"
        if text_device:
            os.environ["TEXT_ENCODER_DEVICE"] = text_device
        from kimodo import load_model
        from kimodo import constraints as kc
        self.torch, self.kc, self.steps, self.cfg = torch, kc, steps, list(cfg)
        self.model = load_model(MODEL_NAME, device=device, default_family="Kimodo")
        self.skel = self.model.skeleton
        assert self.model.fps == FPS, self.model.fps
        self._drop_empty_crops()

    def _drop_empty_crops(self):
        """Kimodo's multi-prompt generation crops every constraint to each prompt's frames; a constraint with no frame in a prompt
        then makes its condition builder / post-processing fail ("cannot reshape tensor of 0 elements").
        Run-time wrap (the repo stays unmodified): constraints without frames are left out there."""
        import kimodo.model.kimodo_model as km
        dev = self.model.device

        def keep(cl):
            # a cropped hand / foot / end-effector constraint is a new object whose joint index tensors are on the CPU
            # ("Expected all tensors to be on the same device" with multi-prompt text + such a constraint): moved to the model's device
            out = [c for c in cl if isinstance(c, list) or len(getattr(c, "frame_indices", [1])) > 0]
            for c in out:
                if not isinstance(c, list) and hasattr(c, "to"):
                    c.to(device=dev)
            return out
        mr = self.model.motion_rep; orig = mr.create_conditions_from_constraints
        mr.create_conditions_from_constraints = lambda cl, *a, **k: orig(keep(cl or []), *a, **k)
        pp = km.post_process_motion

        def post(*a, **k):
            a = list(a)
            if len(a) > 4 and a[4]:
                a[4] = [keep(x) for x in a[4]] if isinstance(a[4][0], list) else keep(a[4])
            return pp(*a, **k)
        km.post_process_motion = post

    def _prev_pose(self, prev):
        """prev (motion_exec.kimodo_prev) -> posed joints (n,22,3) and global rotations (n,22,3,3) of Kimodo's body, Kimodo frame,
        each frame moved vertically so its lowest foot joint is prev['feet'] above Kimodo's standing level."""
        torch = self.torch
        from scipy.spatial.transform import Rotation as R
        ro = np.asarray(prev["root_orient"], float); bp = np.asarray(prev["body_pose"], float); n = len(ro)
        rm = np.concatenate([R.from_rotvec(ro).as_matrix()[:, None], R.from_rotvec(bp.reshape(-1, 3)).as_matrix().reshape(n, 21, 3, 3)], 1)
        root = np.asarray(prev["pelvis"], float)
        gr, pj, _ = self.skel.fk(torch.tensor(rm, dtype=torch.float32, device=self.model.device),
                                 torch.tensor(root, dtype=torch.float32, device=self.model.device))
        pj = pj.cpu().numpy(); gr = gr.cpu().numpy()
        dy = (np.asarray(prev["feet"], float) + KIM_FEET) - pj[:, FEET, 1].min(1)
        pj[..., 1] += dy[:, None]
        return pj, gr

    def fk(self, root_positions, local_joints_rot, **_):
        """Joint positions (Kimodo's body, Kimodo frame) of poses given as in its constraint JSON: root_positions (F,3) and
        local_joints_rot (F,22,3) axis-angles (inspect fk)."""
        torch = self.torch
        from kimodo.geometry import axis_angle_to_matrix
        rot = torch.tensor(np.asarray(local_joints_rot, float).reshape(-1, 22, 3), dtype=torch.float32, device=self.model.device)
        root = torch.tensor(np.asarray(root_positions, float).reshape(-1, 3), dtype=torch.float32, device=self.model.device)
        if len(rot) != len(root):
            raise ValueError(f"root_positions has {len(root)} frames, local_joints_rot {len(rot)}")
        _, pj, _ = self.skel.fk(axis_angle_to_matrix(rot), root)
        return dict(joints=pj.cpu().numpy().round(4).tolist())

    def native(self, prev, **_):
        """Executed frames (motion_exec.kimodo_prev) as Kimodo's native pose on its own body (feet kept on the floor):
        root_positions, local_joints_rot and the heading angle (inspect pose)."""
        torch = self.torch
        pj, _ = self._prev_pose(prev)
        rot = np.concatenate([np.asarray(prev["root_orient"], float)[:, None], np.asarray(prev["body_pose"], float)], 1)
        from kimodo.motion_rep.feature_utils import compute_heading_angle
        h = compute_heading_angle(torch.tensor(pj[None], dtype=torch.float32), self.skel)[0].numpy()
        return dict(root_positions=pj[:, 0].round(4).tolist(), local_joints_rot=rot.round(4).tolist(), heading=h.round(4).tolist())

    def generate(self, text, seconds, constraints=None, seed=0, post=True, cfg=None, steps=None, prev=None, **_):
        torch, kc, dev = self.torch, self.kc, self.model.device
        texts = [str(t) for t in text] if isinstance(text, (list, tuple)) else [str(text)]
        secs = [float(s) for s in seconds] if isinstance(seconds, (list, tuple)) else [float(seconds)]
        if len(secs) == 1 and len(texts) > 1:
            secs = [secs[0] / len(texts)] * len(texts)
        if len(secs) != len(texts):
            raise ValueError(f"seconds: give one length per text ({len(texts)} texts, {len(secs)} lengths)")
        if any(s > MAX_PROMPT_S for s in secs):
            raise ValueError(f"each text can last at most {MAX_PROMPT_S:.0f} s")
        nf = [max(1, int(round(s * FPS))) for s in secs]; nf[0] += 1          # frames t = 0 .. sum(seconds)
        T = sum(nf)
        g, k, cl, fha = np.zeros(2), 0, [], None
        if prev:
            pj, gr = self._prev_pose(prev); n = len(pj); k = n - 1; g = pj[0, 0, [0, 2]].copy()
            pj[..., 0] -= g[0]; pj[..., 2] -= g[1]
            pt = torch.tensor(pj, dtype=torch.float32, device=dev)
            cl.append(kc.FullBodyConstraintSet(self.skel, torch.arange(n, device=dev), pt, torch.tensor(gr, dtype=torch.float32, device=dev)))
            from kimodo.motion_rep.feature_utils import compute_heading_angle
            fha = compute_heading_angle(pt[None], self.skel)[:, 0]
        if constraints:
            cs = []
            for c in constraints:
                if not isinstance(c, dict) or "type" not in c or "frame_indices" not in c:
                    raise ValueError(f"each constraint needs 'type' and 'frame_indices': {str(c)[:200]}")
                fi = [int(f) for f in c["frame_indices"]]
                if any(f < 0 or f >= T for f in fi):
                    raise ValueError(f"{c['type']}: frame_indices must be within 0..{T - 1} (this segment)")
                c = _shift_xz(c, g); c["frame_indices"] = [f + k for f in fi]; cs.append(c)
            cl += kc.load_constraints_lst(cs, self.skel, device=dev)
        nf[0] += k
        if seed is not None:
            from kimodo.tools import seed_everything
            seed_everything(int(seed))
        t0 = time.time()
        kw = dict(num_denoising_steps=int(steps or self.steps), constraint_lst=cl, cfg_weight=list(cfg or self.cfg), cfg_type="separated",
                  num_samples=1, post_processing=bool(post), return_numpy=False, progress_bar=lambda x, **_k: x)
        if fha is not None:
            kw["first_heading_angle"] = fha
        if len(texts) > 1:
            out = self.model(texts, nf, multi_prompt=True, **kw)
        else:
            out = self.model(texts[0], nf[0], **kw)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        gen_s = time.time() - t0
        from kimodo.geometry import matrix_to_axis_angle
        lr = out["local_rot_mats"][0][k:]; root = out["root_positions"][0][k:].detach().cpu().numpy().copy()
        pj = out["posed_joints"][0][k:].detach().cpu().numpy().copy()
        for a in (root[..., None, :], pj):
            a[..., 0] += g[0]; a[..., 2] += g[1]
        rot = matrix_to_axis_angle(lr).detach().cpu().numpy()
        J = kim_to_seg(pj)
        return dict(joints=J.astype(np.float32), fps=FPS, frame="segment", gen_s=gen_s, rot=rot.astype(np.float32),
                    root=root.astype(np.float32), notes=[], ignored=[])
