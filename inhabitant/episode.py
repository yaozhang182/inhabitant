"""One episode: a household task in an apartment, carried out by a person whose body the agent moves.

`Episode` builds the world of a task (inhabitant.world), holds the five tools the agent acts through and keeps the snapshots and
the logs:
  generate_segment  Kimodo makes one segment from the agent's inputs, continuing from the executed end of the last committed
                    segment; the motion is fitted to the simulated body, placed with inverse kinematics and executed in the
                    half-physics simulator with the grasp / release rules. Returns whether the goal is satisfied and eight frames
                    of the video. Every try starts from the last committed state.
  inspect           read-only measurements of the world and of the tries (where, distance, contacts, extent, pose, fk)
  commit            keeps the last try; the episode ends when a committed segment completes the task
  get_state         the full state text
  retry_from        undo: back to the start of a committed segment
The model is given no rule-written feedback: it measures what it needs with inspect. `feedback()` only writes a summary of each
try into the log, which the evaluation reads.

Outputs in the run directory: log.json (messages, tries, program; best = the program that reached the goal), tries/t<n>.mp4
(follow camera | side camera), t<n>_sheet.jpg (the eight frames the model sees), t<n>_joints.npz, t<n>_world.npz."""
import copy, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

from .paths import REPO, RUNS, BENCHMARK, EB_META
from .generator import kimodo_env
from .prompt import build_prompt
from .world.eb_world import EBWorld, HANDS, category, eb_name
from .world.hp_bridge import executed_joints, action_joints, floor_point, heading, terrain_offsets, W
from .world.agent_io import SceneText, to_map, heading_map, short, art_state
from .world.bench_goal import BenchGoal
from .motion.motion_metrics import motion_summary
from .motion.motion_exec import (to_local, action_from_local, local_root_motion, sim_bones, to_sim, kimodo_prev, seg_to_kim,
                                 kim_to_seg, palm_offset, FPS, MAX_T, JOINTS)
from .motion.smpl_fk import sim_rest

FEET_J = [7, 8, 10, 11]                             # ankles and toes
FOOT = ("foot", "ankle", "toe")
NEAR = 2.0                                         # m: commit() lists the places / objects / handles within this distance
N_CONT = 5                                         # frames of the executed end of the last segment that Kimodo continues from


def _fill(content, state):
    """Put the state text into a message's {STATE} placeholder (content: a string or a list of parts)."""
    if isinstance(content, list):
        return [dict(p, text=p["text"].replace("{STATE}", state)) if p["type"] == "text" else p for p in content]
    return content.replace("{STATE}", state)


def seg_frame(pel, f):
    """Segment frame at a state: (yaw of the facing, origin on the floor below the pelvis), Z-up world."""
    fz = W @ f; return float(np.arctan2(fz[1], fz[0])), W @ pel


def contact_sheet(video, out, n=8):
    """8 frames of a try video in one image (2 per row, time stamps), 768 px wide, JPEG quality 80, ~0.11 MB (small enough that
    a conversation carrying many of them stays within the APIs' request size limits)."""
    import imageio
    from PIL import Image, ImageDraw
    rd = imageio.get_reader(video); m = rd.count_frames()
    idx = np.linspace(0, m - 1, n).round().astype(int)
    tiles = []
    for i in idx:
        im = Image.fromarray(rd.get_data(int(i))); im = im.resize((im.width // 2, im.height // 2))
        ImageDraw.Draw(im).text((4, 2), f"t = {i / 30:.1f} s", fill=(255, 255, 0)); tiles.append(np.asarray(im))
    rows = [np.concatenate(tiles[r:r + 2], 1) for r in range(0, n, 2)]
    im = Image.fromarray(np.concatenate(rows, 0)).convert("RGB")
    im.resize((768, round(im.height * 768 / im.width)), Image.LANCZOS).save(out, "JPEG", quality=80)


class GenClient:
    """The generator service (inhabitant.generator.gen_server) as a subprocess, file RPC. With GEN_RPC_SHARED=<dir> set, a
    running `gen_server --shared --rpc <dir>` is used instead (one model for several episodes; this client gets the
    subdirectory named after the run)."""
    def __init__(self, rpc, log, timeout=900):
        shared = os.environ.get("GEN_RPC_SHARED")
        if shared:
            rpc = Path(shared) / "_".join(Path(rpc).parent.parts[-2:])
        self.rpc = Path(rpc); self.rpc.mkdir(parents=True, exist_ok=True); self.n = 0
        for f in self.rpc.glob("*"):
            f.unlink()
        if shared:
            self.proc = None; t0 = time.time()
            while not (Path(shared) / "READY").exists():
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"shared generator {shared} not ready")
                time.sleep(0.5)
            return
        self.proc = subprocess.Popen([sys.executable, "-m", "inhabitant.generator.gen_server", "--rpc", str(self.rpc)],
                                     cwd=str(REPO), env=kimodo_env(), stdin=subprocess.DEVNULL, stdout=open(log, "w"),
                                     stderr=subprocess.STDOUT)
        t0 = time.time()
        while not (self.rpc / "READY").exists():
            if self.proc.poll() is not None or time.time() - t0 > timeout:
                raise RuntimeError(f"generator did not start; see {log}")
            time.sleep(0.5)

    def call(self, **req):
        self.n += 1; p = self.rpc / f"req_{self.n:05d}.json"; tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(req)); tmp.rename(p)
        r = self.rpc / f"resp_{self.n:05d}.json"
        while not r.exists():
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("generator exited")
            time.sleep(0.05)
        out = json.loads(r.read_text()); p.unlink(missing_ok=True); r.unlink(missing_ok=True)
        if "error" in out:
            raise RuntimeError(f"generator: {out['error']}")
        return out

    def close(self):
        try:
            self.call(cmd="quit")
        except Exception:  # noqa: BLE001
            pass


class Episode:
    def __init__(self, a):
        """a: a namespace with `task` (an id of benchmark/inhabit) or `spec` (a task file), and `model`, `tag`, `max_calls`;
        optional `out` (the run directory; default <runs>/<model>/<tag>/<set>_<id>)."""
        if getattr(a, "spec", None):                 # a task file anywhere (same format as benchmark/inhabit/*.json)
            self.spec = json.load(open(a.spec))
            a.eval_set, a.id = "spec", str(self.spec.get("id") or Path(a.spec).stem)
        else:
            a.eval_set, a.id = "inhabit", a.task
            self.spec = json.load(open(BENCHMARK / f"{a.id}.json"))
        self.a = a
        self.out = Path(a.out).resolve() if getattr(a, "out", None) else RUNS / a.model / a.tag / f"{a.eval_set}_{a.id}"
        (self.out / "tries").mkdir(parents=True, exist_ok=True)
        base = self.spec["base"]                     # the EB-Habitat episode whose scene the task starts from
        self.w = EBWorld(base["set"], base["id"], video=True, third_res=(480, 640), contacts=True,
                         extra=self.spec.get("add_objects"))       # ReplicaCAD household objects added by the task
        self.setup_bench()
        self.st = SceneText(self.w)
        self.end_reason = ""
        self.tool_log, self.usage = [], dict(input_tokens=0, cached_tokens=0, cache_write_tokens=0, output_tokens=0, cost_usd=0.0)
        self.system, self.tools = build_prompt()
        self.rest = sim_rest(); self.bones = sim_bones()                              # the simulated body
        rs = self.spec.get("start") or json.load(open(EB_META / base["set"] / f"{base['id']}.json"))["robot_start"]
        rot = rs["base_rot"]
        pel0 = floor_point(self.w.sim, np.array(rs["base_pos"])); f0 = np.array([np.cos(rot), 0.0, -np.sin(rot)])  # EB's robot faces +x
        self.floor = float(pel0[1]); self.st.floor_y = self.floor
        self.surfaces = {short(k): k for k in self.w.receptacles}
        self.bench = BenchGoal(self.spec, self.w, self.floor, self._eb_goal); self._bench_ok = False
        self.snaps = [dict(world=self.w.snapshot(human=True), pel=pel0, f=f0, started=False, jnow=None)]
        self.load(self.snaps[0])
        self.program, self.versions, self.tries, self.pending = [], [], [], None
        self.n_since_commit = 0; self.done = False
        self.calls = 0; self.msgs = []; self.cam_eye = None
        self.gen = None                              # the generator service, started with the first generate_segment

    # ----------------------------------------------------------------- the task
    def setup_bench(self):
        """Apply the task on top of its base EB-Habitat episode: the instruction, objects moved (habitat position), joint
        states of handles. The goal is judged by BenchGoal (world/bench_goal.py)."""
        import magnum as mn
        sp, w = self.spec, self.w
        w.ep["instruction"] = sp["instruction"]
        if not sp.get("eb_original"):
            w.on_top_tol = 0.06                       # "on a surface" also 0.06 m below its region (the sofa's starts above the cushion)
        for o, p in (sp.get("objects") or {}).items():
            ob = w.rom.get_object_by_handle(o); ob.translation = mn.Vector3(*map(float, p))
            ob.linear_velocity = ob.angular_velocity = mn.Vector3(0, 0, 0)
        for m, q in (sp.get("joints") or {}).items():
            w.set_marker_js(m, float(q))
        if sp.get("objects"):
            w.settle(1.0, park_human=True)                                    # moved objects come to rest
        self._eb_goal = w.goal_satisfied                                      # EB's own goal (condition kind "eb")

    def goal_now(self, out):
        """Goal satisfied after this try: BenchGoal over the committed program + this try (result also in out['bench'])."""
        seg = lambda t: dict(jr=t["_jr"], contacts=t["_contacts"], cframes=t["_cframes"], held=t["_held"], objs=[f["obj"] for f in t["_trace"]],
                             js=[f["js"] for f in t["_trace"]])
        r = self.bench.evaluate([seg(self.tries[p["try_n"] - 1]) for p in self.program] + [seg(out)])
        out["bench"] = r; self._bench_ok = r["ok"]; self._bench_last = r; return r["ok"]

    @staticmethod
    def cond_text(c):
        """one condition as text, e.g. kick(055_baseball_:0000, Tbl1_Top1, under=True)"""
        args = [short(v) if isinstance(v, str) else f"{k}={v}" for k, v in c.items() if k not in ("id", "kind")]
        return f"{c['kind']}(" + ", ".join(args) + ")"

    def goal_text(self):
        """The goal as text (for the log): the task's conditions and their order."""
        sp = self.spec; t = " AND ".join(self.cond_text(c) for c in sp["conditions"])
        return t + ("; order: " + ", ".join(" before ".join(o) for o in sp["order"]) if sp.get("order") else "")

    # ----------------------------------------------------------------- state
    def save(self):
        return dict(world=self.w.snapshot(human=True), pel=self.pel.copy(), f=self.f.copy(), started=self.started,
                    jnow=self.jnow)

    def load(self, s):
        self.w.restore(s["world"]); self.pel, self.f, self.started = s["pel"].copy(), s["f"].copy(), s["started"]
        self.jnow = s.get("jnow")             # executed joints (22,3 Z-up) at the end of the last try of this state; None before any

    def held(self):
        return {h: (v["target"] if v else None) for h, v in self.w.held.items()}

    def state_text(self):
        m = self.st.ascii_map([("^", self.pel + 0.25 * self.f), ("H", self.pel)])
        return (f"## Top-down map of the floor (cell {self.st.cell} m; '.' free floor, '#' furniture / wall / unreachable; H = you, "
                f"^ = one cell ahead of you)\n{m}\n\n{self.st.state(self.pel, self.f, self.held())}")

    def near_state(self, r=NEAR):
        """The state without the map, with only the places / objects / handles within r m (and what you hold)."""
        out = []
        for sec in self.state_text().split("\n\n## ")[1:]:
            head, *rows = sec.split("\n")
            if head.startswith("Top-down map"):
                continue
            if rows and rows[0].startswith("- ") and "from you:" in sec:
                rows = [x for x in rows if "in your hand" in x or float(re.search(r"from you: ([\d.]+) m", x).group(1)) <= r]
                head += f" within {r:.1f} m of you"
            out.append("## " + head + "\n" + ("\n".join(rows) or "(none)"))
        return "\n\n".join(out)

    # ----------------------------------------------------------------- body
    def ground_rise(self, j, dead=0.05):
        """How much the ground under the lowest foot of pose j (22,3 Z-up) is above the episode's start floor (0 below `dead`)."""
        if not hasattr(self, "nav0"):
            self.nav0 = float(self.ground([(W @ self.snaps[0]["pel"])[:2]])[0])
        r = float(self.ground([j[FEET_J[int(np.argmin(j[FEET_J, 2]))], :2]])[0]) - self.nav0
        return r if abs(r) >= dead else 0.0

    def ground(self, xy):
        """Navmesh height (Z-up) under Z-up (x, y) points: walkable floor and stairs; furniture and the body itself do not count."""
        import magnum as mn
        pf = self.w.sim.pathfinder; out = []
        for x, y in np.asarray(xy, float):
            p = np.asarray(pf.snap_point(mn.Vector3(-x, self.pel[1], y)), float)
            out.append(p[1] if np.isfinite(p[1]) else np.nan)
        out = np.asarray(out); return np.where(np.isnan(out), np.nanmedian(out) if np.isfinite(out).any() else 0.0, out)

    def target_point(self, ev, p_end):
        """Side camera target: the first event's object / handle / surface, else the planned end pelvis `p_end` (1.5 m ahead of
        the pelvis if it hardly moves)."""
        for e in ev:
            if e["kind"] == "grasp":
                return self.w.marker_pos(e["target"]) if e["target"] in self.w.markers else self.w.obj_pos(e["target"])
            if e["surface"] in self.w.receptacles:
                return self.w.recep_centre(e["surface"])
            if e["surface"] in self.w.objs:
                return np.asarray(self.w.obj_pos(e["surface"]))
        p, f = self.w.pelvis(); d = p_end - p; d[1] = 0
        return p_end if np.linalg.norm(d) > 0.3 else p + 1.5 * f

    # ----------------------------------------------------------------- cameras
    def free(self, a, b):
        """Distance from a towards b before the first hit that is not the body itself (|b - a| if nothing is hit)."""
        import magnum as mn
        ag = self.w.env.agent; own = {ag.object_id, *ag.link_object_ids.keys()}
        v = np.asarray(b, float) - np.asarray(a, float)
        h = self.w.sim.cast_ray(self.w.hs.geo.Ray(mn.Vector3(*map(float, a)), mn.Vector3(*map(float, v))), max_distance=1.0)
        hit = next((x for x in h.hits if x.object_id not in own), None)
        return float(hit.ray_distance if hit is not None else 1.0) * float(np.linalg.norm(v))

    def pick_eye(self, at, dirs, dist, hs, looks):
        """Camera position among directions `dirs` [(horizontal unit vector, penalty)] and heights `hs` around `at`, pulled in
        0.3 m in front of the first hit; scored by how many of `looks` it sees (first hit within 0.35 m of the point), then by
        its distance from `at`, minus the penalty (+0.3 for the higher heights). -> (eye, index of the direction, height)."""
        best = None
        for i, (u, pen) in enumerate(dirs):
            for j, h in enumerate(hs):
                want = at + u * dist + np.array([0, h, 0]); L = float(np.linalg.norm(want - at))
                eye = at + (want - at) * max(self.free(at, want) - 0.3, 0.3 * L) / L
                vis = sum(self.free(eye, q) >= np.linalg.norm(q - eye) - 0.35 for q in looks)
                sc = 10 * vis + min(float(np.linalg.norm(eye - at)), dist) - pen - 0.3 * j
                if best is None or sc > best[0]:
                    best = (sc, eye, i, h)
        return best[1], best[2], best[3]

    @staticmethod
    def rot(u, deg):
        """Rotate a habitat XZ direction by deg (+ = to the left, i.e. counter-clockwise on the map)."""
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([c * u[0] + s * u[2], 0.0, -s * u[0] + c * u[2]])

    def side_eye(self, target, p_end):
        """Side camera for a segment: looks at the midpoint of the pelvis and the target from roughly perpendicular to the line
        between them, far enough to hold both; the direction with free space and a view of the pelvis now, the planned end
        pelvis `p_end` and the target is taken (the pelvis is the body's actual one, after the first segment's reset)."""
        p = self.w.pelvis()[0]; g = np.array(target, float); d = g - p; d[1] = 0; L = max(float(np.linalg.norm(d)), 0.3)
        u = d / L; mid = 0.5 * (p + g); mid[1] = self.floor + float(np.clip(0.5 * (g[1] - self.floor + 1.0), 0.6, 1.2))
        dirs = [(self.rot(u, a), abs(abs(a) - 90) / 60) for a in (90, -90, 60, -60, 120, -120, 150, -150)]
        return self.pick_eye(mid, dirs, max(2.2, L + 1.6), (0.5, 1.2, 1.9), [p, np.asarray(p_end, float), g])[0], mid

    FOLLOW = (22, -22, 0, 60, -60, 90, -90, 135, -135, 180)   # camera direction from behind (deg, + = behind-left)
    FOLLOW_HD = ((0.7, 2.6), (1.3, 1.9))                      # (height above the look-at point, horizontal distance)
    FOLLOW_MIN = 1.3                                          # a view is free if the camera can be this far from the body

    def follow_setup(self):
        self.follow_cur = None; self.cam_eye = None

    def follow_cam(self):
        """Camera following the body, chosen every frame among directions FOLLOW (relative to the facing) and FOLLOW_HD: the
        current view while it stays free, else the free view nearest behind-left (else the one reaching farthest). Pulled in 0.3 m
        in front of the first hit; a change of view is a cut, not a sweep through furniture."""
        p, f = self.w.pelvis(); at = p + np.array([0, 0.2, 0]); views = {}
        for a in self.FOLLOW:
            for j, (h, d) in enumerate(self.FOLLOW_HD):
                want = at + self.rot(-f, -a) * d + np.array([0, h, 0]); L = float(np.linalg.norm(want - at))
                r = min(max(self.free(at, want) - 0.3, 0.2), L); views[a, j] = (at + (want - at) * r / L, r, abs(a - 22) / 90 + 0.3 * j)
        cur = self.follow_cur
        if cur is None or views[cur][1] < self.FOLLOW_MIN:
            ok = [k for k, v in views.items() if v[1] >= self.FOLLOW_MIN]
            new = min(ok, key=lambda k: views[k][2]) if ok else max(views, key=lambda k: views[k][1])
            if new != cur:
                self.follow_cur = new; self.cam_eye = None
        want = views[self.follow_cur][0]
        self.cam_eye = want if self.cam_eye is None else 0.85 * self.cam_eye + 0.15 * want
        return self.w.render_at(self.cam_eye, at)

    def sink(self, writer, side):
        def f(_ego, _third):
            img = np.concatenate([self.follow_cam()[:, 80:560], self.w.render_at(*side)[:, 80:560]], 1)
            writer.append_data(np.asarray(img, np.uint8))
        return f

    def execute(self, ans):
        import imageio
        k = len(self.program) + 1
        J = ans["_J"]; act, fit = action_from_local(J, *seg_frame(self.pel, self.f))
        act["transl"][:, 2] += terrain_offsets(action_joints(act, self.rest), self.ground)      # uneven ground; 0 on a flat floor
        fw, le, tu = ans["root"]; sel = dict(chosen=0, ik_fit_mm=round(fit, 1), cands=[dict(fwd=fw, left=le, turn=tu)])
        if not self.started:
            self.w.env.reset(initial_transl=act["transl"][0], initial_global_orient=act["global_orient"][0],
                             initial_body_pose=act["body_pose"][0]); self.started = True
        held = self.held()
        before = dict(held=held, objs={o: (self.w.obj_pos(o), self.place_of(o, held)) for o in self.w.objs},
                      js={m: self.w.marker_js(m) for m in self.w.markers})
        p_end = W @ action_joints(act, self.rest)[-1, 0]                # planned end pelvis (habitat)
        side = self.side_eye(self.target_point(ans["events"], p_end), p_end); self.follow_setup()
        vid = self.out / "tries" / f"t{len(self.tries) + 1:03d}.mp4"
        writer = imageio.get_writer(vid, fps=30, quality=6, macro_block_size=1)
        last = (len(act["transl"]) - 1) / 30.0                            # a release written at the very end still happens
        evs = [dict(e, t=min(e["t"], last)) if e["kind"] == "release" else e for e in ans["events"]]
        info = self.w.execute(act, evs, settle_s=1.5 if any(e["kind"] == "release" for e in ans["events"]) else 0.3,
                              sink=self.sink(writer, side))
        writer.close()
        held = [{h: (r[h]["held"] if isinstance(r.get(h), dict) and "held" in r[h] else None) for h in HANDS} for r in info["rule_log"]]
        out = self.feedback(ans, sel, info, before); jr = out.pop("jr"); out["_held"] = held
        out["_cframes"] = [[(p, n) for p, n, _ in fr] for fr in self.contact_frames(info)]
        out.update(_trace=info["trace"], _jr=jr, _go=np.asarray(info["body_state"]["global_orient"]), _contacts=out.pop("contacts_list"),
                   _dec=J, _native=ans.get("_native"), _frame=seg_frame(self.pel, self.f))
        p = W @ jr[-1, 0]; self.pel = np.array([p[0], self.floor + self.ground_rise(jr[-1]), p[2]])
        yaw = heading(info["body_state"]["global_orient"][-1]); fz = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.f = W @ fz; self.f[1] = 0; self.f /= np.linalg.norm(self.f)
        self.jnow = jr[-1].copy()
        bs = info["body_state"]                                          # placed reference / executed joints; executed SMPL (HalfPhysicsEnv)
        nat = {f"kim_{k_}": v for k_, v in (ans.get("_native") or {}).items()}   # Kimodo's native motion (rot, root) of the segment
        self.save_world(str(vid).replace(".mp4", "_world.npz"), info, held)
        np.savez_compressed(str(vid).replace(".mp4", "_joints.npz"), ref=action_joints(act, self.rest), exe=jr, fps=30.0, sub=J,
                            gen=ans.get("_gen", J), exe_transl=np.asarray(bs["transl"], np.float32),
                            exe_global_orient=np.asarray(bs["global_orient"], np.float32),
                            exe_body_pose=np.asarray(bs["body_pose"], np.float32), **nat)
        out.update(segment=k, ans=ans, sel=sel, video=str(vid.relative_to(self.out)), goal=self.goal_now(out),
                   path=[to_map(W @ q).tolist() for q in jr[::5, 0]], end_pel=to_map(self.pel).tolist(), end_heading=heading_map(self.f))
        fb = out["feedback"].split("\n")                    # the goal line is rewritten after this try's evaluation (feedback()
        fb = [l for l in fb if not l.startswith("- Goal: ")]   # wrote it before, with the previous try's result)
        out["feedback"] = "\n".join(fb + ["- Goal: " + "; ".join(f"{t}: {'yes' if v else 'no'}" for t, v in self.goal_parts()) +
                                           f". Goal satisfied: {'YES' if out['goal'] else 'no'}."])
        out["end"] = self.save()
        return out

    def save_world(self, path, info, held):
        """Per frame of a try, what the judge and the reports use but log.json does not keep, so that goals can be
        re-judged and objects analysed from the records alone: object positions / orientations (w x y z) / boxes, palms,
        handles' positions and joints, what each hand holds, body contacts (part, other, normal force); habitat coordinates;
        frames of the trace (30 fps over the motion); plus the objects' state after the settling time."""
        w = self.w; tr = info["trace"]; objs = list(w.objs); mk = list(w.markers); F = len(tr)
        hands = [str(h) for h in HANDS]
        def g(k, keys, shape):
            out = np.full((F, len(keys)) + shape, np.nan)
            for i, f in enumerate(tr):
                for j, x in enumerate(keys):
                    v = f.get(k, {}).get(x)
                    if v is not None:
                        out[i, j] = np.array([list(map(float, r)) for r in v], float).reshape(shape) if len(shape) == 2 else np.array(list(map(float, v)), float)
            return out
        cf = [(i, p, n, f) for i, fr in enumerate(self.contact_frames(info)) for p, n, f in fr]
        np.savez_compressed(path, objects=np.array(objs, dtype="U96"), markers=np.array(mk, dtype="U64"), hands=np.array(hands, dtype="U8"),
                            obj_pos=g("obj", objs, (3,)), obj_rot=g("rot", objs, (4,)), obj_box=g("box", objs, (2, 3)),
                            palm=g("palm", hands, (3,)), marker_pos=g("mpos", mk, (3,)),
                            marker_js=np.array([[float(f["js"].get(m, np.nan)) for m in mk] for f in tr], float).reshape(F, len(mk)),
                            held=np.array([[str(h.get(x) or "") for x in hands] for h in held], dtype="U96").reshape(len(held), len(hands)),
                            contact_frame=np.array([c[0] for c in cf], int), contact_part=np.array([c[1] for c in cf], dtype="U64"),
                            contact_other=np.array([c[2] for c in cf], dtype="U96"), contact_force=np.array([c[3] for c in cf], float),
                            end_obj_pos=np.array([list(map(float, w.obj_pos(o))) for o in objs], float).reshape(len(objs), 3),
                            end_marker_js=np.array([float(w.marker_js(m)) for m in mk], float), floor=np.float64(self.floor))

    # ----------------------------------------------------------------- log summary of a try (not shown to the model)
    def contact_frames(self, info):
        """Per frame the body contacts [(part, other name)], the feet on the floor itself left out. The building is one mesh
        (stage): a contact with it is "the floor" if the contact point is within 5 cm of the floor, else "a wall or pillar"."""
        out = []; floor = getattr(self, "floor", None)
        for rows in (info.get("metric_frames") or {}).get("agent_contacts") or []:
            fr = []
            for c in rows:
                part = c["body_part"]; oth = c["other"]; name = oth.get("name", "?")
                if oth.get("type") == "stage":                   # the building: floor, walls, pillar, stairs (one mesh)
                    if any(s in part for s in FOOT):
                        continue
                    pos = c.get("position_on_other_ws") or c.get("position_on_agent_ws")
                    low = pos is not None and floor is not None and float(pos[1]) - floor < 0.05
                    name = "the floor" if low else "a wall or pillar"      # by the contact point's height
                elif oth.get("type") == "rigid_object" and "DYNAMIC" in str(oth.get("motion_type", "")).upper():
                    name = name.split("_:")[0]
                else:
                    name = short(name.split("_:")[0])
                fr.append((part, name, c.get("normal_force", 0.0)))
            out.append(fr)
        return out

    def contacts(self, info):
        """All body contacts except the feet on the floor itself: [(part, other, t0, t1, max force)] (first and last frame each
        pair touches), in time order."""
        seen = {}
        for t, rows in enumerate(self.contact_frames(info)):
            for part, name, f in rows:
                key = (part, name); s = seen.setdefault(key, [t, t, 0.0]); s[1] = t; s[2] = max(s[2], f)
        return sorted([(p, n, a / 30, b / 30, f) for (p, n), (a, b, f) in seen.items()], key=lambda x: x[2])

    def place_of(self, o, held=None):
        """Where an object is, in the words of the state: in a hand / on a surface / inside a drawer / on the floor / elsewhere."""
        w = self.w; held = self.held() if held is None else held
        hand = [h for h, x in held.items() if x == o]
        if hand:
            return f"in your {hand[0]} hand"
        on = [short(r) for r in w.receptacles if w.on_top_own(o, r)]
        if on:
            return "on " + on[0]
        box = [x for x in w.objs if x != o and category(x) in ("cardboard box", "basket") and w.in_obj_box(x, w.obj_pos(o))]
        if box:
            return "in " + box[0]
        ins = [m for m in w.markers if w.inside(o, m)]
        if ins:
            return "inside " + ins[0]
        return "on the floor" if w.obj_pos(o)[1] - self.floor < 0.1 else "elsewhere"

    def body_axes(self, orient):
        """habitat forward / left unit vectors of the body facing given a Z-up global orient."""
        yaw = heading(orient); f = W @ np.array([np.cos(yaw), np.sin(yaw), 0.0]); f[1] = 0; f /= np.linalg.norm(f)
        return f, np.array([f[2], 0.0, -f[0]])

    def rel(self, v, orient):
        f, l = self.body_axes(orient); return float(v @ f), float(v @ l), float(v[1])

    def goal_parts(self):
        """The task's conditions, each with yes / no as of the last evaluated try."""
        r = (getattr(self, "_bench_last", None) or {}).get("conds", {})
        return [(self.cond_text(c), bool(r.get(c["id"], {}).get("ok"))) for c in self.spec["conditions"]]

    def feedback(self, ans, sel, info, before):
        """Log-only summary of a try (the model never sees it; it measures with inspect): 1. body, 2. each event's result, 3. what
        changed in the world, 4. holding and goal. Used by the evaluation (inhabitant/eval/episode_report.py) and the run logs."""
        w = self.w; jr = executed_joints(info["body_state"], self.rest); go_r = np.asarray(info["body_state"]["global_orient"]); tr = info["trace"]
        c = sel["cands"][sel["chosen"]]; ef, el, et = motion_summary(jr, go_r)
        lines = [f"- You moved (forward / left in m along your facing at the start of the segment, turn in deg, + = left; in brackets the "
                 f"decoded motion before the simulator): forward {ef:+.2f} ({c['fwd']:+.2f}), left {el:+.2f} ({c['left']:+.2f}), "
                 f"turned {et:+.0f} ({c['turn']:+.0f})."]
        ct = self.contacts(info)
        lines.append("- Contacts: " + ("; ".join(f"{p} - {n} {a:.1f}-{b:.1f} s" for p, n, a, b, _ in ct) if ct else "none") + ".")
        summ = [f"moved fwd {ef:+.2f} left {el:+.2f} turn {et:+.0f}"]
        fr = lambda i: go_r[min(i, len(go_r) - 1)]
        for e in ans["events"]:
            if e["kind"] == "grasp":
                tg = e["target"]; ds = [(i, row[e["hand"]]["d"]) for i, row in enumerate(info["rule_log"])
                                        if isinstance(row.get(e["hand"]), dict) and "d" in row[e["hand"]]]
                got = any(isinstance(row.get(e["hand"]), dict) and row[e["hand"]].get("grasped") for row in info["rule_log"])
                thr = w.rules.r_handle if tg in w.markers else w.rules.r_obj
                if got:
                    lines.append(f"- grasp {e['hand']} {tg}: SUCCESS.")
                elif ds:
                    i, d = min(ds, key=lambda x: x[1]); run = best = 0
                    for _, dd in ds:
                        run = run + 1 if dd <= thr else 0; best = max(best, run)
                    p = tr[i]["palm"][e["hand"]]
                    if tg in w.markers:
                        q = tr[i]["mpos"][tg]
                    else:
                        lo, hi = w.obj_aabb(tg); q = np.clip(p, lo - tr[-1]["obj"][tg] + tr[i]["obj"][tg], hi - tr[-1]["obj"][tg] + tr[i]["obj"][tg])
                    fw, le, up = self.rel(q - p, fr(i))
                    lines.append(f"- grasp {e['hand']} {tg}: MISSED; closest {d:.2f} m at t = {i / 30:.1f} s (it must stay within {thr:.2f} m for "
                                 f"{w.rules.k / 30:.1f} s; it stayed {best / 30:.2f} s); at that moment the target was forward {fw:+.2f}, left "
                                 f"{le:+.2f}, up {up:+.2f} m from the palm.")
                else:
                    lines.append(f"- grasp {e['hand']} {tg}: MISSED (the hand was already holding something, or the window was outside the segment).")
                summ.append(f"grasp {tg}: {'ok' if got else 'missed'}")
            else:                                            # what the hand held just before the release (it may have been
                o = before["held"].get(e["hand"])              # grasped in this segment, e.g. a drawer handle)
                rl = info["rule_log"]; i = min(max(int(round(e["t"] * 30)) - 1, 0), len(rl) - 1) if rl else -1
                if i >= 0 and isinstance(rl[i].get(e["hand"]), dict) and rl[i][e["hand"]].get("held"):
                    o = rl[i][e["hand"]]["held"]
                if o is None:
                    lines.append(f"- release {e['hand']}: the {e['hand']} hand was not holding anything.")
                elif o in w.markers:
                    lines.append(f"- release {e['hand']}: let go of {o}.")
                else:
                    p = w.obj_pos(o)
                    if e["surface"] in w.receptacles:                # the surface's own (rotated) box
                        ok = w.on_top_own(o, e["surface"]); q = w.recep_nearest(e["surface"], p)
                    elif e["surface"] in w.objs:                     # into a container object: inside its own (rotated) box
                        ok = w.in_obj_box(e["surface"], p); q = w.obj_nearest(e["surface"], p)
                    else:                                            # the floor
                        ok = bool(p[1] - self.floor < 0.1); q = p
                    if ok:
                        lines.append(f"- release {e['hand']} {o} on {e['surface_name']}: SUCCESS.")
                    else:
                        dist = float(np.linalg.norm((q - p)[[0, 2]])); fw, le, _ = self.rel(q - p, go_r[-1])
                        lines.append(f"- release {e['hand']} {o} on {e['surface_name']}: MISSED; it ended {self.place_of(o)}, {dist:.2f} m from that "
                                     f"surface's region (the region is forward {fw:+.2f}, left {le:+.2f} m from where it ended).")
                    summ.append(f"release {o.split('_:')[0]} on {e['surface_name']}: {'ok' if ok else 'missed'}")
        held = self.held(); ch = []
        for o in w.objs:
            p0, pl0 = before["objs"][o]; p = w.obj_pos(o)
            if pl0.startswith("in your") and held.get(pl0.split()[2]) == o:
                continue
            if np.linalg.norm(p - p0) > 0.05:
                ch.append(f"{o}: {pl0} -> {self.place_of(o, held)} (map {to_map(p)[0]:.2f}, {to_map(p)[1]:.2f}, height {p[1] - self.floor:.2f})")
        for m, q0 in before["js"].items():
            if abs(w.marker_js(m) - q0) > 0.05:
                ch.append(f'{m}{" (" + eb_name(m) + ")" if eb_name(m) else ""}: joint {q0:.2f} -> {art_state(w, m)}')
                summ.append(f"{m} {q0:.2f}->{w.marker_js(m):.2f}")
        lines.append("- Changes in the world: " + ("; ".join(ch) if ch else "none") + ".")
        lines.append("- Holding now: " + (", ".join(f"{h}: {o}" for h, o in held.items() if o) or "nothing") + ".")
        lines.append("- Goal: " + "; ".join(f"{t}: {'yes' if v else 'no'}" for t, v in self.goal_parts()) +
                     f". Goal satisfied: {'YES' if w.goal_satisfied() else 'no'}.")
        return dict(feedback="\n".join(lines), summary="; ".join(summ), jr=jr, contacts_list=ct)

    # ----------------------------------------------------------------- inspect: read-only measurements
    PARTS = dict(pelvis=0, left_knee=4, right_knee=5, left_foot=10, right_foot=11, head=15, left_shoulder=16, right_shoulder=17,
                 left_elbow=18, right_elbow=19)

    def _measured(self):
        """The try that time-based queries measure: the uncommitted try, else the try of the last committed segment (after
        retry_from that is the try the world is now at, not a dropped one)."""
        if self.pending is not None:
            return self.pending
        return self.tries[self.program[-1]["try_n"] - 1] if self.program else None

    def _now(self, fn):
        """fn() on the world after the last try (committed or not), then back to where we were."""
        cur = self.save(); self.load(self.pending["end"] if self.pending else self.snaps[-1])
        try:
            return fn()
        finally:
            self.load(cur)

    def _frame(self, t, n):
        return int(np.clip(round(float(t) * 30), 0, n - 1))

    def _point(self, T, i, x):
        """Habitat position of id x at frame i of try T (None if unknown)."""
        tr = T["_trace"]
        if x in ("left_palm", "right_palm"):
            return np.asarray(tr[i]["palm"][x.split("_")[0]])
        if x in self.PARTS:
            return W @ T["_jr"][min(i, len(T["_jr"]) - 1), self.PARTS[x]]
        if x in tr[i]["obj"]:
            return np.asarray(tr[i]["obj"][x])
        if x in tr[i]["mpos"]:
            return np.asarray(tr[i]["mpos"][x])
        return None

    def _furn_box(self, x):
        """(lo, hi) world box of a piece of furniture by its name (as in the tool outputs), now; None if unknown."""
        ob = self._by_name(x)
        if ob is None:
            return None
        boxes = [self._node_box(ob.root_scene_node)]
        if hasattr(ob, "get_link_ids"):              # articulated furniture: the root node box is empty, use its links
            boxes += [self._node_box(ob.get_link_scene_node(i)) for i in ob.get_link_ids()]
        boxes = [bx for bx in boxes if np.ptp(bx[0] - bx[1]) > 0 or np.any(bx[1] - bx[0] > 1e-3)] or boxes
        return np.min([bx[0] for bx in boxes], 0), np.max([bx[1] for bx in boxes], 0)

    def _box(self, tr, i, x):
        """(lo, hi) of an object at frame i, of a surface, or of a piece of furniture (its box now), else None."""
        k = self.surfaces.get(x) or self.surfaces.get(short(str(x)))
        if k:
            return self.w.recep_box_now(k)
        if x in self.w.objs:
            if "box" in tr[i]:                          # the box at that frame, as the grasp rule sees it
                return tuple(np.asarray(v) for v in tr[i]["box"][x])
            lo, hi = self._now(lambda: self.w.obj_aabb(x)); c = self._now(lambda: self.w.obj_pos(x))
            return lo - c + tr[i]["obj"][x], hi - c + tr[i]["obj"][x]
        if x in self.w.markers or x in self.PARTS:
            return None
        return self._now(lambda: self._furn_box(x))

    def _kframe(self, p):
        """'; in Kimodo's frame of your next segment: X .., Y .., Z ..' (the frame of pose and of constraints: origin on the
        floor below the pelvis of the last committed state, Z its facing, X left, Y up)"""
        yc, oc = seg_frame(self.snaps[-1]["pel"], self.snaps[-1]["f"]); q = to_local(W @ np.asarray(p, float), yc, oc)
        return f"; in Kimodo's frame of your next segment: X {q[1]:+.2f}, Y {q[2]:.2f}, Z {q[0]:+.2f}"

    def _fmt(self, p):
        return f"map ({to_map(p)[0]:.2f}, {to_map(p)[1]:.2f}), height {p[1] - self.floor:.2f}"

    def _where_now(self, x):
        def now():
            w = self.w; pel = self.pel
            if x in self.PARTS or x in ("left_palm", "right_palm"):
                p = w.hand_point(x.split("_")[0]) if x.endswith("palm") else (W @ self.jnow[self.PARTS[x]] if self.jnow is not None else None)
                if p is None:
                    return f"{x}: no try yet"
                extra = f"; facing {heading_map(self.f):+.0f} deg on the map" if x == "pelvis" else ""
            elif x in w.objs:
                p = w.obj_pos(x); extra = f"; {self.place_of(x)}"
            elif x in w.markers:
                p = np.asarray(w.marker_pos(x)); extra = f"; {art_state(w, x)}"
            elif self.surfaces.get(x) or self.surfaces.get(short(str(x))):
                k = self.surfaces.get(x) or self.surfaces.get(short(str(x))); p = w.recep_centre(k); p[1] = w.recep_heights(k)[0]; extra = " (surface centre)"
            else:
                return f"unknown id {x!r}"
            f, l = self.f, np.array([self.f[2], 0, -self.f[0]]); v = p - pel
            return f"{x} now: {self._fmt(p)}{extra}; from your pelvis: forward {v @ f:+.2f}, left {v @ l:+.2f} m{self._kframe(p)}"
        return self._now(now)

    def inspect(self, q):
        fn_ = q.get("fn"); T = self._measured(); need_try = "no try to measure: times refer to your last try or to a committed segment"
        if q.get("segment") is not None:
            k = int(q["segment"])
            if not 1 <= k <= len(self.program):
                return f"segment must be one of the committed segments 1..{len(self.program)}"
            T = self.tries[self.program[k - 1]["try_n"] - 1]
        tr = T["_trace"] if T else None; n = len(tr) if tr else 0
        if "t" in q and "t0" not in q:                                   # where(id, t) is read as where(id, t0)
            q = dict(q, t0=q["t"])
        t0 = float(q.get("t0", 0.0)); t1 = float(q.get("t1", (n - 1) / 30 if n else 0.0))
        if fn_ == "where":
            x = q.get("id")
            if "t0" not in q:
                return self._where_now(x)
            if not T:
                return need_try
            if "t1" not in q:                                            # one moment
                i = self._frame(t0, n); p = self._point(T, i, x)
                if p is None:
                    return f"unknown id {x!r}"
                go = T["_go"][min(i, len(T["_go"]) - 1)]
                fw, le, _ = self.rel(p - W @ T["_jr"][min(i, len(T["_jr"]) - 1), 0], go)
                extra = f"; facing {heading_map(self.body_axes(go)[0]):+.0f} deg on the map" if x == "pelvis" else ""
                return (f"{x} at t = {i / 30:.1f} s: {self._fmt(p)}{extra}; from your pelvis at that moment: forward {fw:+.2f}, "
                        f"left {le:+.2f} m{self._kframe(p)}")
            out = []
            for i in range(self._frame(t0, n), self._frame(t1, n) + 1, 3):
                p = self._point(T, i, x)
                if p is None:
                    return f"unknown id {x!r}"
                fa = f", facing {heading_map(self.body_axes(T['_go'][min(i, len(T['_go']) - 1)])[0]):+.0f}" if x == "pelvis" else ""
                out.append(f"{i / 30:.1f}: ({to_map(p)[0]:.2f}, {to_map(p)[1]:.2f}, h {p[1] - self.floor:.2f}{fa})")
            return f"{x} (map x, y, height{', facing deg' if x == 'pelvis' else ''}) every 0.1 s: " + "; ".join(out)
        if fn_ == "fk":
            return self.fk_query(q)
        if fn_ in ("distance", "contacts", "pose") and not T:
            return need_try
        if fn_ == "distance":
            a, b = q.get("a"), q.get("b"); idx = list(range(self._frame(t0, n), self._frame(t1, n) + 1, 3)); ds = []
            for i in idx:
                pa, pb = self._point(T, i, a), self._point(T, i, b)
                ba, bb = self._box(tr, i, a), self._box(tr, i, b)
                if pa is None and ba is None or pb is None and bb is None:
                    return f"unknown id {a!r} or {b!r}"
                if pa is None:
                    pa = 0.5 * (ba[0] + ba[1])
                qb = np.clip(pa, *bb) if bb is not None else pb
                ds.append((i, float(np.linalg.norm(qb - pa)), qb - pa))
            i, d, v = min(ds, key=lambda x: x[1]); fw, le, up = self.rel(v, T["_go"][min(i, len(T["_go"]) - 1)])
            return (f"distance {a} -> {b}, {idx[0] / 30:.1f}-{idx[-1] / 30:.1f} s: min {d:.2f} m at t = {i / 30:.1f} s; at that moment {b} was "
                    f"forward {fw:+.2f}, left {le:+.2f}, up {up:+.2f} m from {a}.\n  every 0.1 s: " + " ".join(f"{x[1]:.2f}" for x in ds))
        if fn_ == "contacts":
            part, oth = q.get("part"), q.get("other")
            oth = oth.split("_:")[0] if isinstance(oth, str) else oth    # contact names leave out "_:0000"
            rows = [c for c in T["_contacts"] if c[3] >= t0 and c[2] <= t1 and (not part or part in c[0]) and (not oth or oth in c[1])]
            return "contacts: " + ("; ".join(f"{p} - {o} {a:.1f}-{b:.1f} s" for p, o, a, b, _ in rows) or "none")
        if fn_ == "pose":
            return self.pose_query(T, q)
        if fn_ == "extent":
            x = q.get("id")
            def ext():
                w = self.w; k = self.surfaces.get(x) or self.surfaces.get(short(str(x))); extra = ""
                if k:
                    lo, hi = w.recep_box_now(k); lo[1], hi[1] = w.recep_heights(k)
                elif x in w.objs:
                    lo, hi = w.obj_aabb(x)
                elif x in w.markers:
                    lo, hi = self._node_box(w.markers[x]["ao_obj"].get_link_scene_node(w.markers[x]["ao_obj"].get_link_id_from_name(w.markers[x]["link"])))
                    j = w.markers[x]["ao_obj"].get_link_scene_node(w.markers[x]["ao_obj"].get_link_id_from_name(w.markers[x]["link"])).absolute_translation
                    jm = to_map(np.array([j[0], j[1], j[2]]))
                    extra = (f"; this is the part that moves with the handle ({w.markers[x]['link']}); its joint "
                             f"(hinge / slide origin) is at map ({jm[0]:.2f}, {jm[1]:.2f})")
                else:
                    fb = self._furn_box(x)
                    if fb is None:
                        return f"unknown id {x!r}"
                    lo, hi = fb
                cs = [to_map(np.array([a, 0, b])) for a, b in ((lo[0], lo[2]), (hi[0], lo[2]), (hi[0], hi[2]), (lo[0], hi[2]))]
                q2 = np.clip(self.pel, lo, hi)
                if k:                                        # a surface: its own box, turned with the furniture
                    cs = [to_map(c) for c in w.recep_corners(k)]; q2 = w.recep_nearest(k, self.pel); q2[1] = np.clip(self.pel[1], lo[1], hi[1])
                if not k and x in w.objs:                    # an object: its own box, turned with it
                    fp = w.obj_footprint(x); cs = [to_map(np.array([c[0], 0, c[2]])) for c in fp]
                    o_, ex, ez = fp[0], fp[1] - fp[0], fp[3] - fp[0]; d = self.pel - o_
                    q2 = o_ + ex * np.clip(d @ ex / max(ex @ ex, 1e-9), 0, 1) + ez * np.clip(d @ ez / max(ez @ ez, 1e-9), 0, 1)
                    q2[1] = np.clip(self.pel[1], lo[1], hi[1])
                v = q2 - self.pel; f, l = self.f, np.array([self.f[2], 0, -self.f[0]])
                hs = (f"surface height {lo[1] - self.floor:.2f} (its region reaches up to {hi[1] - self.floor:.2f})" if k else
                      f"top height {hi[1] - self.floor:.2f}, bottom {lo[1] - self.floor:.2f}")
                return (f"extent {x}: corners " + " ".join(f"({c[0]:.2f}, {c[1]:.2f})" for c in cs) + f"; {hs}; its nearest point is "
                        f"{np.linalg.norm(v[[0, 2]]):.2f} m from your pelvis (forward {v @ f:+.2f}, left {v @ l:+.2f}){extra}.")
            return self._now(ext)
        return f"unknown fn {fn_!r} (where, distance, contacts, extent, pose, fk)"

    def pose_query(self, T, q):
        """Frame t of what the body executed in try T, as a Kimodo fullbody constraint in Kimodo's frame of the next segment (the
        state after the last commit), on Kimodo's body with the feet where they were (the executed pose, not the generated
        one, so the last frame of the last committed segment has its root at x = z = 0)."""
        z = np.load(self.out / T["video"].replace(".mp4", "_joints.npz")); i = self._frame(q.get("t0", 0.0), len(z["exe"]))
        yc, oc = seg_frame(self.snaps[-1]["pel"], self.snaps[-1]["f"])
        prev = kimodo_prev(z["exe"][i:i + 1], z["exe_global_orient"][i:i + 1], z["exe_body_pose"][i:i + 1], yc, oc)
        r = self.gen_client().call(cmd="native", prev=prev)["result"]
        c = dict(type="fullbody", frame_indices=[0], root_positions=[np.round(r["root_positions"][0], 3).tolist()],
                 local_joints_rot=[np.round(r["local_joints_rot"][0], 3).tolist()])
        th = float(r["heading"][0])
        return (f"executed pose at t = {i / 30:.2f} s of {'segment ' + str(q['segment']) if q.get('segment') is not None else 'your last try'} "
                f"(Kimodo's frame of your next segment; facing global_root_heading [{np.cos(th):.3f}, {np.sin(th):.3f}], theta "
                f"{np.degrees(th):+.0f} deg; set frame_indices as you need): " + json.dumps(c, separators=(",", ":")))

    def fk_query(self, q):
        """Joint positions of poses given in Kimodo's format (one constraint, or root_positions + local_joints_rot), Kimodo's body."""
        c = q.get("pose") if isinstance(q.get("pose"), dict) else q
        if "root_positions" not in c or "local_joints_rot" not in c:
            return "fk needs root_positions and local_joints_rot (as in a Kimodo constraint), or pose = such a constraint"
        J = np.asarray(self.gen_client().call(cmd="fk", root_positions=c["root_positions"], local_joints_rot=c["local_joints_rot"])["result"]["joints"])
        fi = c.get("frame_indices") or list(range(len(J)))
        B = seg_to_kim(to_sim(kim_to_seg(J), self.bones))                  # the same poses on your (simulated) body, as executed
        po = palm_offset(); rows = []
        for k, (P, Q) in enumerate(zip(J[:6], B[:6])):
            mine = {n: Q[JOINTS.index(n)] for n in ("pelvis", "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_foot", "right_foot")}
            for hd in ("left", "right"):
                w, e = Q[JOINTS.index(hd + "_wrist")], Q[JOINTS.index(hd + "_elbow")]
                mine[hd + "_palm"] = w + po * (w - e) / max(np.linalg.norm(w - e), 1e-9)
            rows.append(f"frame {fi[k] if k < len(fi) else k}: Kimodo's body: " + ", ".join(f"{n} ({p[0]:+.3f}, {p[1]:.3f}, {p[2]:+.3f})" for n, p in zip(JOINTS, P))
                        + "\n    fitted to your body: " + ", ".join(f"{n} ({p[0]:+.3f}, {p[1]:.3f}, {p[2]:+.3f})" for n, p in mine.items()))
        return ("fk (X, Y, Z in Kimodo's frame; the constraint acts on Kimodo's body, the motion is then fitted to your body, "
                "before physics): " + "\n  ".join(rows))

    def _node_box(self, node):
        """World AABB (habitat) of a scene node's subtree."""
        import magnum as mn
        bb, T = node.cumulative_bb, node.absolute_transformation()
        pts = [T.transform_point(mn.Vector3(x, y, z)) for x in (bb.min[0], bb.max[0]) for y in (bb.min[1], bb.max[1]) for z in (bb.min[2], bb.max[2])]
        c = np.array([[p[0], p[1], p[2]] for p in pts])
        return c.min(0), c.max(0)

    def _by_name(self, x):
        """A scene object (rigid or articulated) by the name shown in tool outputs (e.g. frl_apartment_chair_01)."""
        for mgr in (self.w.rom, self.w.aom):
            for h, o in mgr.get_objects_by_handle_substring().items():
                if h == x or short(h.split("_:")[0]) == x or h.split("_:")[0] == x:
                    return o
        return None

    # ----------------------------------------------------------------- tools
    def parse_events(self, args, errs):
        events, valid = [], set(self.w.objs) | set(self.w.markers)
        for e in args.get("events") or []:
            try:
                if e.get("kind") == "grasp":
                    if e.get("target") not in valid:
                        errs.append(f"unknown grasp target {e.get('target')!r} (use an object id or a handle id from the state)"); continue
                    events.append(dict(kind="grasp", hand=e["hand"], target=e["target"], t0=float(e["t0"]), t1=float(e["t1"])))
                elif e.get("kind") == "release":
                    s = e.get("surface"); key = self.surfaces.get(s) or self.surfaces.get(short(str(s)))
                    if key is None and s in self.w.objs:             # a container object (basket, box)
                        key = s
                    elif key is None and str(s).lower() == "floor":
                        key = "floor"
                    if key is None:
                        errs.append(f"unknown surface {s!r} (use a surface id or an object id from the state, or 'floor')"); continue
                    events.append(dict(kind="release", hand=e["hand"], surface=key, surface_name=short(key), t=float(e["t"])))
                else:
                    errs.append(f"event kind must be grasp or release: {e}")
            except (KeyError, TypeError, ValueError):
                errs.append(f"incomplete event {e} (grasp: hand, target, t0, t1; release: hand, surface, t)")
        return events

    @staticmethod
    def check_event_times(events, dur, errs):
        for e in events if dur else []:
            if e["kind"] == "release" and not 0 <= e["t"] <= dur:
                errs.append(f"release t = {e['t']} s is outside the segment (0-{dur} s)")
            if e["kind"] == "grasp" and (e["t1"] < e["t0"] or e["t0"] > dur or e["t1"] < 0):
                errs.append(f"grasp window {e['t0']}-{e['t1']} s is empty or outside the segment (0-{dur} s)")

    def gen_client(self):
        if self.gen is None:
            import atexit
            self.gen = GenClient(self.out / "gen_rpc", self.out / "gen_server.log")
            atexit.register(self.gen.close)
        return self.gen

    def generate(self, args):
        """The generator's motion for a generate_segment call -> (ans for execute, None) or (None, error text)."""
        if isinstance(args.get("events"), str):
            try:
                args["events"] = json.loads(args["events"])
            except json.JSONDecodeError:
                pass
        errs = []; events = self.parse_events(args, errs)
        if errs:
            return None, "Not executed:\n" + "\n".join(f"- {x}" for x in errs)
        self.gen_client()
        s = self.snaps[-1]; yaw, origin = seg_frame(s["pel"], s["f"])
        secs = args.get("seconds", 4.0)
        total = sum(float(x) for x in secs) if isinstance(secs, list) else float(secs)
        if not 0.5 <= total <= 8.0:
            return None, f"Not executed: a segment is 0.5 to 8 s (seconds add up to {total:g})"
        req = dict(cmd="generate", text=args.get("text", ""), seconds=secs, seed=int(args.get("seed") or 0))
        req.update(constraints=args.get("constraints") or [], post=bool(args.get("post", True)))
        if args.get("cfg_weight") is not None:
            req["cfg"] = [float(x) for x in args["cfg_weight"]]
        if self.program:                              # continue from the executed end of the last committed segment
            z = np.load(self.out / self.program[-1]["joints"])
            req["prev"] = kimodo_prev(z["exe"][-N_CONT:], z["exe_global_orient"][-N_CONT:], z["exe_body_pose"][-N_CONT:], yaw, origin)
        try:
            r = self.gen.call(**req)
        except Exception as e:                        # the generator failing (or rejecting the inputs) must not end the episode
            return None, f"Not executed: the generator failed: {e}"
        G = np.load(r["joints"]).astype(np.float64); Path(r["joints"]).unlink(missing_ok=True)
        native = None
        if r.get("extra"):
            z = np.load(r["extra"]); native = {k: z[k] for k in z.files}; Path(r["extra"]).unlink(missing_ok=True)
        J = to_sim(G, self.bones)
        J = J[:MAX_T]
        self.check_event_times(events, (len(J) - 1) / FPS, errs)
        if errs:
            return None, "Not executed:\n" + "\n".join(f"- {x}" for x in errs)
        return dict(intent=str(args.get("intent", "")), request={k: v for k, v in args.items() if k not in ("intent", "events")},
                    events=events, root=local_root_motion(J), frames=len(J), _J=J, _gen=G, _native=native, notes=[]), None

    def try_result(self, t):
        """What generate_segment returns for an executed try: the text and the 8-frame sheet of its video."""
        goal = f" Goal satisfied: {'YES' if t['goal'] else 'no'}."
        txt = (f"Try {self.n_since_commit} of segment {t['segment']} executed (not committed).{goal}\nThe video of this try is attached (left: camera following you; right: side camera). "
               f"Use inspect to measure what happened.")
        vid = str(self.out / t["video"])
        sheet = vid[:-4] + "_sheet.jpg"; contact_sheet(vid, sheet)          # 8 frames of the video in one image
        txt = txt.replace("The video of this try is attached (", "8 frames of the video of this try are attached (time stamps on "
                          "each frame; in each frame ")
        return [{"type": "text", "text": txt}, {"type": "image", "path": sheet}]

    def tool(self, name, args):
        """-> (content: str | parts, state text or None). Tool errors are returned to the model as text."""
        names = [t["function"]["name"] for t in self.tools]
        if name not in names:
            return f"Unknown tool {name!r} (available: {', '.join(names)}).", None
        if name == "generate_segment":
            ans, err = self.generate(args)
            if err:
                return err, None
            self.load(self.snaps[-1]); t0 = time.time(); t = self.execute(ans); self.tries.append(t); self.pending = t
            self.n_since_commit += 1; t["n"] = len(self.tries)
            print(f"  try {t['n']} (seg {t['segment']}, {time.time() - t0:.0f} s): {ans['intent'][:80]} -> {t['summary']}", flush=True)
            return self.try_result(t), None
        if name == "commit":
            if self.pending is None:
                return "Nothing to commit: no try since the last commit.", None
            t = self.pending; self.load(t["end"]); self.snaps.append(self.save()); self.pending = None; self.n_since_commit = 0
            jf = str(t["video"]).replace(".mp4", "_joints.npz")
            self.program.append(dict(segment=t["segment"], try_n=t["n"], intent=t["ans"]["intent"], request=t["ans"]["request"],
                                     frames=t["ans"].get("frames"), root=t["ans"]["root"], events=t["ans"]["events"], summary=t["summary"],
                                     video=t["video"], goal=t["goal"], joints=jf if (self.out / jf).exists() else None))
            print(f"  COMMIT segment {len(self.program)}: {t['summary']}{' | GOAL' if t['goal'] else ''}", flush=True)
            msg = f"Committed segment {len(self.program)}."
            if t["goal"]:                        # the task is accomplished: the episode ends
                self.versions.append(copy.deepcopy(self.program)); self.done = True; self.end_reason = "goal"
                return msg + " The task is accomplished; the episode ends.", None
            return msg + "\n\n# Current state (near you; get_state() gives the full state)\n{STATE}", self.near_state()
        if name == "inspect":
            qs = args.get("queries") or []
            if isinstance(qs, str):
                try:
                    qs = json.loads(qs)
                except json.JSONDecodeError:
                    return "queries must be a list of {fn, ...}", None
            out = []
            for k, q in enumerate(qs[:20]):
                try:
                    q = q if isinstance(q, dict) else {}
                    out.append(f"[{k + 1}] " + self.inspect(q))
                except Exception as e:                      # a bad query must not end the episode
                    out.append(f"[{k + 1}] error: {type(e).__name__}: {e}")
            return "\n".join(out) or "no queries", None
        if name == "get_state":
            if self.pending is not None:
                cur = self.save(); self.load(self.pending["end"]); st = self.state_text(); self.load(cur)
                return ("# State after your last try (NOT committed yet: commit() keeps it; another generate_segment starts again "
                        "from the last committed state)\n{STATE}"), st
            return "# Current state (last committed)\n{STATE}", self._committed_state()
        if name == "retry_from":
            i = int(args.get("i", 0))
            if not 1 <= i <= len(self.program):
                return f"i must be between 1 and {len(self.program)}.", None
            self.program = self.program[:i - 1]; self.snaps = self.snaps[:i]; self.load(self.snaps[-1]); self.pending = None
            self.n_since_commit = 0
            return f"The world is back at the start of segment {i}; segments {i}.. are dropped.\n\n# Current state\n{{STATE}}", self.state_text()
        return f"Unknown tool {name!r}.", None

    def call_tool(self, name, args):
        """Entry point for the agent's tool calls -> (content, state text or None); see tool()."""
        return self.tool(name, args)

    def _committed_state(self):
        cur = self.save(); self.load(self.snaps[-1]); s = self.state_text(); self.load(cur); return s

    def task_message(self):
        return f"# Task\nInstruction: {self.w.ep['instruction']}"

    def save_log(self, best=None):
        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items() if k != "end" and not str(k).startswith("_")}
            if isinstance(x, (list, tuple)):
                return [clean(v) for v in x]
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x
        json.dump(clean(dict(eval_set=self.a.eval_set, id=self.a.id, instruction=self.w.ep["instruction"], goal=self.goal_text(),
                             model=self.a.model, args=vars(self.a), spec=self.spec, llm_calls=self.calls, success=bool(self.versions),
                             end_reason=getattr(self, "end_reason", "") or ("max_calls" if self.calls >= self.a.max_calls else ""),
                             best=best, versions=self.versions, program=self.program, tries=self.tries, system_prompt=self.system,
                             tool_log=self.tool_log, usage=self.usage,
                             messages=self.msgs)), open(self.out / "log.json", "w"), indent=1)
