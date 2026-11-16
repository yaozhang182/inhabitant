"""Goal conditions of the Inhabit benchmark (benchmark/inhabit/README.md), evaluated on the final committed program (+ the current try).

  spec = dict(conditions=[dict(id="c1", kind="sit_on", surface="sofa"), ...], order=[["c1", "c2"], ...])
  ev = BenchGoal(spec, world, floor_y); ev.evaluate(segs) -> dict(ok, conds={id: dict(ok, t, why)}, judge=[ids needing a person])

Time semantics: object conditions (on_top, inside, under, in_container, opened / closed, throw, kick, push) are checked on the world state now (after the
last try and its settling; as EB); body conditions (sit_on, lie_on, stand_on, squat, kneel, handstand, jump, jump_onto, circle,
dance_on) must hold continuously for their duration at some time of the program; constraints (avoid, hand_only) over the whole
program. `order` [a, b]: a is first achieved before b (an object condition is achieved from the frame it becomes true and stays
true to the end of its segment's trace, else at the end).

segs: one dict per executed segment in program order: jr (T,22,3) executed joints (Z-up world), cframes [[(part, other)]] body
contacts per frame (feet on the floor itself not listed), held [{hand: target | None}] per frame, objs [{handle: habitat position}]
and js [{marker: joint}] per frame.
All geometry here is in habitat coordinates (y up); surfaces are the episode's receptacle regions (world AABB, `world.receptacles`),
named by any substring of the receptacle name (e.g. "sofa", "Tbl2_Top1", "chair_01_:0001")."""
import numpy as np
from .hp_bridge import W

FPS = 30.0
PEL, LKNEE, RKNEE, LANK, RANK, LTOE, RTOE, NECK, HEAD, LWR, RWR = 0, 4, 5, 7, 8, 10, 11, 12, 15, 20, 21
FEET = (LANK, RANK, LTOE, RTOE)
HOLD = dict(sit_on=1.0, lie_on=1.0, stand_on=1.0, squat=1.0, kneel=1.0, handstand=1.0, dance_on=5.0)
BODY = set(HOLD) | {"jump", "jump_onto", "circle"}
FOOT_PARTS = ("foot", "ankle", "toe")


def _runs(mask, n):
    """start indices of runs of >= n consecutive True"""
    out, k = [], 0
    for i, m in enumerate(mask):
        k = k + 1 if m else 0
        if k == n:
            out.append(i - n + 1)
    return out


def _angle_from_vertical(v):
    return np.degrees(np.arccos(np.clip(v[..., 1] / np.maximum(np.linalg.norm(v, axis=-1), 1e-9), -1, 1)))


class BenchGoal:
    def __init__(self, spec, world, floor_y, eb_goal=None):
        self.spec, self.w, self.floor, self.eb_goal = spec, world, float(floor_y), eb_goal or world.goal_satisfied

    # ---------------- helpers
    def rkey(self, name):
        ks = [k for k in self.w.receptacles if name in k]
        if not ks:
            raise KeyError(f"no surface matching {name!r}")
        return ks[0]

    def region(self, name):
        """(lo, hi) habitat AABB of the receptacle whose name contains `name` (EB's: squashed for turned furniture; used here
        only for heights, which it gets right). Inside / distance tests use the receptacle's own box (`inr`, `rdist`)."""
        k = self.rkey(name)
        if hasattr(self.w, "recep_box_now"):          # its own box now (heights, distances; a drawer's surface moves)
            lo, hi = self.w.recep_box_now(k); lo[1], hi[1] = self.w.recep_heights(k); return lo, hi
        return self.w.receptacles[k]

    def inr(self, name, p, m=0.0):
        """x / z of point(s) p inside the receptacle's own (rotated) box, +- m"""
        return self.w.in_recep(self.rkey(name), p, m=m, y=False)

    def rdist(self, name, p):
        return self.w.recep_dist_xz(self.rkey(name), p)

    @staticmethod
    def inside_xz(p, lo, hi, m=0.0):
        return (p[..., 0] >= lo[0] - m) & (p[..., 0] <= hi[0] + m) & (p[..., 2] >= lo[2] - m) & (p[..., 2] <= hi[2] + m)

    def touching(self, segs, n, part=None, other=None):
        """(n,) per global frame: some body part matching `part` (substring or tuple of substrings) touches something matching
        `other` (substring)"""
        m = np.zeros(n, bool); k = 0
        for s in segs:
            T = len(s["jr"])
            for i, rows in enumerate(s["cframes"][:T]):
                m[k + i] = any((part is None or any(x in p for x in (part if isinstance(part, tuple) else (part,))))
                               and (other is None or other in o) for p, o in rows)
            k += T
        return m

    # ---------------- evaluation
    def evaluate(self, segs):
        H = [(W @ s["jr"].reshape(-1, 3).T).T.reshape(s["jr"].shape) for s in segs]      # habitat joints per segment
        J = np.concatenate(H) if H else np.zeros((0, 22, 3))
        t0s = np.cumsum([0] + [len(h) / FPS for h in H])[:-1]
        held = [h for s in segs for h in s["held"]]
        conds = {}
        for c in self.spec["conditions"]:
            try:
                ok, t, why = getattr(self, "c_" + c["kind"])(c, J, segs, t0s, held)
            except KeyError as e:
                ok, t, why = False, None, f"bad condition: {e}"
            conds[c["id"]] = dict(ok=bool(ok), t=t, why=why)
        order_ok = all(conds[a]["ok"] and conds[b]["ok"] and conds[a]["t"] is not None and conds[b]["t"] is not None
                       and conds[a]["t"] < conds[b]["t"] for a, b in self.spec.get("order", []))
        judge = [c["id"] for c in self.spec["conditions"] if c["kind"] == "dance_on"]
        return dict(ok=all(v["ok"] for v in conds.values()) and order_ok, order_ok=order_ok, conds=conds, judge=judge)

    def _first_hold(self, mask, hold_s):
        r = _runs(mask, max(1, int(round(hold_s * FPS))))
        return (True, r[0] / FPS) if r else (False, None)

    def _obj_time(self, segs, t0s, pred):
        """global time from which pred(position) holds to the end of the trace of the last segment where it became true"""
        t = None
        for s, t0 in zip(segs, t0s):
            for i, f in enumerate(s["objs"]):
                if pred(f):
                    t = t0 + i / FPS if t is None else t
                else:
                    t = None
        return t

    # ---------------- object conditions (state now)
    def c_on_top(self, c, J, segs, t0s, held):
        k = self.rkey(c["surface"]); o = c["obj"]; tol = getattr(self.w, "on_top_tol", 0.0)       # as EBWorld.on_top, own box
        pred = lambda p: bool(self.w.in_recep(k, p, below=tol))
        ok = pred(self.w.obj_pos(o)) and not self.w.holding(o)
        return ok, self._obj_time(segs, t0s, lambda f: pred(f[o])) if ok else None, ""

    def c_inside(self, c, J, segs, t0s, held):
        ok = self.w.inside(c["obj"], c["container"]) and not self.w.holding(c["obj"])
        lo, hi = self.w.marker_container_aabb(c["container"]); o = c["obj"]
        return ok, self._obj_time(segs, t0s, lambda f: bool(np.all(f[o] >= lo) and np.all(f[o] <= hi))) if ok else None, ""

    def c_under(self, c, J, segs, t0s, held):
        """object on the floor below a surface: xz inside the region, < 0.15 m above the floor"""
        o = c["obj"]; fl = self.floor; s_ = c["surface"]
        pred = lambda p: bool(self.inr(s_, p) and p[1] - fl < 0.15)
        ok = pred(self.w.obj_pos(o)) and not self.w.holding(o)
        return ok, self._obj_time(segs, t0s, lambda f: pred(f[o])) if ok else None, ""

    def _joint(self, c, J, segs, t0s, fn):
        m = c["marker"]; ok = fn(m)
        t = None
        for s, t0 in zip(segs, t0s):
            for i, f in enumerate(s.get("js", [])):
                if fn(m, f[m]):
                    t = t0 + i / FPS if t is None else t
                else:
                    t = None
        return ok, (t if ok else None), ""

    def c_opened(self, c, J, segs, t0s, held):
        w = self.w; fr = c["marker"].startswith("fridge")
        th = (lambda m, q=None: (w.marker_js(m) if q is None else q) > (1.22 - 0.15 if fr else 0.45 - 0.1))
        return self._joint(c, J, segs, t0s, th)

    def c_closed(self, c, J, segs, t0s, held):
        w = self.w
        return self._joint(c, J, segs, t0s, lambda m, q=None: abs(w.marker_js(m) if q is None else q) < 0.15)

    def obj_box(self, name):
        """(lo, hi) habitat AABB now of an episode object whose handle contains `name` (an added container: basket, box)"""
        hs = [h for h in self.w.objs if name in h]
        if not hs:
            raise KeyError(f"no object matching {name!r}")
        return self.w.obj_aabb(hs[0])

    def c_in_container(self, c, J, segs, t0s, held):
        """object inside an open container object (an added basket / box): its centre inside the container's own
        box, rotated with it (walls 1 cm), height between its bottom and top (+ 0.02 m), not in a hand. The container's pose
        is the one now (it can be moved)."""
        o = c["obj"]; hs = [h for h in self.w.objs if c["in"] in h]
        if not hs:
            raise KeyError(f"no object matching {c['in']!r}")
        pred = lambda p: self.w.in_obj_box(hs[0], p)
        ok = pred(self.w.obj_pos(o)) and not self.w.holding(o)
        return ok, self._obj_time(segs, t0s, lambda f: pred(f[o])) if ok else None, ""

    def _target(self, c):
        """the condition a throw / kick / push has to end in, and the target's (lo, hi) for distances"""
        if "in" in c:
            return dict(c, kind="in_container"), self.obj_box(c["in"])
        if "container" in c:
            return dict(c, kind="inside"), self.w.marker_container_aabb(c["container"])
        return dict(c, kind="under" if c.get("under") else "on_top"), self.region(c["surface"])

    def c_throw(self, c, J, segs, t0s, held):
        """the object ends on / in the target; when it left the hand the palm was >= 1.0 m (horizontal) from the target region"""
        o = c["obj"]; tgt, (lo, hi) = self._target(c)
        ok, t, _ = getattr(self, "c_" + tgt["kind"])(tgt, J, segs, t0s, held)
        rel = [i for i in range(1, len(held)) if o in held[i - 1].values() and o not in held[i].values()]
        if not rel:
            return False, None, "never released"
        pal = J[rel[-1], (RWR if held[rel[-1] - 1].get("right") == o else LWR)]
        if c.get("surface"):                                     # a surface: its own box
            d = float(self.rdist(c["surface"], pal))
        else:
            d = float(np.linalg.norm(np.maximum(0, np.maximum(lo[[0, 2]] - pal[[0, 2]], pal[[0, 2]] - hi[[0, 2]]))))
        return ok and d >= 1.0, t, f"released {d:.2f} m from the target"

    def c_kick(self, c, J, segs, t0s, held):
        """the object ends in the target (surface region, or `under` a surface); a foot touched it after it last left a hand
        (or at any time if it was never held) and it was not in a hand afterwards"""
        o = c["obj"]; tgt, _ = self._target(c)
        ok, t, _ = getattr(self, "c_" + tgt["kind"])(tgt, J, segs, t0s, held)
        inh = np.array([o in h.values() for h in held] + [False] * max(0, len(J) - len(held)))[:len(J)]
        last = int(np.where(inh)[0][-1]) + 1 if inh.any() else 0
        foot = bool(self.touching(segs, len(J), FOOT_PARTS, o.split("_:")[0])[last:].any())
        return ok and foot, t, f"foot contact after it left the hand {foot}"

    def c_push(self, c, J, segs, t0s, held):
        """the object (e.g. a stool, a box) ends in the target (surface region, `under` a surface, or `in` a container), or with
        `away` = dict(from, dist) at least dist m from where it started; it was never in a hand and the body touched it"""
        o = c["obj"]
        if "away" in c:                                          # "nudge it aside": ends >= dist m (horizontal) from where it started
            ok = float(np.linalg.norm((self.w.obj_pos(o) - np.asarray(c["away"]["from"]))[[0, 2]])) >= c["away"]["dist"]; t = 0.0 if ok else None
        else:
            tgt, _ = self._target(c)
            ok, t, _ = getattr(self, "c_" + tgt["kind"])(tgt, J, segs, t0s, held)
        never = not any(o in h.values() for h in held)
        touched = bool(self.touching(segs, len(J), None, o.split("_:")[0]).any())
        return ok and never and touched, t, f"never held {never}, touched {touched}"

    def c_eb(self, c, J, segs, t0s, held):
        """an original EB episode's goal on the world state now (EB's own definition: no on_top tolerance)"""
        tol, self.w.on_top_tol = getattr(self.w, "on_top_tol", 0.0), 0.0
        try:
            return bool(self.eb_goal()), None, ""
        finally:
            self.w.on_top_tol = tol

    def c_at(self, c, J, segs, t0s, held):
        """the pelvis within `dist` (default 1.0 m, horizontal) of a surface region for >= 0.5 s"""
        d = self.rdist(c["surface"], J[:, PEL])
        return (*self._first_hold(d <= c.get("dist", 1.0), 0.5), "")

    # ---------------- body conditions (some time of the program, held)
    def c_sit_on(self, c, J, segs, t0s, held):
        lo, hi = self.region(c["surface"]); top = lo[1]; p = J[:, PEL]
        m = self.inr(c["surface"], p) & (p[:, 1] - top > 0.05) & (p[:, 1] - top < 0.35) & (_angle_from_vertical(J[:, NECK] - p) < 45)
        return (*self._first_hold(m, HOLD["sit_on"]), "")

    def c_lie_on(self, c, J, segs, t0s, held):
        p, h = J[:, PEL], J[:, HEAD]; flat = _angle_from_vertical(J[:, NECK] - p) > 60
        if c.get("surface", "floor") == "floor":
            m = flat & (p[:, 1] - self.floor < 0.35) & (h[:, 1] - self.floor < 0.35)
        else:
            lo, hi = self.region(c["surface"]); top = lo[1]
            m = flat & self.inr(c["surface"], p, 0.05) & self.inr(c["surface"], h, 0.25) & (p[:, 1] - top < 0.35) & (p[:, 1] - top > -0.05)
        return (*self._first_hold(m, HOLD["lie_on"]), "")

    def _stand_on(self, c, J):
        lo, hi = self.region(c["surface"]); top = lo[1]; a = J[:, [LANK, RANK]]
        return (self.inr(c["surface"], a, 0.1).all(1) & (a[..., 1].min(1) >= top - 0.05)        # ankles may be at the edge
                & (J[:, PEL, 1] - a[..., 1].mean(1) > 0.6))

    def c_stand_on(self, c, J, segs, t0s, held):
        return (*self._first_hold(self._stand_on(c, J), HOLD["stand_on"]), "")

    def c_dance_on(self, c, J, segs, t0s, held):
        """rule part only (stand on it >= 5 s); whether it is dancing is judged by people / an LLM from the video"""
        return (*self._first_hold(self._stand_on(c, J), HOLD["dance_on"]), "rule part only; judged from the video")

    def c_squat(self, c, J, segs, t0s, held):
        fl = self.floor
        m = ((J[:, PEL, 1] - fl < 0.55) & (J[:, PEL, 1] - fl > 0.25) & (J[:, [LANK, RANK], 1].max(1) - fl < 0.15)
             & (J[:, [LKNEE, RKNEE], 1].min(1) - fl > 0.12))                      # pelvis off the floor: not sitting on it
        return (*self._first_hold(m, HOLD["squat"]), "")

    def c_kneel(self, c, J, segs, t0s, held):
        # a knee on the floor with the pelvis off it (0.25-0.8 m: not sitting or lying on the floor, not standing); the trunk may
        # lean forward, as when reaching under a table on the knees.
        pel = J[:, PEL, 1] - self.floor
        m = (J[:, [LKNEE, RKNEE], 1].min(1) - self.floor < 0.1) & (pel > 0.25) & (pel < 0.8)
        return (*self._first_hold(m, HOLD["kneel"]), "")

    def c_handstand(self, c, J, segs, t0s, held):
        sup = self.floor if c.get("surface", "floor") == "floor" else self.region(c["surface"])[0][1]
        m = (J[:, HEAD, 1] < J[:, PEL, 1] - 0.3) & (np.abs(J[:, [LWR, RWR], 1] - sup).max(1) < 0.25)   # wrist joint ~0.2 m above flat palms
        m &= ~self.touching(segs, len(J), FOOT_PARTS) & (J[:, FEET, 1].min(1) - self.floor > 0.3)   # feet touch nothing
        return (*self._first_hold(m, HOLD["handstand"]), "")

    def _airborne(self, J, segs, t0s):
        return (J[:, FEET, 1].min(1) - self.floor > 0.12) & ~self.touching(segs, len(J), FOOT_PARTS)   # hands may touch a wall

    def c_jump(self, c, J, segs, t0s, held):
        """the lowest foot joint > 0.12 m above the floor and the feet touch nothing for >= 0.15 s"""
        return (*self._first_hold(self._airborne(J, segs, t0s), 0.15), "")

    def c_jump_onto(self, c, J, segs, t0s, held):
        """airborne (as jump) and standing on the surface within 2 s after; no walking on the floor in between"""
        air = _runs(self._airborne(J, segs, t0s), int(0.15 * FPS)); st = _runs(self._stand_on(c, J), int(HOLD["stand_on"] * FPS))
        for a in air:
            if any(a < s <= a + 2 * FPS for s in st):
                return True, a / FPS, ""
        return False, None, ""

    def c_circle(self, c, J, segs, t0s, held):
        """the pelvis goes once around the target (cumulative angle >= 360 deg) while staying within 2.5 m of it"""
        if "surface" in c:
            lo, hi = self.region(c["surface"]); ctr = 0.5 * (lo + hi)
        else:
            ctr = self.w.obj_pos(c["obj"]) if c.get("obj") in self.w.objs else self.w.entity_pos(c["obj"])
        d = J[:, PEL, [0, 2]] - ctr[[0, 2]]; ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); near = np.linalg.norm(d, axis=1) < 2.5
        i = 0
        while i < len(ang):
            if not near[i]:
                i += 1; continue
            j = i
            while j + 1 < len(ang) and near[j + 1]:
                j += 1
            a = ang[i:j + 1]
            for k in range(len(a)):
                hit = np.where(np.abs(a[k:] - a[k]) >= 2 * np.pi)[0]
                if len(hit):
                    return True, (i + k + hit[0]) / FPS, ""
            i = j + 1
        return False, None, f"max turn {np.degrees(np.ptp(ang)) if len(ang) else 0:.0f} deg"

    # ---------------- constraints (whole program)
    def c_avoid(self, c, J, segs, t0s, held):
        hits = int(self.touching(segs, len(J), None, c["thing"].split("_:")[0]).sum())   # contacts name objects without "_:0000"
        return not hits, 0.0 if not hits else None, f"{hits} frames in contact"

    def c_held_together(self, c, J, segs, t0s, held):
        """both objects in hands at the same time at some frame (carried together, one in each hand)"""
        a, b = c["a"], c["b"]
        i = next((i for i, h in enumerate(held) if a in h.values() and b in h.values()), None)
        return i is not None, (None if i is None else i / FPS), ""

    def c_hand_only(self, c, J, segs, t0s, held):
        o, hd = c["obj"], c["hand"]; other = "left" if hd == "right" else "right"
        bad = any(h.get(other) == o for h in held); used = any(h.get(hd) == o for h in held)
        return used and not bad, 0.0 if used and not bad else None, ""
