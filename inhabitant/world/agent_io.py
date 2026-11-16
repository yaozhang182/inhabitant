"""Text side of the EB humanoid agent: what the LLM is told about the world (scene text, top-down map, goal). Geometry only
(perception out of scope); no paths or waypoints (navigation is the LLM's job).

Map frame (top-down, metres): x = habitat x, y = -habitat z; heading 0 = +x, counter-clockwise positive (= turning left).
Egocentric numbers are given too: forward / left of the person's current pelvis and heading, and the direction in words
('35 deg to your left'): to face it, turn by that angle."""
import numpy as np

from .eb_world import category, eb_name


def label(name, shown):
    """Id shown to the agent, followed by EB's readable name when EB has one: 'Tbl2_Top1_table_02 ("brown table")'."""
    n = eb_name(name); return f'{shown} ("{n}")' if n else shown


def art_state(w, m):
    """'open' / 'partly open' / 'closed' of a drawer or the fridge door, with the thresholds of EB's predicates."""
    fridge = m.startswith("fridge"); q = w.marker_js(m)
    op, cl = (w.opened_fridge(m), w.closed_fridge(m)) if fridge else (w.opened_cab(m), w.closed_cab(m))
    st = "open" if op else "closed" if cl else "partly open"
    return f"{st} (joint {q:.2f}; open above {1.07 if fridge else 0.35:.2f}, closed below 0.15)"


def to_map(p):
    return np.array([p[0], -p[2]])


def heading_map(f):
    """habitat forward vector -> map heading (deg)."""
    return float(np.degrees(np.arctan2(-f[2], f[0])))


def ego(p, pel, f):
    """habitat point -> (forward, left, height, distance, bearing) relative to the pelvis position / habitat forward f."""
    d = to_map(p) - to_map(pel); fm = to_map(f); fm = fm / np.linalg.norm(fm); lm = np.array([-fm[1], fm[0]])
    fw, le = float(d @ fm), float(d @ lm)
    return fw, le, float(np.degrees(np.arctan2(le, fw))), float(np.hypot(fw, le))


def compass(h):
    """Map heading (deg, 0 = +x) -> 'east', 'north-east', ... with +x = east, +y = north."""
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(((h % 360) / 45.0))) % 8]


def side(b):
    """Bearing in words: 'ahead', '35 deg to your left', '120 deg to your right'."""
    return "straight ahead" if abs(b) < 5 else f"{abs(b):.0f} deg to your {'left' if b > 0 else 'right'}"


def short(name):
    return name.replace("receptacle_aabb_", "").replace("_frl_apartment", "")


class SceneText:
    def __init__(self, world, cell=0.25):
        self.w = world; self.cell = cell
        self.floor_y = None; self._grid()

    def _grid(self):
        """Occupancy of the floor from the navmesh (agent radius 0.2 m): '.' free, '#' blocked."""
        w = self.w; pf = w.navmesh(); lo, hi = pf.get_bounds()
        p0 = np.array(w.ep["start_position"], float); self.floor_y = float(p0[1])
        xs = np.arange(lo[0], hi[0] + 1e-6, self.cell); zs = np.arange(lo[2], hi[2] + 1e-6, self.cell)
        self.xs, self.ys = xs, -zs                            # zs ascending -> ys descending: row 0 (top) = largest y (north)
        occ = np.zeros((len(self.ys), len(xs)), bool)
        for r, y in enumerate(self.ys):
            for c, x in enumerate(xs):
                q = np.array([x, self.floor_y, -y])
                s = np.array(pf.snap_point(q))
                occ[r, c] = bool(np.isfinite(s).all() and np.linalg.norm((s - q)[[0, 2]]) < 0.5 * self.cell
                                 and abs(s[1] - self.floor_y) < 0.3)
        self.free = occ

    def ascii_map(self, marks):
        """marks: list of (char, habitat point). Columns: x; rows: y (top = north = high y)."""
        g = np.where(self.free, ".", "#").astype("<U1")
        for ch, p in marks:
            m = to_map(p); c = int(round((m[0] - self.xs[0]) / self.cell)); r = int(round((self.ys[0] - m[1]) / self.cell))
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                g[r, c] = ch
        lines = []
        for r in range(g.shape[0]):
            y = self.ys[r]; lab = f"{y:6.1f} " if abs(y / 1.0 - round(y)) < self.cell / 2 else "       "
            lines.append(lab + "".join(g[r]))
        ticks = [x for x in self.xs if abs(x - round(x)) < self.cell / 2]
        ruler = "       " + "".join("|" if abs(x - round(x)) < self.cell / 2 else " " for x in self.xs)
        return "\n".join(lines + [ruler, "       x of the | marks, left to right: " + ", ".join(f"{round(x):.0f}" for x in ticks)
                                   + ". Row labels on the left are y (top = larger y)."])

    def state(self, pel, f, held):
        """pel: pelvis (habitat), f: habitat forward unit vector, held: {hand: object id or None}."""
        w = self.w; out = []
        out.append(f"## You (the person)\n- pelvis at map (x {to_map(pel)[0]:.2f}, y {to_map(pel)[1]:.2f}), facing heading "
                   f"{heading_map(f):.0f} deg = {compass(heading_map(f))} (map: +x = east = 0, +y = north = 90, -y = south = -90; "
                   f"turning left increases the heading)\n"
                   f"- holding: " + (", ".join(f"{h} hand: {o}" for h, o in held.items() if o) or "nothing"))
        rows = []
        for name, (lo, hi) in sorted(w.receptacles.items()):
            if hasattr(w, "recep_centre"):                   # its own box now (a drawer's surface moves with it)
                c = w.recep_centre(name); lo = np.array(lo, float); lo[1] = w.recep_heights(name)[0]
            else:
                c = 0.5 * (lo + hi)
            fw, le, b, d = ego(c, pel, f)
            sx, sz = w.recep_size(name) if hasattr(w, "recep_size") else (hi[0] - lo[0], hi[2] - lo[2])   # its own box
            rows.append(f"- {label(name, short(name))}: centre map ({to_map(c)[0]:.2f}, {to_map(c)[1]:.2f}), surface height {lo[1] - self.floor_y:.2f}, size "
                        f"{sx:.2f} x {sz:.2f} m | from you: {d:.2f} m, {side(b)} (forward {fw:+.2f}, left {le:+.2f})")
        out.append("## Places where objects can be put (receptacle surfaces)\n" + "\n".join(rows))
        rows = []
        for o in w.objs:
            p = w.obj_pos(o); fw, le, b, d = ego(p, pel, f)
            on = [short(r) for r in w.receptacles if (w.on_top_own(o, r) if hasattr(w, "on_top_own") else w.on_top(o, r))]
            ins = [m for m in w.markers if w.inside(o, m)]
            box = [x for x in w.objs if x != o and category(x) in ("cardboard box", "basket") and w.in_obj_box(x, p)] if hasattr(w, "in_obj_box") else []
            where = (("on " + on[0]) if on else ("inside " + ins[0]) if ins else ("in " + box[0]) if box else
                     ("in your hand" if o in held.values() else "elsewhere"))
            rows.append(f"- {o} (type '{category(o)}'): map ({to_map(p)[0]:.2f}, {to_map(p)[1]:.2f}), height {p[1] - self.floor_y:.2f}, "
                        f"{where} | from you: {d:.2f} m, {side(b)} (forward {fw:+.2f}, left {le:+.2f})")
        out.append("## Objects\n" + "\n".join(rows))
        rows = []
        for m in w.markers:
            p = w.marker_pos(m); fw, le, b, d = ego(p, pel, f)
            fridge = m.startswith("fridge")
            q = w.marker_js(m)
            q0 = q; eps = 0.02; w.set_marker_js(m, q0 + eps); dp = (w.marker_pos(m) - p) / eps; w.set_marker_js(m, q0)
            pull = to_map(dp); pull = pull / (np.linalg.norm(pull) + 1e-9)
            kind = "fridge door (hinged, swings open)" if fridge else f"drawer of the {w.markers[m]['ao'].split('_:')[0]} ({w.markers[m]['link']})"
            need = "open the door to at least 1.07 rad" if fridge else "pull it out at least 0.35 m"
            rows.append(f"- {label(m, m)}: handle of the {kind}; handle at map ({to_map(p)[0]:.2f}, {to_map(p)[1]:.2f}), height {p[1] - self.floor_y:.2f}; "
                        f"{art_state(w, m)}; to open: grasp the handle and move the hand towards map direction "
                        f"({pull[0]:+.2f}, {pull[1]:+.2f}) ({need}) | from you: {d:.2f} m, {side(b)} (forward {fw:+.2f}, left {le:+.2f})")
        out.append("## Handles of drawers and the fridge\n" + "\n".join(rows))
        return "\n\n".join(out)
