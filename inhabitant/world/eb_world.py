"""One simulator for EB-Habitat with a human body: an EB episode (ReplicaCAD scene, YCB objects, drawer / door states) rebuilt
inside HumanCLAW's HalfPhysicsEnv, EB's PDDL predicates re-implemented on that simulator's state (no Fetch), and the hand-object
interaction rules (grasp / carry / release / pull a handle).

  w = EBWorld("base", 39)                      # subset and index of the episode in EB's dataset pickle
  w.true_predicates(), w.goal_satisfied()      # same definitions as EB's pddl_domain_replica_cad.yaml + habitat-lab 0.3.0
  w.env.reset(...); w.execute(action, events)  # action: HalfPhysicsEnv SMPL-X chunk (hp_bridge.place); events: see execute()

Predicates (habitat-lab 0.3.0 pddl_sim_state.py, thresholds from its defaults and EB's domain file):
  on_top(obj, recep)   object position inside the receptacle's world AABB (find_receptacles, as RearrangeSim)
  in(obj, marker)      object position inside the drawer link's AABB (fridge: the whole fridge AABB)
  opened_cab(m)        drawer joint > 0.45 - 0.1      closed_cab(m)    |joint| < 0.15
  opened_fridge(m)     door joint > 1.22 - 0.15       closed_fridge(m) |joint| < 0.15
  holding(obj) / not_holding()                         from the grasp rule (EB: the Fetch suction gripper)
  human_at(entity)     pelvis within 2.0 m of the entity and facing it within 1.57 rad (EB robot_at, for feedback only)
Interaction rules (the hand is not simulated):
  grasp   a grasp event (hand, target) is active, the object is not inside a closed drawer / fridge (EB's pick precondition),
          and the hand point (palm centre) is within r of the target (object: distance to
          its world AABB; handle marker: distance to the marker) for k consecutive frames -> the object is carried by the hand
          (kinematic, no collision), or the handle is held: the joint then follows the hand (dq = J.dh / |J|^2, clamped)
  release a release event (hand) -> a carried object becomes dynamic again with the velocity it had in the hand over the last
          frame (so it can be thrown) and falls / rests where physics puts it
Articulated furniture is kinematic (as in EB): the body collides with it but only the rules move its joints."""
import dataclasses, pickle, re
from pathlib import Path
import numpy as np

from ..paths import EB_HABITAT as EBH, EB_DATA as DATA

RCAD = DATA / "replica_cad/replicaCAD.scene_dataset_config.json"
OBJ_PATHS = [DATA / "objects/ycb/configs", DATA / "replica_cad/configs/objects"]
ART_THRESH, OPEN_CAB, OPEN_CAB_THRESH, OPEN_FRIDGE = 0.15, 0.45, 0.1, 1.22
AT_DIST, AT_ANGLE = 2.0, 1.57
HANDS = {"right": ("right_wrist", "right_middle1"), "left": ("left_wrist", "left_middle1")}


@dataclasses.dataclass
class Rules:
    r_obj: float = 0.10       # m, palm centre to the object's AABB
    r_handle: float = 0.12    # m, palm centre to a handle marker
    k: int = 3                # consecutive frames (30 fps)


def load_episode(eval_set, ep_id):
    d = pickle.load(open(EBH / "datasets" / f"{eval_set}.pickle", "rb")); n = d["idx_to_name"]; ep = dict(d["all_eps"][int(ep_id)])
    ep["rigid_objs"] = [(n[ni], np.asarray(d["all_transforms"][ti], float)) for ni, ti in ep["rigid_objs"]]
    ep["ao_states"] = {n[k]: v for k, v in ep["ao_states"].items()}
    ep["markers"] = [dict(name=n[a], offset=np.asarray(o, float), link=n[l], ao=n[h]) for a, _, o, l, h in ep["markers"]]
    ep["episode_id"] = str(ep_id)
    return ep


_OBJECT_SETS = None
# names of added ReplicaCAD objects that would share a YCB category (gelatin box, cups, bowl, plate): kept distinct
FRL_NAMES = {"box": "cardboard box", "cup": "coffee mug", "bowl": "serving bowl", "plate": "dinner plate", "remote-control": "remote control",
             "shoebox": "shoe box", "stool": "stool"}


def category(handle):
    """PDDL type of an object, from EB's dataset/configs/dataset.yaml object_sets (substring rules), e.g.
    '016_pear_:0000' -> 'pear', '007_tuna_fish_can_:0000' -> 'can', '072-d_toy_airplane_:0000' -> 'toy airplane'."""
    global _OBJECT_SETS
    if _OBJECT_SETS is None:
        import yaml
        _OBJECT_SETS = yaml.safe_load(open(EBH / "dataset/configs/dataset.yaml"))["object_sets"]
    base = handle.split("_:")[0]
    if base.startswith("frl_apartment_"):           # ReplicaCAD household objects added by a task: 'frl_apartment_shoe_01' -> 'shoe'
        c = re.sub(r"_\d+$", "", base[len("frl_apartment_"):]).replace("_", " ")
        return FRL_NAMES.get(c, c)
    for s in _OBJECT_SETS:
        if s["name"].isupper():                     # union sets such as CLUTTER_OBJECTS
            continue
        if any(x in base for x in s["included_substrings"]) and not any(x in base for x in s.get("excluded_substrings") or []):
            return s["name"]
    return re.sub(r"^\d+(-[a-z])?_", "", base)


def eb_category(handle):
    """The category EB's goals quantify over (EXISTS / FORALL): EB's own object_sets first, also for ReplicaCAD objects that EB
    lists (frl_apartment_bowl_03 -> 'bowl'); category() renames those for the text shown to the agent ('serving bowl')."""
    category(handle)                                 # loads _OBJECT_SETS
    base = handle.split("_:")[0]
    for s in _OBJECT_SETS:
        if s["name"].isupper():
            continue
        if any(x in base for x in s["included_substrings"]) and not any(x in base for x in s.get("excluded_substrings") or []):
            return s["name"]
    return category(handle)


_NAMES = None


def eb_name(name):
    """EB's readable name of a receptacle / handle (dataset/configs/instructions.yaml name_mappings, the names EB's own agents
    see in their action list), e.g. 'receptacle_aabb_Tbl2_Top1_frl_apartment_table_02' -> 'brown table'; None if EB has none."""
    global _NAMES
    if _NAMES is None:
        import yaml
        _NAMES = yaml.safe_load(open(EBH / "dataset/configs/instructions.yaml"))["name_mappings"]
    return _NAMES.get(name) or _NAMES.get("receptacle_aabb_" + name)


def v3(x):
    return np.array([x[0], x[1], x[2]], float)


def _aabb_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))


class EBWorld:
    def __init__(self, eval_set, ep_id, video=False, settle_s=2.0, rules=None, third_res=(480, 640), contacts=False, extra=None):
        from .hp_bridge import half_physics_env
        import habitat_sim
        self.hs = habitat_sim; self.rules = rules or Rules()
        self.ep = load_episode(eval_set, ep_id)
        self.extra = list(extra or [])     # objects added to the episode by the task: dict(template, pos, yaw)
        self.scene = Path(self.ep["scene_id"]).name.replace(".scene_instance.json", "")
        self.env = half_physics_env(scene_id=self.scene, scene_dataset_config=RCAD, video_enabled=video,
                                    third_person_resolution=third_res, collect_metric_contacts=contacts)
        self.sim = self.env.sim; self.rom = self.sim.get_rigid_object_manager(); self.aom = self.sim.get_articulated_object_manager()
        self.objs = self._add_objects()
        for ao in self._furniture():
            ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            for mid in ao.existing_joint_motor_ids:
                ao.remove_joint_motor(mid)
        self._set_ao_states()
        self.receptacles = self._receptacles()
        self.markers = {m["name"]: m for m in self.ep["markers"]}
        for m in self.markers.values():
            m["ao_obj"] = self.aom.get_object_by_handle(m["ao"]); m["link_id"] = m["ao_obj"].get_link_id_from_name(m["link"])
            m["q_idx"] = m["ao_obj"].get_link_joint_pos_offset(m["link_id"])
        self._apply_start_preds()
        self._local_bb = {h: self.rom.get_object_by_handle(h).root_scene_node.cumulative_bb for h in self.objs}
        self.held = {"right": None, "left": None}
        self._hand_links = {h: [self.env.agent.get_link_id_from_name(n) for n in ls] for h, ls in HANDS.items()}
        self._install_hook()
        if settle_s:
            self.settle(settle_s, park_human=True)
        self.initial = self.snapshot()

    # ---------------- building ----------------
    def _add_objects(self):
        import magnum as mn
        otm = self.sim.get_object_template_manager()
        for p in OBJ_PATHS:
            otm.load_configs(str(p))
        handles = []
        for name, T in sorted(self.ep["rigid_objs"], key=lambda x: x[0]):          # RearrangeSim adds them sorted by name
            tpl = next(str(p / name) for p in OBJ_PATHS if (p / name).is_file())
            ro = self.rom.add_object_by_template_handle(tpl)
            ro.transformation = mn.Matrix4([[T[j][i] for j in range(4)] for i in range(4)])
            ro.linear_velocity = mn.Vector3.zero_init(); ro.angular_velocity = mn.Vector3.zero_init()
            handles.append(ro.handle)
        for e in self.extra:          # ReplicaCAD household objects (basket, bins, cloth, shoes, cushions, stool, box, ...): dynamic
            tpl = next(str(p / (e["template"] + ".object_config.json")) for p in OBJ_PATHS if (p / (e["template"] + ".object_config.json")).is_file())
            ro = self.rom.add_object_by_template_handle(tpl)
            ro.rotation = mn.Quaternion.rotation(mn.Deg(float(e.get("yaw", 0.0))), mn.Vector3(0, 1, 0))
            ro.translation = mn.Vector3(*map(float, e["pos"]))
            ro.linear_velocity = mn.Vector3.zero_init(); ro.angular_velocity = mn.Vector3.zero_init()
            handles.append(ro.handle)
        return handles

    def _furniture(self):
        return [self.aom.get_object_by_handle(h) for h in self.aom.get_object_handles()
                if self.aom.get_object_by_handle(h).object_id != self.env.agent.object_id]

    def _set_ao_states(self):
        for h, js in self.ep["ao_states"].items():
            o = self.aom.get_object_by_handle(h); q = o.joint_positions
            for link, v in js.items():
                q[o.get_link_joint_pos_offset(int(link))] = v
            o.joint_positions = q

    def _apply_start_preds(self):
        """EB sets the episode's start_preds at reset (PddlSimState.set_state with the domain's sample values); in all EB-Habitat
        subsets these are only opened_fridge (1.22) / closed_fridge (0.0)."""
        val = {"opened_fridge": OPEN_FRIDGE, "closed_fridge": 0.0, "opened_cab": OPEN_CAB, "closed_cab": 0.0}
        for p in self.ep.get("start_preds") or []:
            m = re.match(r"(\w+)\((\w+)\)", p.replace(" ", ""))
            if not m or m.group(1) not in val:
                raise NotImplementedError(f"start predicate {p}")
            self.set_marker_js(m.group(2), val[m.group(1)])

    def _receptacles(self):
        """As RearrangeSim._create_recep_info: world AABB of every receptacle_aabb in the scene (episode objects ignored).
        Like habitat, only the min and max corners are transformed, so for furniture turned by an angle that is not a multiple
        of 90 deg the box is squashed (a sofa turned ~14 deg: 0.10 m deep instead of ~0.5 m). Kept for EB's own predicates
        (tasks taken unchanged from EB); the Inhabit conditions and what the agent is told use the receptacle's own box (`recep_frames`,
        `in_recep`)."""
        from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
        out = {}; self.recep_frames = {}; self._moving_receps = {}
        for r in find_receptacles(self.sim, ignore_handles=list(self.objs)):
            if r.parent_link is not None and r.parent_link >= 0:              # on a drawer / door link: it moves
                self._moving_receps[r.name] = r
            T = r.get_global_transform(self.sim)
            b = np.stack([v3(T.transform_point(r.bounds.min)), v3(T.transform_point(r.bounds.max))])
            out[r.name] = (b.min(0), b.max(0))
            M = np.array([[T[c][rr] for c in range(4)] for rr in range(4)], float)        # magnum is column-major: T[col][row]
            b0, b1 = v3(r.bounds.min), v3(r.bounds.max)                       # some receptacles (drawers) store min / max swapped
            self.recep_frames[r.name] = (M, np.linalg.inv(M), np.minimum(b0, b1), np.maximum(b0, b1))
        return out

    def recep_frame(self, name):
        """(M, M^-1, local lo, local hi) of a receptacle's own box now: one on a drawer / door link follows the link (a drawer's
        surface moves out with it); EB's `receptacles` stay as computed at the start (as EB's RearrangeSim)"""
        name = name.split("_:")[0]; r = self._moving_receps.get(name)
        if r is None:
            return self.recep_frames[name]
        T = r.get_global_transform(self.sim); M = np.array([[T[c][rr] for c in range(4)] for rr in range(4)], float)
        b0, b1 = v3(r.bounds.min), v3(r.bounds.max)
        return M, np.linalg.inv(M), np.minimum(b0, b1), np.maximum(b0, b1)

    def recep_centre(self, name):
        """centre of a receptacle's own box now (habitat)"""
        M, _, lo, hi = self.recep_frame(name); return 0.5 * (lo + hi) @ M[:3, :3].T + M[:3, 3]

    def _recep_axes(self, name):
        """(M, Mi, lo, hi, up axis, its sign, the two horizontal axes) of a receptacle's own box: the local axis closest to the
        world's up is its height (most receptacles are y-up; the fridge shelves are not)"""
        M, Mi, lo, hi = self.recep_frame(name)
        up = int(np.argmax(np.abs(M[1, :3]))); sg = 1.0 if M[1, up] > 0 else -1.0
        return M, Mi, lo, hi, up, sg, [a for a in range(3) if a != up]

    def in_recep(self, name, p, m=0.0, below=0.0, y=True):
        """point(s) p (..., 3) habitat inside receptacle `name`'s own (rotated) box: horizontally within its bounds +- m,
        in height from its bottom - `below` to its top (y=False: horizontally only)"""
        M, Mi, lo, hi, up, sg, hz = self._recep_axes(name); p = np.asarray(p, float)
        q = p @ Mi[:3, :3].T + Mi[:3, 3]; ok = np.ones(q.shape[:-1], bool)
        for a in hz:
            ok &= (q[..., a] >= lo[a] - m) & (q[..., a] <= hi[a] + m)
        if y:
            ok &= ((q[..., up] >= lo[up] - below) & (q[..., up] <= hi[up])) if sg > 0 else ((q[..., up] >= lo[up]) & (q[..., up] <= hi[up] + below))
        return ok

    def recep_corners(self, name):
        """the 4 corners (habitat xyz, at the bottom of the region) of a receptacle's own box"""
        M, _, lo, hi, up, sg, (a, b) = self._recep_axes(name)
        c = []
        for u, v in ((lo[a], lo[b]), (hi[a], lo[b]), (hi[a], hi[b]), (lo[a], hi[b])):
            x = np.zeros(3); x[a], x[b] = u, v; x[up] = lo[up] if sg > 0 else hi[up]; c.append(x)
        return np.array(c) @ M[:3, :3].T + M[:3, 3]

    def recep_heights(self, name):
        """(bottom, top) world height of a receptacle's own box now"""
        M, _, lo, hi, up, sg, _ = self._recep_axes(name)
        ys = [(np.where(np.arange(3) == up, v, 0.5 * (lo + hi)) @ M[:3, :3].T + M[:3, 3])[1] for v in (lo[up], hi[up])]
        return float(min(ys)), float(max(ys))

    def recep_box_now(self, name):
        """(lo, hi) world AABB of a receptacle's own box now (all 8 corners; for distances and `extent` heights)"""
        M, _, lo, hi = self.recep_frame(name)
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ M[:3, :3].T + M[:3, 3]
        return c.min(0), c.max(0)

    def recep_size(self, name):
        _, _, lo, hi, _, _, (a, b) = self._recep_axes(name); return float(hi[a] - lo[a]), float(hi[b] - lo[b])

    def recep_dist_xz(self, name, p):
        """horizontal distance (m) from point(s) p to a receptacle's own box (0 inside)"""
        M, Mi, lo, hi, up, sg, hz = self._recep_axes(name); p = np.asarray(p, float); q = p @ Mi[:3, :3].T + Mi[:3, 3]
        d = [np.maximum(0, np.maximum(lo[a] - q[..., a], q[..., a] - hi[a])) for a in hz]
        return np.hypot(d[0], d[1])

    def recep_nearest(self, name, p):
        """the point of a receptacle's own box nearest to p (habitat)"""
        M, Mi, lo, hi = self.recep_frame(name); q = np.asarray(p, float) @ Mi[:3, :3].T + Mi[:3, 3]
        return np.clip(q, lo, hi) @ M[:3, :3].T + M[:3, 3]

    def obj_nearest(self, h, p):
        """the point of an object's own (rotated) box nearest to p (habitat)"""
        import magnum as mn
        bb, T = self._local_bb[h], self.rom.get_object_by_handle(h).transformation
        q = v3(T.inverted().transform_point(mn.Vector3(*map(float, p))))
        return v3(T.transform_point(mn.Vector3(*map(float, np.clip(q, v3(bb.min), v3(bb.max))))))

    def on_top_own(self, obj, recep):
        """on_top on the receptacle's own box (with on_top_tol below it), not EB's squashed AABB"""
        return bool(self.in_recep(recep, self.obj_pos(obj), below=self.on_top_tol))

    # ---------------- state ----------------
    def obj_pos(self, h):
        return v3(self.rom.get_object_by_handle(h).translation)

    def obj_aabb(self, h):
        import magnum as mn
        bb, T = self._local_bb[h], self.rom.get_object_by_handle(h).transformation
        c = np.array([v3(T.transform_point(mn.Vector3(x, y, z))) for x in (bb.min[0], bb.max[0])
                      for y in (bb.min[1], bb.max[1]) for z in (bb.min[2], bb.max[2])])
        return c.min(0), c.max(0)

    def obj_footprint(self, h):
        """the 4 corners (habitat xyz, at the bottom) of an object's own box, rotated with it (the world AABB of a
        rotated box is larger than the box, so a ball lying beside a turned cardboard box counted as inside it)"""
        import magnum as mn
        bb, T = self._local_bb[h], self.rom.get_object_by_handle(h).transformation
        return np.array([v3(T.transform_point(mn.Vector3(x, bb.min[1], z))) for x, z in
                         ((bb.min[0], bb.min[2]), (bb.max[0], bb.min[2]), (bb.max[0], bb.max[2]), (bb.min[0], bb.max[2]))])

    def in_obj_box(self, h, p, shrink=0.01, up=0.02):
        """point p (habitat) inside object h's own (rotated) box: in its local frame, x / z within the box shrunk by `shrink`
        (the walls), height from its bottom to its top + `up`"""
        import magnum as mn
        bb, T = self._local_bb[h], self.rom.get_object_by_handle(h).transformation
        q = v3(T.inverted().transform_point(mn.Vector3(*map(float, p))))
        lo, hi = v3(bb.min), v3(bb.max)
        return bool(lo[0] + shrink <= q[0] <= hi[0] - shrink and lo[2] + shrink <= q[2] <= hi[2] - shrink and lo[1] - up <= q[1] <= hi[1] + up)

    def marker_T(self, name):
        import magnum as mn
        m = self.markers[name]
        node = m["ao_obj"].get_link_scene_node(m["link_id"])
        return node.absolute_transformation() @ mn.Matrix4.translation(mn.Vector3(*m["offset"]))

    def marker_pos(self, name):
        return v3(self.marker_T(name).translation)

    def marker_js(self, name):
        m = self.markers[name]; return float(m["ao_obj"].joint_positions[m["q_idx"]])

    def set_marker_js(self, name, v):
        m = self.markers[name]; o = m["ao_obj"]; q = o.joint_positions; q[m["q_idx"]] = float(v); o.joint_positions = q

    def marker_limits(self, name):
        m = self.markers[name]; lo, hi = m["ao_obj"].joint_position_limits; return float(lo[m["q_idx"]]), float(hi[m["q_idx"]])

    def marker_container_aabb(self, name):
        """Region for in(obj, marker): the drawer link's AABB, or the whole fridge (habitat-lab _is_object_inside)."""
        m = self.markers[name]
        if name.startswith("fridge"):
            from habitat.sims.habitat_simulator.sim_utilities import get_ao_global_bb
            bb = get_ao_global_bb(m["ao_obj"])
        else:
            node = m["ao_obj"].get_link_scene_node(m["link_id"])
            bb = self.hs.geo.get_transformed_bb(node.cumulative_bb, node.transformation)
        return v3(bb.min), v3(bb.max)

    def hand_point(self, hand):
        a, b = (v3(self.env.agent.get_link_scene_node(i).absolute_translation) for i in self._hand_links[hand])
        return 0.5 * (a + b)

    def hand_T(self, hand):
        return self.env.agent.get_link_scene_node(self._hand_links[hand][0]).absolute_transformation()

    def pelvis(self):
        """Pelvis position (habitat Y-up) and facing direction (unit XZ vector, from the hip line: forward = across x up)."""
        ag = self.env.agent
        if not hasattr(self, "_hips"):
            self._hips = [ag.get_link_id_from_name(n) for n in ("left_hip", "right_hip")]
        a = v3(ag.get_link_scene_node(self._hips[0]).absolute_translation) - v3(ag.get_link_scene_node(self._hips[1]).absolute_translation)
        f = np.array([-a[2], 0.0, a[0]])
        return v3(ag.translation), f / (np.linalg.norm(f) + 1e-9)

    def snapshot(self, human=False):
        """Objects + articulated furniture; with human=True also the humanoid (root, joints, velocities) and what the hands
        hold, so a segment can be rolled back and retried from exactly the same state."""
        s = dict(objs={h: self.rom.get_object_by_handle(h).transformation for h in self.objs},
                 aos={o.handle: list(o.joint_positions) for o in self._furniture()})
        if human:
            ag = self.env.agent
            s["human"] = dict(T=ag.transformation, q=list(ag.joint_positions), qd=list(ag.joint_velocities),
                              v=ag.root_linear_velocity, w=ag.root_angular_velocity, mt=ag.motion_type)
            s["held"] = {h: (dict(v) if v else None) for h, v in self.held.items()}
            s["dyn"] = {h: (self.rom.get_object_by_handle(h).motion_type, self.rom.get_object_by_handle(h).collidable) for h in self.objs}
        return s

    def restore(self, snap):
        """Back to a snapshot. With the human: the state is set twice with one physics step (body parked 50 m below) in between,
        because the physics engine keeps contact state from the last try: after a try that ended with the body inside furniture,
        the next first-segment try was placed up to 1.9 m off its frame 0."""
        self._restore(snap)
        if "human" in snap:
            import magnum as mn
            self.env.agent.translation = mn.Vector3(0.0, -50.0, 0.0); self.sim.step_physics(1.0 / 120); self._restore(snap)

    def _restore(self, snap):
        D = self.hs.physics.MotionType.DYNAMIC
        for h, T in snap["objs"].items():
            o = self.rom.get_object_by_handle(h); o.motion_type, o.collidable = snap.get("dyn", {}).get(h, (D, True))
            o.transformation = T
            if o.motion_type == D:
                o.linear_velocity = o.angular_velocity = type(o.linear_velocity)(0, 0, 0)
        for h, q in snap["aos"].items():
            self.aom.get_object_by_handle(h).joint_positions = q
        self.held = {h: (dict(v) if v else None) for h, v in snap.get("held", {"right": None, "left": None}).items()}
        if "human" in snap:
            ag, s = self.env.agent, snap["human"]
            ag.motion_type = s["mt"]; ag.transformation = s["T"]; ag.joint_positions = s["q"]; ag.joint_velocities = s["qd"]
            ag.root_linear_velocity = s["v"]; ag.root_angular_velocity = s["w"]

    # ---------------- predicates ----------------
    on_top_tol = 0.0          # m the object centre may be below the region's bottom; 0 = EB's definition (Inhabit tasks: 0.06)

    def on_top(self, obj, recep):
        """EB: the object centre inside the receptacle's region. With on_top_tol > 0 also up to that far below it: the sofa's region
        starts above its cushion, so 70% of the objects EB itself puts on the sofa are below it."""
        lo, hi = self.receptacles[recep.split("_:")[0]]; lo = lo - np.array([0.0, self.on_top_tol, 0.0]); p = self.obj_pos(obj)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    def inside(self, obj, marker):
        lo, hi = self.marker_container_aabb(marker); p = self.obj_pos(obj); return bool(np.all(p >= lo) and np.all(p <= hi))

    def holding(self, obj):
        return any(v is not None and v["kind"] == "obj" and v["target"] == obj for v in self.held.values())

    def not_holding(self):
        return not any(v is not None and v["kind"] == "obj" for v in self.held.values())

    def opened_cab(self, m):
        return self.marker_js(m) > OPEN_CAB - OPEN_CAB_THRESH

    def closed_cab(self, m):
        return abs(self.marker_js(m)) < ART_THRESH

    def opened_fridge(self, m):
        return self.marker_js(m) > OPEN_FRIDGE - ART_THRESH

    def closed_fridge(self, m):
        return abs(self.marker_js(m)) < ART_THRESH

    def entity_pos(self, name):
        if name in self.markers:
            return self.marker_pos(name)
        if name.split("_:")[0] in self.receptacles:
            lo, hi = self.receptacles[name.split("_:")[0]]; return 0.5 * (lo + hi)
        return self.obj_pos(name)

    def human_at(self, name, dist=AT_DIST, angle=AT_ANGLE):
        p, f = self.pelvis(); d = self.entity_pos(name) - p; d[1] = 0; n = np.linalg.norm(d)
        return bool(n <= dist and np.arccos(np.clip(d @ f / (n + 1e-9), -1, 1)) <= angle)

    def eval_atom(self, name, args):
        fn = {"on_top": self.on_top, "in": self.inside, "holding": self.holding, "not_holding": self.not_holding,
              "opened_cab": self.opened_cab, "closed_cab": self.closed_cab, "opened_fridge": self.opened_fridge,
              "closed_fridge": self.closed_fridge, "robot_at": self.human_at, "robot_at_obj": self.human_at}[name]
        return fn(*args)

    def true_predicates(self):
        """All true object / articulation predicates over the episode's objects, receptacles and markers (EB's
        get_true_predicates without the robot ones), as strings 'on_top(a,b)'."""
        out = []
        cabs = [m for m in self.markers if m.startswith("cab")]; fr = [m for m in self.markers if m.startswith("fridge")]
        for o in self.objs:
            out += [f"on_top({o},{r})" for r in self.receptacles if self.on_top(o, r)]
            out += [f"in({o},{m})" for m in self.markers if self.inside(o, m)]
            out += [f"holding({o})"] if self.holding(o) else []
        out += [f"opened_cab({m})" for m in cabs if self.opened_cab(m)] + [f"closed_cab({m})" for m in cabs if self.closed_cab(m)]
        out += [f"opened_fridge({m})" for m in fr if self.opened_fridge(m)] + [f"closed_fridge({m})" for m in fr if self.closed_fridge(m)]
        return sorted(out)

    def goal_satisfied(self, expr=None, bind=None):
        """Evaluate EB's goal_preds (AND / NAND / OR / NOR, EXISTS / FORALL over object categories) on the current state."""
        expr = self.ep["goal_preds"] if expr is None else expr; bind = dict(bind or {})
        if isinstance(expr, str):
            m = re.match(r"(\w+)\((.*)\)", expr.replace(" ", ""))
            args = [bind.get(a, a) for a in m.group(2).split(",") if a]
            return self.eval_atom(m.group(1), args)
        q, ins = expr.get("quantifier"), expr.get("inputs") or []
        if q in ("EXISTS", "FORALL") and ins:
            v, rest = ins[0], dict(expr, inputs=ins[1:])
            cands = [o for o in self.objs if eb_category(o) == v["expr_type"]]
            res = (self.goal_satisfied(rest, {**bind, v["name"]: c}) for c in cands)
            return any(res) if q == "EXISTS" else all(res)
        subs = [self.goal_satisfied(s, bind) for s in expr["sub_exprs"]]
        t = expr["expr_type"].upper(); v = (all if t in ("AND", "NAND") else any)(subs)
        return not v if t in ("NAND", "NOR") else v

    # ---------------- execution + rules ----------------
    def _install_hook(self):
        rt, world = self.env._runtime, self

        class HP:
            def __getattr__(self, k):
                return getattr(rt.hp, k)

            def hp_step(self, *a, **kw):
                out = rt.hp.hp_step(*a, **kw); world._on_frame(); return out
        self._hp = rt.hp; self.env._runtime = dataclasses.replace(rt, hp=HP())
        self._events, self._frame, self._fps, self.log = [], 0, 30.0, []; self._count = {h: 0 for h in HANDS}; self.trace = []

    def _attach(self, hand, target):
        if target in self.markers:
            self.held[hand] = dict(kind="art", target=target, h_prev=self.hand_point(hand), t=self._frame)
        else:
            o = self.rom.get_object_by_handle(target); o.motion_type = self.hs.physics.MotionType.KINEMATIC; o.collidable = False
            self.held[hand] = dict(kind="obj", target=target, offset=self.hand_T(hand).inverted() @ o.transformation, t=self._frame)

    def _release(self, hand):
        h = self.held[hand]
        if h and h["kind"] == "obj":
            o = self.rom.get_object_by_handle(h["target"]); o.motion_type = self.hs.physics.MotionType.DYNAMIC; o.collidable = True
            v, w = h.get("vel", (np.zeros(3), np.zeros(3)))                 # leaves the hand with the hand's velocity (throwing)
            o.linear_velocity, o.angular_velocity = type(o.linear_velocity)(*map(float, v)), type(o.linear_velocity)(*map(float, w))
        self.held[hand] = None

    def _follow(self, hand):
        h = self.held[hand]
        if h["kind"] == "obj":
            T = self.hand_T(hand) @ h["offset"]; self.rom.get_object_by_handle(h["target"]).transformation = T
            M = np.array([[T[c][r] for c in range(4)] for r in range(4)])          # row-major 4x4 of the carried object
            if "M" in h:                                                        # its velocity over the last frame
                dR = M[:3, :3] @ h["M"][:3, :3].T; a = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                w = ax / (2 * np.sin(a)) * a if a > 1e-6 else np.zeros(3)
                h["vel"] = ((M[:3, 3] - h["M"][:3, 3]) * self._fps, w * self._fps)
            h["M"] = M
            return
        name, p = h["target"], self.hand_point(hand); dh = p - h["h_prev"]; h["h_prev"] = p
        q0, p0 = self.marker_js(name), self.marker_pos(name); eps = 1e-3
        self.set_marker_js(name, q0 + eps); J = (self.marker_pos(name) - p0) / eps
        lo, hi = self.marker_limits(name)
        self.set_marker_js(name, float(np.clip(q0 + J @ dh / max(J @ J, 1e-9), lo, hi)))

    def _on_frame(self):
        t = self._frame / self._fps; row = dict(t=round(t, 3))
        for hand in HANDS:
            for e in self._events:
                if e["kind"] == "release" and e["hand"] == hand and self.held[hand] and abs(t - e["t"]) < 0.5 / self._fps:
                    self._release(hand)
            if self.held[hand] is not None:
                self._follow(hand); row[hand] = dict(held=self.held[hand]["target"]); continue
            act = [e for e in self._events if e["kind"] == "grasp" and e["hand"] == hand and e["t0"] <= t <= e["t1"]]
            if not act:
                self._count[hand] = 0; continue
            tg = act[0]["target"]; p = self.hand_point(hand)
            if tg not in self.markers and any(self.inside(tg, m) and not (self.opened_fridge(m) if m.startswith("fridge") else self.opened_cab(m))
                                              for m in self.markers):
                row[hand] = dict(target=tg, d=9.9, blocked="inside a closed drawer / fridge"); self._count[hand] = 0; continue
            d = float(np.linalg.norm(p - self.marker_pos(tg))) if tg in self.markers else _aabb_dist(p, *self.obj_aabb(tg))
            r = self.rules.r_handle if tg in self.markers else self.rules.r_obj
            self._count[hand] = self._count[hand] + 1 if d <= r else 0; row[hand] = dict(target=tg, d=round(d, 4))
            if self._count[hand] >= self.rules.k:
                self._attach(hand, tg); row[hand]["grasped"] = True
        self.log.append(row); self._frame += 1; self._record()

    def _record(self):
        """Per-frame trace for the agent's read-only measurements: palms, object positions and boxes, handle positions and joints."""
        rot = {}
        for o in self.objs:                                     # orientation too (saved per try for later analysis)
            q = self.rom.get_object_by_handle(o).rotation; rot[o] = np.array([q.scalar, *q.vector], float)
        self.trace.append(dict(palm={h: self.hand_point(h) for h in HANDS}, obj={o: self.obj_pos(o) for o in self.objs}, rot=rot,
                               box={o: self.obj_aabb(o) for o in self.objs},          # what the grasp rule measures against
                               mpos={m: self.marker_pos(m) for m in self.markers}, js={m: self.marker_js(m) for m in self.markers}))

    def execute(self, action, events=(), settle_s=1.0, sink=None):
        """Run one SMPL-X chunk with interaction events, then let the world settle (human held still).
        events: dict(kind="grasp", hand="right"|"left", target=<object handle | marker name>, t0, t1)  (seconds in the chunk)
                dict(kind="release", hand, t)
        Returns HalfPhysicsEnv info plus 'rule_log' (per frame: distance to the grasp target, grasp / held)."""
        self._events, self._frame, self._fps, self.log = list(events), 0, float(action.get("fps", 30.0)), []
        self._count = {h: 0 for h in HANDS}; self.trace = []
        self.env.set_video_frame_sink(sink)
        _, _, _, info = self.env.step(action)
        if settle_s:
            self.settle(settle_s, sink=sink)
        info["rule_log"] = self.log; info["trace"] = self.trace; return info

    def settle(self, seconds, park_human=False, sink=None):
        """Advance physics with the human frozen (kinematic), using half-physics' own gravity handling (world gravity is 0;
        hp applies g to dynamic objects every 1/120 s substep). Carried objects keep following the frozen hand."""
        ag = self.env.agent; mt, T = ag.motion_type, ag.transformation
        ag.motion_type = self.hs.physics.MotionType.KINEMATIC
        if park_human:
            import magnum as mn
            ag.translation = mn.Vector3(0, -20, 0)
        self.sim.set_gravity([0.0, 0.0, 0.0])
        for i in range(int(round(seconds * 120))):
            self._hp._step_physics_with_movable_object_gravity(self.sim, 1.0 / 120)
            if i % 4 == 3 and hasattr(self, "trace"):
                self._record()
            if sink is not None and i % 4 == 3:
                self.env._update_cameras(); o = self.sim.get_sensor_observations()
                sink(np.asarray(o["ego_rgb"])[..., :3], np.asarray(o["third_person_rgb"])[..., :3])
        if park_human:
            ag.transformation = T
        ag.motion_type = mt

    def navmesh(self, radius=0.2):
        """Pathfinder for the scene (ReplicaCAD ships no navmesh for the v3 scenes): recomputed with static furniture."""
        pf = self.sim.pathfinder
        if not pf.is_loaded:
            ns = self.hs.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = radius; ns.include_static_objects = True
            self.sim.recompute_navmesh(pf, ns)
        return pf

    # ---------------- cameras (video) ----------------
    def render_at(self, eye, at, uuid="third_person_rgb"):
        """Render sensor `uuid` from a world pose (eye, look-at); HalfPhysicsEnv re-aims its follow camera every frame."""
        import magnum as mn
        node = self.sim._sensors[uuid]._sensor_object.node
        world = mn.Matrix4.look_at(mn.Vector3(*map(float, eye)), mn.Vector3(*map(float, at)), mn.Vector3(0, 1, 0))
        node.transformation = node.parent.absolute_transformation().inverted() @ world
        return np.asarray(self.hs.Simulator.get_sensor_observations(self.sim)[uuid])[..., :3]

    def side_eye(self, at, d, dist=2.5, h=0.8):
        """Eye `dist` m from `at` perpendicular to the horizontal direction d (raised h), on the side with more free space,
        pulled in 0.3 m in front of the first wall / furniture hit."""
        import magnum as mn
        best = None
        for s in (1, -1):
            v = s * np.array([-d[2], 0, d[0]]) * dist + [0, h, 0]
            hits = self.sim.cast_ray(self.hs.geo.Ray(mn.Vector3(*at), mn.Vector3(*v)), max_distance=1.0)
            free = hits.hits[0].ray_distance if hits.has_hits() else 1.0
            if best is None or free > best[0]:
                best = (free, at + v * max(free - 0.3 / dist, 0.25))
        return best[1]

    def close(self):
        self.env.close()
