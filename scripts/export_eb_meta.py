"""Export per-episode EB-Habitat metadata to <data>/eb_meta/<subset>/<episode_id>.json (episode_id = index in EB's dataset
pickle; EB's iterator shuffles, so every reset is matched by its id): instruction, goal, true predicates, entities, rigid /
articulated objects, receptacle AABBs, handle markers and the robot start pose. Inhabitant reads only `robot_start` from these
files: the person starts where EB's robot starts. Run it once per subset used by the benchmark (all six):
  for s in base common_sense complex_instruction spatial_relationship visual_appearance long_horizon; do
      python scripts/export_eb_meta.py --eval_set $s --n 50; done
Note: for episodes that start with the fridge open, the exported fridge marker position is EB's stale closed-door value;
inhabitant.world.eb_world computes the marker from the opened door and does not use the exported one.
Needs EmbodiedBench importable and its EB-Habitat data in place (scripts/install_eb_habitat.sh) and a GPU (EB creates a
renderer); it changes the working directory to EB's eb_habitat package directory, as EB-Habitat requires."""
import argparse, json, os, sys, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inhabitant.paths import EMBODIEDBENCH_ROOT, EB_HABITAT as EBH, EB_META      # noqa: E402
sys.path.insert(0, str(EMBODIEDBENCH_ROOT))


def v3(x):
    return [round(float(a), 3) for a in (x[0], x[1], x[2])]


def aabb_world(obj):
    """World-frame axis-aligned box of a rigid / articulated object (from its root scene node)."""
    try:
        bb = obj.root_scene_node.cumulative_bb; tr = obj.transformation
        cs = [tr.transform_point(c) for c in (bb.min, bb.max, type(bb.min)(bb.min[0], bb.max[1], bb.min[2]),
                                               type(bb.min)(bb.max[0], bb.min[1], bb.max[2]))]
        a = np.array([[c[0], c[1], c[2]] for c in cs]); return dict(min=v3(a.min(0)), max=v3(a.max(0)))
    except Exception as e:
        return dict(error=str(e)[:80])


def export_meta(env, task, sim, ep_idx):
    import magnum as mn
    pddl = task.pddl
    meta = dict(episode=ep_idx, instruction=env.episode_language_instruction,
                goal=str(getattr(task, "_goal_expr", None)), scene=getattr(env.current_episode(), "scene_id", None))
    try:
        meta["true_predicates"] = sorted(str(p) for p in pddl.get_true_predicates())
    except Exception as e:
        meta["true_predicates_error"] = str(e)[:200]
    ents = {}
    for name, e in pddl.all_entities.items():
        ents[name] = dict(type=str(e.expr_type.name))
    meta["entities"] = ents
    rom = sim.get_rigid_object_manager(); objs = {}
    for h in rom.get_object_handles():
        o = rom.get_object_by_handle(h)
        objs[h] = dict(position=v3(o.translation), motion=str(o.motion_type).split(".")[-1], aabb=aabb_world(o))
    meta["rigid_objects"] = objs
    aom = sim.get_articulated_object_manager(); arts = {}
    for h in aom.get_object_handles():
        ao = aom.get_object_by_handle(h)
        links = {}
        for lid in ao.get_link_ids():
            try:
                node = ao.get_link_scene_node(lid); tr = node.absolute_transformation()
                links[ao.get_link_name(lid)] = dict(id=int(lid), position=v3(tr.translation),
                                                    joint_type=str(ao.get_link_joint_type(lid)).split(".")[-1])
            except Exception:
                pass
        arts[h] = dict(position=v3(ao.translation), joint_positions=[round(float(q), 3) for q in ao.joint_positions], links=links)
    meta["articulated_objects"] = arts
    try:
        from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
        meta["receptacles"] = {r.unique_name: dict(parent=r.parent_object_handle, link=getattr(r, "parent_link", None))
                               for r in find_receptacles(sim)}
    except Exception as e:
        meta["receptacles_error"] = str(e)[:200]
    meta["episode_id"] = getattr(env.current_episode(), "episode_id", None)     # = index in the dataset pickle
    meta["receptacle_aabb"] = {k: dict(min=v3(r.min), max=v3(r.max)) for k, r in sim.receptacles.items()}
    meta["markers"] = {k: dict(position=v3(m.get_current_position()), js=round(float(m.get_targ_js()), 4))
                       for k, m in sim.get_all_markers().items()}
    try:
        ag = sim.articulated_agent; meta["robot_start"] = dict(base_pos=v3(ag.base_pos), base_rot=float(ag.base_rot))
    except Exception as e:
        meta["robot_error"] = str(e)[:200]
    return meta



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval_set", default="base"); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    os.chdir(EBH)
    from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
    env = EBHabEnv(eval_set=a.eval_set, start_epi_index=0, resolution=256)
    task = env.env.env.env._env.task; sim = task._sim
    out = EB_META / a.eval_set; out.mkdir(parents=True, exist_ok=True)
    for k in range(a.n):
        env.reset(); meta = export_meta(env, task, sim, k)
        json.dump(meta, open(out / f"{meta['episode_id']}.json", "w"), indent=1)
        print(f"{k}: episode {meta['episode_id']}: {meta['instruction']}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
