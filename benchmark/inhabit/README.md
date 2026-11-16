# Inhabit

75 household tasks in the ReplicaCAD apartments of EB-Habitat, carried out with a human body. Every task starts from the
scene of one EB-Habitat episode (its furniture layout, its YCB objects and the start pose of the agent) and changes it: a new
instruction, added household objects, moved objects, drawer / door states, and its own success conditions.

| ids | level | tasks | conditions per task |
|---|---|---|---|
| `low_01` ... `low_30` | low | 30 | one (one task: two): put, throw, kick or push one thing |
| `mid_01` ... `mid_20` | mid | 20 | mostly two: two things, or a drawer / the fridge and a thing |
| `high_01` ... `high_15` | high | 15 | three |
| `body_01` ... `body_10` | body | 10 | two, one of them on the body itself: sit, lie, kneel |

Eight tasks (`low_18`-`low_20`, `mid_18`-`mid_20`, `high_14`, `high_15`) are EB-Habitat episodes taken unchanged, with
EB's own goal (`eb_original`).

## Task file

```json
{
 "id": "mid_17",
 "level": "mid",
 "categories": ["articulated", "drawer"],
 "physics_levels": [0, 0],
 "base": {"set": "visual_appearance", "id": 31},
 "instruction": "Could you put the banana away in the left drawer in the kitchen?",
 "conditions": [{"id": "c1", "kind": "opened", "marker": "cab_push_point_7"},
                {"id": "c2", "kind": "inside", "obj": "011_banana_:0000", "container": "cab_push_point_7"}],
 "order": [["c1", "c2"]],
 "joints": {"cab_push_point_7": 0.0},
 "add_objects": [],
 "objects": {}
}
```

| field | meaning |
|---|---|
| `id`, `level`, `categories` | name and grouping; the evaluation tables group by `level` and by each category |
| `physics_levels` | per condition, how much the task relies on physics: 0 = pick and place, 1 = the object flies (throw), 2 = the body moves it without grasping (kick, push) |
| `base` | the EB-Habitat episode whose scene is used: subset (`base`, `common_sense`, `complex_instruction`, `spatial_relationship`, `visual_appearance`, `long_horizon`) and index in EB's dataset file |
| `instruction` | what the agent is told; the conditions are hidden from it |
| `conditions` | all must hold (below); each has an `id` and a `kind` |
| `order` | pairs `[a, b]`: condition `a` must be achieved before `b` |
| `joints` | handle id -> joint value set at the start (drawer: metres, fridge door: radians; 0 = closed) |
| `add_objects` | ReplicaCAD household objects added to the scene: `template` (e.g. `frl_apartment_box`), `pos` (habitat x, y, z; y is up), `yaw` (degrees), `handle` (the id it gets) |
| `objects` | object id -> new position (habitat x, y, z) for objects of the base episode that are moved |
| `start` | optional: `{"base_pos": [x, y, z], "base_rot": radians}` where the person starts; default: where EB's robot starts in the base episode |
| `eb_original` | true for an EB-Habitat episode taken unchanged (condition kind `eb`; EB's definition of "on a surface" without tolerance) |

Object ids are the simulator's handles (`011_banana_:0000`), surfaces are named by any substring of the receptacle name
(`receptacle_aabb_Tbl1_Top1_frl_apartment_table_01`, or just `Tbl1_Top1`), handles of drawers and of the fridge by their marker
(`cab_push_point_7`, `fridge_push_point`).

## Conditions

The rules are in `inhabitant/world/bench_goal.py`. Object conditions are checked on the world after the last try has
settled; body conditions must hold continuously for their duration at some time of the committed motion.

| kind | arguments | holds when |
|---|---|---|
| `on_top` | `obj`, `surface` | the object's centre is inside the surface's region (up to 0.06 m below it) and it is not in a hand |
| `inside` | `obj`, `container` (a handle) | the object is inside that drawer / the fridge, not in a hand |
| `in_container` | `obj`, `in` (an object) | the object's centre is inside the basket's / box's own box, not in a hand |
| `under` | `obj`, `surface` | the object lies on the floor (< 0.15 m) below the surface's region |
| `opened`, `closed` | `marker` | the drawer / fridge door is open / closed (EB's thresholds) |
| `throw` | `obj` + a target (`surface`, or `in`) | the object ends on / in the target, and the palm was at least 1.0 m (horizontal) from the target when it let go |
| `kick` | `obj` + a target (`surface`, with `under: true` for below it) | the object ends at the target; a foot touched it after it last left a hand and it was not held afterwards |
| `push` | `obj` + a target (`surface`, `under`, `in`) or `away: {from, dist}` | the object ends at the target (or `dist` m from where it started); the body touched it and it was never in a hand |
| `sit_on` | `surface` | pelvis over the surface, 0.05-0.35 m above it, trunk within 45 deg of upright, for 1 s |
| `lie_on` | `surface` (default the floor) | trunk more than 60 deg from upright, pelvis and head low on the surface, for 1 s |
| `kneel` | - | a knee on the floor, the pelvis 0.25-0.8 m above it, for 1 s |
| `held_together` | `a`, `b` | both objects are in the hands at the same moment |
| `avoid` | `thing` | the body never touches it |
| `eb` | - | the base episode's own EB-Habitat goal |

`bench_goal.py` has more kinds that the 75 tasks do not use (`at`, `stand_on`, `squat`, `jump`, `jump_onto`, `circle`,
`handstand`, `dance_on`, `hand_only`); they work in task files of your own (`--spec my_task.json`).
