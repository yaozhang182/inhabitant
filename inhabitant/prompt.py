"""What the model is told: the system prompt, the tool definitions and Kimodo's documentation.

The prompt is the same for every task: how to move with Kimodo (its inputs and frames), the task framing, the tools, the rules of
the world, the body's size, and Kimodo's own documentation (docs/kimodo_docs.md). `build_prompt()` returns the system prompt
and the tool definitions (OpenAI function format; the LLM client and the MCP server convert them).

  python -m inhabitant.agent --print_prompt"""
from pathlib import Path

from .motion.motion_exec import body_text, JOINTS

DOCS = Path(__file__).resolve().parent / "docs/kimodo_docs.md"


def kimodo_docs():
    """Kimodo's documentation as the agent reads it (docs/kimodo_docs.md without its header comment)."""
    return "\n".join(DOCS.read_text().split("-->", 1)[1].splitlines()).strip()


# ----------------------------------------------------------------- prompt
KIMODO_API = """# Moving with Kimodo

You move by asking a motion generator, Kimodo, for one segment of motion at a time. Its own documentation is at the end
("Kimodo's documentation"). generate_segment takes Kimodo's inputs:
- text: a description of the motion, or a list of descriptions that are generated one after another;
- seconds: its length (one per text); a segment is 0.5 to 8 s, each text at most 10 s;
- constraints: Kimodo's constraint list (its JSON format), in Kimodo's frame of this segment;
- seed (default 0; the same inputs with the same seed give the same motion; its execution in the simulator can still
  differ by a few cm between identical tries), post (post-processing, default true), cfg_weight ([text, constraint], default [2, 2]).
Kimodo's frame of a segment: the origin is on the floor below your pelvis at the start of the segment, +Z is the direction you
face at the start, Y is up, X is to your left; frame_indices count frames from the start of the segment, 30 per second. The
other tools report positions as forward / left of your pelvis and heights above the floor: Kimodo's Z / X / Y. The skeleton
has 22 joints, in this order (one rotation each in local_joints_rot): {joints}. local_joints_rot are SMPL-X's local joint
rotations: with all of them zero the body stands in a T-pose facing +Z, arms along X. global_root_heading [cos θ, sin θ] is
the facing as its (Z, X) components: θ = 0 faces +Z, θ = 90° faces +X (turned to the left). A hand constraint fixes the wrist
(position and rotation), a foot constraint the ankle and the toe joint; both also fix the root's position, height and facing
at that frame.
Every segment starts from your actual pose: Kimodo gets the last frames of what your body did as a full-body keyframe just
before the segment. The generated motion is fitted to your body, placed in the scene and executed in a physics simulator."""


AGENT = """## Your task

You are the mind of a person in an apartment, and you must complete a household task by moving your own body. Nothing happens
by itself: every movement is a segment of motion that you {make}. You act only through the tools.
Move as a real person would: do not bump into furniture, walls, plants or other things (only your hands touch what
you grasp or hold), and do not let your feet slide on the floor.

### Tools
{tools}

### The room
- In the top-down map, '.' means a body 0.2 m in radius around the pelvis fits there; your shoulders and swinging
  arms reach further.
- The state lists all places, objects and handles with their distance and direction from you.

### Rules of this world
- Pick up: a grasp event names a hand, an object id and a time window. If, inside the window, the centre of that palm comes
  within 0.10 m of the object (its bounding box) for 0.1 s, the object attaches to that hand and moves with it.
- Put down: with the object in hand, a release event names the hand, where it should go (a surface id, an object id such as a
  basket or box, or "floor") and the time you let go. The object then leaves the hand with the velocity it had there (so it
  can be thrown), falls and rests where physics puts it; it counts as on a surface if it ends inside that surface's region,
  and in a basket or box if it ends inside it.
- Drawers and the fridge: grasp the handle id (palm within 0.12 m of the handle), then move that hand along the given opening
  direction; the drawer / door follows the hand while it is held. Release it afterwards (name any surface; not checked).
- After the motion of a segment the world runs on for 0.3 s (1.5 s after a release) with your body held still, so that things
  settle; the times of a try include it.
- One hand holds one thing. Whether the task is accomplished is checked after every {unit}.

### Your body
{body}"""

INSPECT_LINE = ("- inspect(queries): read-only measurements of the world and of your tries: where things and body parts are (now or "
                "over a try), distances over time,\n  body contacts, the size of any thing{pose}. It changes nothing.\n")
CLOSED_TOOLS = """- generate_segment({sig}): {who} one segment from the last committed state, executed in a
  physics simulator (furniture and walls stop the body). It returns whether the goal is now satisfied and a video of the try
  (left: a camera following you; right: a side camera looking at you and at the target of the segment); measure anything else
  with inspect. Nothing is kept until you commit: every try starts again from the last committed state.
- commit(): keeps the last try; the world continues from where it ended. Returns the new state near you
  (within 2 m; get_state() gives all of it).
- get_state(): the current state.
- retry_from(i): undo: puts the world back to the start of committed segment i and drops segments i..end; then try / commit
  as before.
{inspect}The episode ends when a segment you commit completes the task.

After each try, look at the result and the video and change the {inputs} until the segment does what you intended; then commit it."""

S = {"type": "string"}; N = {"type": "number"}
EVENT = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["grasp", "release"]}, "hand": {"type": "string", "enum": ["right", "left"]},
                                          "target": {"type": "string", "description": "grasp: object or handle id"},
                                          "surface": {"type": "string", "description": "release: surface id, object id (basket, box) or 'floor'"},
                                          "t0": N, "t1": N, "t": N}, "required": ["kind", "hand"]}
TEXT_OR_LIST = {"description": "a description of the motion, or a list of descriptions generated one after another"}
SECS = {"description": "length in seconds, or a list (one per text); 0.5-8 s in all"}


def fn(name, desc, props=None, req=()):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props or {}, "required": list(req)}}}


def gen_props():
    """Arguments of generate_segment: Kimodo's inputs plus the grasp / release events."""
    p = {"intent": {"type": "string", "description": "what this segment should do"}}
    p.update(text=TEXT_OR_LIST, seconds=SECS, seed={"type": "integer", "description": "default 0"})
    p.update(constraints={"type": "array", "items": {"type": "object"}, "description": "Kimodo's constraint list (its JSON format), "
                          "in Kimodo's frame of this segment"},
             post={"type": "boolean", "description": "Kimodo's post-processing (default true)"},
             cfg_weight={"type": "array", "items": N, "description": "[text, constraint] guidance weights (default [2, 2])"})
    p["events"] = {"type": "array", "items": EVENT, "description": "grasp: hand, target, t0, t1; release: hand, surface, t (s within the segment)"}
    return p


def inspect_tool():
    pose = (" pose(t, segment?): the pose your body executed at time t of your last try (or of committed segment k), in Kimodo's "
            "format as a fullbody constraint (root_positions, local_joints_rot) in Kimodo's frame of your next segment, with its facing; "
            "fk(root_positions, local_joints_rot) or fk(pose: a constraint): the positions of the 22 joints of poses given in Kimodo's "
            "format (Kimodo's frame and body), and where wrists, palms, feet, pelvis and head end up when the pose is fitted to your body. "
            "where (now or at one moment) also gives the position in Kimodo's frame of your next segment (X, Y, Z), the frame of "
            "pose and of your constraints.")
    return fn("inspect", "Read-only measurements of the world and of your tries (it changes nothing). `queries` is a list; each item "
              "is {\"fn\": ..., arguments}: where(id, t0?, t1?) where a thing or a body part is: without times now (and on which "
              "surface, in which hand; a handle's open / closed state), with t0 at that moment, with t0 and t1 every 0.1 s (for the "
              "pelvis also the facing); distance(a, b, t0?, t1?) distance between two things (body parts, objects, surfaces, "
              "furniture: to the nearest point of its box) every 0.1 s, with the direction at the closest moment; contacts(t0?, "
              "t1?, part?, other?) body contacts; extent(id) corners and heights of any thing: a surface, an object, a piece of "
              "furniture (any name that appears in the tool outputs) or a handle (the door / drawer part it moves, with its "
              "joint)." + pose + " Ids: objects, surfaces and handles as in the state, names of furniture from the tool outputs, or "
              "body parts pelvis, head, left_palm, right_palm, left_elbow, right_elbow, left_shoulder, right_shoulder, left_knee, "
              "right_knee, left_foot, right_foot. t, t0, t1: seconds within your last try (the try since the last commit, else the "
              "last committed segment); add segment=k to measure committed segment k instead.",
              {"queries": {"type": "array", "items": {"type": "object"}}}, ["queries"])


def build_prompt():
    """-> (system prompt, tool definitions); the same for every task."""
    api = KIMODO_API.format(joints=", ".join(JOINTS))
    sig = "intent, text, seconds, constraints, seed, post, cfg_weight, events"
    insp = INSPECT_LINE.format(pose=", an executed pose in Kimodo's format and the joint positions of a pose in Kimodo's format")
    tools_text = CLOSED_TOOLS.format(sig=sig, who="Kimodo makes", inspect=insp, inputs="inputs")
    system = api + "\n\n" + AGENT.format(make="have Kimodo make (above)", tools=tools_text, unit="try", body=body_text())
    system += "\n\n# Kimodo's documentation\n\n" + kimodo_docs()
    tools = [fn("generate_segment", "Kimodo makes one segment (0.5-8 s) from your inputs; it is executed from the last committed state and "
                "the result and a video are returned.", gen_props(), ["intent", "text", "seconds"]),
             fn("commit", "Keep the last try; the world continues from where it ended. Returns the new state near you (get_state() gives all of it)."),
             fn("retry_from", "Undo: put the world back to the start of committed segment i and drop segments i..end.",
                {"i": {"type": "integer"}}, ["i"]),
             fn("get_state", "The current state (positions of you, every surface, object and handle); after an uncommitted try, "
                             "the state at the end of that try."),
             inspect_tool()]
    return system, tools


def prompt_text():
    """The system prompt followed by the tool definitions, as printed by --print_prompt."""
    import json
    system, tools = build_prompt()
    return system + "\n\n## Tool definitions\n\n```json\n" + json.dumps([t["function"] for t in tools], indent=1) + "\n```"
