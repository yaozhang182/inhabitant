"""Report of one episode (a run directory with log.json): the committed segments joined into one video with a caption per
segment, statistics of the run and of the motion's quality, and a Markdown table of every segment (its tries, rejected calls,
time) and every try (intent, root, result).
Outputs <run dir>/final.mp4, final_motion.npz (the executed motion of the kept version), report.md and stats.json.
  python -m inhabitant.eval.episode_report <run dir> [<run dir> ...]"""
import json, re, sys
from pathlib import Path
import numpy as np

from ..paths import BENCHMARK


def caption(frame, lines):
    from PIL import Image, ImageDraw
    bar = Image.new("RGB", (frame.shape[1], 18 * len(lines) + 8), (0, 0, 0)); g = ImageDraw.Draw(bar)
    for i, s in enumerate(lines):
        g.text((6, 4 + 18 * i), s, fill=(255, 255, 0) if i == 0 else (255, 255, 255))
    return np.concatenate([np.asarray(bar), frame], 0)


def join_video(run, d):
    import imageio
    prog = d.get("best") or d["program"]; n = len(prog); w = imageio.get_writer(run / "final.mp4", fps=30, quality=7, macro_block_size=1)
    for i, s in enumerate(prog):
        lines = [f"segment {i + 1}/{n} (try {s['try_n']}): {s['intent'][:110]}", f"result: {s['summary'][:120]}"]
        frames = imageio.get_reader(run / s["video"])
        for f in frames:
            w.append_data(caption(f, lines))
    w.close()


def final_motion(run, d):
    """The executed motion of the kept version as one file: joints (T,22,3, Z-up world, 30 fps), the executed SMPL of
    HalfPhysicsEnv (transl, global_orient, body_pose), the first frame of each segment, and each segment's intent and generator request.
    Needs the per-try tries/tNNN_joints.npz; skipped otherwise."""
    prog = d.get("best") or d["program"]
    files = [run / (s.get("joints") or s["video"].replace(".mp4", "_joints.npz")) for s in prog]
    if not prog or not all(f.exists() for f in files):
        print("final_motion.npz skipped: no per-try joints"); return
    zs = [np.load(f) for f in files]; keys = ["exe", "exe_transl", "exe_global_orient", "exe_body_pose"]
    out = {k[4:] if k != "exe" else "joints": np.concatenate([z[k] for z in zs]) for k in keys if all(k in z for z in zs)}
    np.savez_compressed(run / "final_motion.npz", **out, fps=30.0, seg_start=np.cumsum([0] + [len(z["exe"]) for z in zs[:-1]]),
                        intent=np.array([s["intent"] for s in prog]), request=np.array([json.dumps(s.get("request") or {}) for s in prog]))
    print(run / "final_motion.npz")


def calls(run, d):
    """[(name, args, reply, t_think, t_run)] per tool call, from the log's tool_log (t_think = the time the agent took before the
    call, t_run = the time the tool took). Logs without a tool_log: from the messages, without times."""
    if d.get("tool_log"):
        return [(c["name"], c["args"], c["reply"], c["t_think"], c["t_run"]) for c in d["tool_log"]]
    ms = d["messages"][1:]
    return [(ms[k]["tool_call"]["name"], ms[k]["tool_call"]["args"], ms[k + 1]["content"], None, None) for k in range(0, len(ms) - 1, 2)]


FOOT_PARTS = ("foot", "ankle", "toe")


FURN = ("sofa", "chair_01", "table_01", "table_02", "table_03", "tvstand", "counter", "cabinet", "beanbag")


def task_spec(d):
    """the task of a run: saved in its log, else the benchmark file, else None"""
    if d.get("spec"):
        return d["spec"]
    f = BENCHMARK / f"{d['id']}.json"
    return json.load(open(f)) if f.exists() else None


def body_moved(d):
    """objects the task asks the body to move without grasping (kick / push conditions): touching and
    moving them is neither a collision nor a disturbed object"""
    spec = task_spec(d)
    if spec is None:
        return set()
    return {c["obj"] for c in spec["conditions"] if c["kind"] in ("kick", "push")}


def task_contacts(d):
    """contact names (substrings) the task itself asks for: its body conditions"""
    spec = task_spec(d); out = set()
    if spec is None:
        return set()
    for c in spec["conditions"]:
        if c["kind"] in ("sit_on", "lie_on", "stand_on", "handstand", "dance_on", "jump_onto"):
            surf = c.get("surface", "floor")
            out |= {"floor", "rug", "mat"} if surf == "floor" else {f for f in FURN if f in surf or (f == "sofa" and "Sofa" in surf)}
        if c["kind"] in ("kneel", "squat"):
            out |= {"floor", "rug", "mat"}
    return out


NEAR = 1.2                                   # m: disturbed objects end this close to the pelvis path (arm reach + falling)


def quality(run, d):
    """Motion quality of the kept version, per segment and in total:
    - collisions: body contacts with anything (walls and the pillar too) except (feet / ankles on rugs or the floor),
      (hands / wrists on the floor) and (hands / wrists with what that
      segment grasps (objects, handles: pushing a door with the hand is fine) or holds; touching the furniture an object is put on
      counts) and (any part with what the task asks the body to rest on: the furniture of a sit_on / lie_on / stand_on /
      handstand / dance_on / jump_onto condition, the floor and rugs for kneel / lying or a handstand on the
      floor); parsed from each try's contact list;
    - blocked: max horizontal distance between the executed pelvis and the submitted (IK-placed) one per segment; > 0.3 m counts
      as a blocked segment (the motion asked to go through furniture or a wall and was stopped; bending to reach alone gives
      0.1-0.3 m);
    - grasps / releases over all tries and in the kept segments (SUCCESS / MISSED lines of the try results);
    - disturbed: objects moved (> 5 cm, the "Changes in the world" line of each kept try) that the segment did not grasp, that
      were not in a hand at its start and that end within NEAR m of the pelvis path: knocked over, pushed, dropped by bumping
      (the side effects a goal rule does not see; far objects that move are settling by themselves, e.g. fruit
      rolling in the sink); objects the task asks to kick / push are neither collisions nor disturbed;
    - skate ratio (the main foot-skate number): GMD's foot skating ratio (Karunratanakul et al. ICCV 2023, code
      calculate_skating_ratio; also OmniControl): feet = joints 10 / 11; contact = the foot below 5 cm in this and the next frame;
      skating = in contact and the horizontal speed > 0.5 m/s, and its 5-frame average too; ratio = frames where either foot
      skates / all frames, over the kept segments (executed joints, 30 fps). Height above the floor; a segment whose feet never
      come below 10 cm (standing on furniture) uses its lowest foot height as the ground (our addition);
    - skate (mean skate speed, secondary): mean horizontal speed (m/s) of a foot joint while it is within 3 cm of its lowest
      height in the segment (close to PhysDiff's Skate). Returns (rows, totals)."""
    prog = d.get("best") or d["program"]; tries = {t["n"]: t for t in d["tries"]}; rows = []; holding = {}
    required = task_contacts(d); moved = body_moved(d)
    for k, s in enumerate(prog, 1):
        before = set(holding.values())                          # what the hands held at the start (a handle stays held across commits)
        for e in s["events"]:
            if e["kind"] == "grasp":
                holding[e["hand"]] = e["target"]
            else:
                holding.pop(e["hand"], None)
        t = tries[s["try_n"]]; line = next((l for l in t.get("feedback", "").split("\n") if l.startswith("- Contacts:")), "- Contacts: none.")
        ok = {e.get("target") for e in s["events"]} | before   # grasp targets (objects, handles); NOT the surface a thing is put on
        held = re.findall(r"in your \w+ hand|(\w+_:\d+)", t.get("feedback", ""))
        bad = []
        for c in line[len("- Contacts: "):].rstrip(".").split("; "):
            m = re.match(r"(\S+) - (.+?) ([\d.]+)-([\d.]+) s$", c)       # names may have spaces ("a wall or pillar")
            if not m:
                continue
            part, other = m.group(1), m.group(2)
            if any(f in part for f in FOOT_PARTS) and ("rug" in other or "floor" in other or "stair" in other):
                continue
            if ("wrist" in part or "hand" in part) and other in ("the floor", "a wall / the floor"):   # a hand on the floor (picking something up)
                continue
            if any(r in other for r in required):                # contact the task asks for (sit / lie / stand on it, kneel ...)
                continue
            if any(o.split("_:")[0] in other for o in moved):     # the object the task asks to kick / push
                continue
            if ("wrist" in part or "hand" in part) and any(o and (other in o or o.split("_:")[0] in other) for o in ok | set(held)):
                continue
            bad.append(f"{part}-{other} {m.group(3)}-{m.group(4)} s")
        f = run / (s.get("joints") or s["video"].replace(".mp4", "_joints.npz")); sk = gmd = None
        if f.exists():
            J = np.load(f)["exe"]; v = []
            for j in (10, 11):
                h = J[:, j, 2] - J[:, j, 2].min(); sp = np.r_[0, np.linalg.norm(np.diff(J[:, j, :2], axis=0), axis=1) * 30]
                v += list(sp[h < 0.03])
            sk = round(float(np.mean(v)), 3) if v else 0.0
            F = J[:, [10, 11]]; h = F[..., 2]; h = h - (h.min() if h.min() > 0.10 else 0.0)
            vel = np.linalg.norm(np.diff(F[..., :2], axis=0), axis=-1) * 30                    # (T-1, 2) m/s
            avg = np.stack([np.convolve(vel[:, i], np.ones(5) / 5, mode="same") for i in (0, 1)], 1)   # centred 5-frame average, zero padding (uniform_filter1d, mode constant)
            contact = (h[:-1] < 0.05) & (h[1:] < 0.05)
            skating = (contact & (vel > 0.5) & (avg > 0.5)).any(1); gmd = (int(skating.sum()), len(skating))
        ch = next((l for l in t.get("feedback", "").split("\n") if l.startswith("- Changes in the world:")), "")
        path = np.asarray(t.get("path") or [[np.inf, np.inf]], float)   # pelvis path (map); objects far from it moved by themselves
        dist = [o for o, pl0, x, y in re.findall(r"([\w-]+_:\d+): (.+?) -> [^;]*?\(map ([-\d.]+), ([-\d.]+),", ch)
                if o not in ok and o not in moved and not pl0.startswith("in your") and np.hypot(*(path - [float(x), float(y)]).T).min() < NEAR]
        rows.append(dict(segment=k, collisions=bad, skate=sk, skate_frames=gmd, disturbed=dist))
        if f.exists():                                           # blocked: the submitted motion went where the body could not
            z = np.load(f); dev = np.linalg.norm(z["exe"][:, 0, :2] - z["ref"][:, 0, :2], axis=-1)
            rows[-1]["blocked_m"] = round(float(dev.max()), 3)
    sks = [r["skate"] for r in rows if r["skate"] is not None]
    bl = [r["blocked_m"] for r in rows if r.get("blocked_m") is not None]
    ev = lambda ts, kind, res: sum(len(re.findall(rf"^- {kind} .*: {res}", t.get("feedback", ""), re.M)) for t in ts)
    kept = [tries[s["try_n"]] for s in prog]
    return rows, dict(collisions=sum(len(r["collisions"]) for r in rows), segments_with_collision=sum(bool(r["collisions"]) for r in rows),
                      skate_mps=round(float(np.mean(sks)), 3) if sks else None,
                      skate_ratio=(round(sum(r["skate_frames"][0] for r in rows if r["skate_frames"]) /
                                         max(sum(r["skate_frames"][1] for r in rows if r["skate_frames"]), 1), 4)
                                   if any(r["skate_frames"] for r in rows) else None),
                      disturbed_objects=sorted({o for r in rows for o in r["disturbed"]}),
                      blocked_segments=sum(b > 0.3 for b in bl), blocked_max_m=round(max(bl), 3) if bl else None,
                      grasp_ok_all=ev(d["tries"], "grasp", "SUCCESS"), grasp_missed_all=ev(d["tries"], "grasp", "MISSED"),
                      release_ok_all=ev(d["tries"], "release", "SUCCESS"), release_missed_all=ev(d["tries"], "release", "MISSED"),
                      grasp_missed_kept=ev(kept, "grasp", "MISSED"), release_missed_kept=ev(kept, "release", "MISSED"))


def stats(run, d, cs, segs, cur):
    """Totals of a run, written to <run>/stats.json. An optional <run>/usage.json is merged in (e.g. {"tokens": n} for an
    external agent whose token count the log does not have)."""
    from collections import Counter
    st = dict(id=f"{d['eval_set']}_{d['id']}", success=d["success"], segments=len(d.get("best") or d["program"]), tries=len(d["tries"]),
              rejected=sum(len(x["rejected"]) for x in segs) + len(cur["rejected"]), calls=len(cs),
              calls_by_tool=dict(Counter(c[0] for c in cs)),
              think_s=round(sum(c[3] or 0 for c in cs)), tool_s=round(sum(c[4] or 0 for c in cs)),
              try_s=round(sum(c[4] or 0 for c in cs if c[0] == "generate_segment")))
    st["total_s"] = st["think_s"] + st["tool_s"]
    st["quality"] = quality(run, d)[1]
    if d.get("usage"):                       # the built-in loop: tokens and cost summed over the LLM calls
        u = d["usage"]           # tokens = the final conversation size; billed totals in usage
        st.update(tokens=u.get("context_tokens", 0), usage=u, cost_usd=round(u.get("cost_usd") or 0.0, 3))
    if (run / "usage.json").exists():
        st.update(json.load(open(run / "usage.json")))
    json.dump(st, open(run / "stats.json", "w"), indent=1)
    return st


def fb_lines(fb):
    keep = []
    for l in fb.split("\n"):
        if re.match(r"- (grasp|release|Body contacts|Contacts|Changes|Drawer|Door)", l) or "MISSED" in l:
            keep.append(l[2:])
    return keep


def fmt_t(s):
    return "-" if s is None else f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def main(run):
    run = Path(run); d = json.load(open(run / "log.json")); cs = calls(run, d)
    tries = {t["n"]: t for t in d["tries"]}
    new = lambda: dict(tries=[], rejected=[], notes=[], other=0, t=0.0)
    segs = []; cur = new(); n_try = 0; goal_seen = False
    for name, args, reply, tt, tr in cs:
        cur["t"] += (tt or 0) + (tr or 0)
        if name == "generate_segment":
            if reply.startswith("Try "):
                n_try += 1; cur["tries"].append((tries[n_try], tt, tr))
            else:
                cur["rejected"].append((args.get("intent", ""), reply.split("\n", 2)[1][2:] if "\n" in reply else reply))
        elif name == "commit" and reply.startswith("Committed"):
            cur["refine"] = goal_seen; goal_seen = goal_seen or cur["tries"][-1][0]["goal"]; segs.append(cur); cur = new()
        elif name == "retry_from":
            cur["notes"].append(f"retry_from({args.get('i')}): {reply.splitlines()[0]}")
        else:
            cur["other"] += 1
    total_t = sum((a or 0) + (b or 0) for *_, a, b in cs)
    st = stats(run, d, cs, segs, cur)
    L = [f"# {d['eval_set']} {d['id']}: {d['instruction']}", "",
         f"- Model: {d['model']}, tag {d['args'].get('tag')}; goal: `{d['goal']}`",
         f"- Result: {'success' if d['success'] else 'failure'}; final version {len(d.get('best') or d['program'])} segments ({len(segs)} commits); "
         f"{len(d['tries'])} tries ({sum(len(s['rejected']) for s in segs) + len(cur['rejected'])} more calls rejected, not executed); "
         f"{len(cs)} tool calls; total time {fmt_t(total_t)}",
         f"- Time: thinking {fmt_t(st['think_s'])}, tools {fmt_t(st['tool_s'])} (tries {fmt_t(st['try_s'])}); calls {st['calls_by_tool']}"
         + (f"; tokens {st['tokens']:,}" if st.get("tokens") else ""),
         f"- Motion quality (final version): {st['quality']['collisions']} collisions ({st['quality']['segments_with_collision']} segments with a "
         f"collision; feet on rugs / the floor and hands on what they grasp or hold do not count); foot skate {st['quality']['skate_mps']} m/s "
         f"(mean horizontal speed of a foot within 3 cm of the floor); per segment: "
         + "; ".join(f"segment {r['segment']} skate {r['skate']}" + (f" collisions {', '.join(r['collisions'])}" if r["collisions"] else "") for r in quality(run, d)[0]),
         f"- Video: `{run / 'final.mp4'}` (per segment the committed try; the caption is its intent and result; left: follow camera, "
         f"right: side camera)", "",
         "## Commits in order", "",
         "| # | segment | intent (the agent's words) | tries | rejected | time | committed result |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(segs):
        p = s["tries"][-1][0]
        L.append(f"| {i + 1} | {p['segment']} | {p['ans']['intent']} | {len(s['tries'])} | {len(s['rejected'])} | "
                 f"{fmt_t(s['t'])} | {p['summary']}{' **GOAL**' if p['goal'] else ''} |")
    L += ["", "Time = from the previous commit to this one: the agent's thinking plus the tools' run time.", "",
          "## Tries of each segment", ""]
    for i, s in enumerate(segs + ([cur] if cur["tries"] or cur["notes"] else [])):
        p = s["tries"][-1][0] if s["tries"] else None; last = i >= len(segs)
        L += [f"### Commit {i + 1} (segment {p['segment'] if p else '-'}): {p['ans']['intent'] if p else ''}" if not last else "### Not committed at the end", ""]
        for x in s["notes"]:
            L.append(f"- {x}")
        for intent, why in s["rejected"]:
            L.append(f"- rejected (not executed): {why}")
        for t, tt, tr in s["tries"]:
            r = t["ans"]["root"]; ev = "; ".join(f"{e['kind']} {e.get('target') or e.get('surface_name') or e.get('surface')}" for e in t["ans"]["events"])
            mark = " **<- committed**" if not last and t is p else ""
            L.append(f"- try {t['n']}{mark} (thinking {fmt_t(tt)}, run {fmt_t(tr)}): {t['ans']['intent']}")
            L.append(f"  - generated root motion: forward {r[0]:+.2f} m, left {r[1]:+.2f} m, turn {r[2]:+.0f} deg{'; events: ' + ev if ev else ''}")
            rq = json.dumps(t["ans"].get("request") or {}, ensure_ascii=False)
            L.append(f"  - request: `{rq if len(rq) <= 600 else rq[:600] + ' ...'}`")
            L.append(f"  - result: {t['summary']}")
            for l in fb_lines(t["feedback"]):
                L.append(f"  - {l}")
        L.append("")
    (run / "report.md").write_text("\n".join(L)); print(run / "report.md")
    join_video(run, d); print(run / "final.mp4")
    final_motion(run, d)


if __name__ == "__main__":
    bad = []
    for r in sys.argv[1:]:                     # many runs in one call: a failing run must not stop the rest
        try:
            main(r)
        except Exception:  # noqa: BLE001
            import traceback; traceback.print_exc(); bad.append(r)
    if bad:
        print(f"FAILED ({len(bad)}): {bad}")
