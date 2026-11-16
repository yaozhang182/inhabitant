"""Tables of one or more sets of runs (a tag directory holds one run directory per task).
Per episode: success; sub-goal completion (share of the task's conditions satisfied after its final try); blocked segments
(executed pelvis > 0.3 m from the submitted one: the motion went into furniture or a wall); share of grasps / releases missed
over all tries; executed tries until the goal was first reached; calls, tokens, time; collisions, disturbed objects (moved
without being grasped), foot skate ratio (GMD's, the main number) and mean skate speed; segments. All but the first two come
from stats.json, written by episode_report.py.
Grouped by level and by task category; plus success within k tries.
  python -m inhabitant.eval.summarize <tag dir> [more tag dirs] [--md out.md]"""
import argparse, json
from pathlib import Path
import numpy as np

from ..paths import BENCHMARK

LEVELS = ["low", "mid", "high", "body", "custom"]


def episode(d, tag_dir):
    L = json.load(open(d / "log.json")); st = json.load(open(d / "stats.json")) if (d / "stats.json").exists() else {}
    spec = L.get("spec") or json.load(open(BENCHMARK / f"{L['id']}.json"))
    prog = L.get("best") or L["program"]; tries = L["tries"]
    first = next((i + 1 for i, t in enumerate(tries) if t.get("goal")), None)
    last = tries[prog[-1]["try_n"] - 1] if prog else (tries[-1] if tries else {})
    conds = (last.get("bench") or {}).get("conds", {})
    sub = (sum(v["ok"] for v in conds.values()) / len(conds)) if conds else float(bool(L["success"]))
    ok = bool(L["success"])
    if st.get("rejudged"):                           # a run whose verdict was corrected afterwards: stats.json carries the
        ok = bool(st["success"]); sub = 1.0 if ok else sub    # verdict and the reason, log.json stays as run
    q = st.get("quality", {})
    return dict(id=L["id"], level=spec.get("level", "custom"), cats=spec.get("categories", []), success=ok, sub=sub, first_try=first,
                calls=st.get("calls", L.get("llm_calls")), tokens=st.get("tokens"), minutes=round(st.get("total_s", 0) / 60, 1),
                tries=len(tries), segments=len(prog),
                collisions=q.get("collisions"), skate=q.get("skate_mps"), skr=q.get("skate_ratio"), blocked=q.get("blocked_segments"),
                disturbed=len(q["disturbed_objects"]) if "disturbed_objects" in q else None,
                gmiss=(q["grasp_missed_all"] / max(q["grasp_missed_all"] + q["grasp_ok_all"], 1)) if "grasp_ok_all" in q else None,
                rmiss=(q["release_missed_all"] / max(q["release_missed_all"] + q["release_ok_all"], 1)) if "release_ok_all" in q else None)


def table(rows, title):
    if not rows:
        return ""
    f = lambda k: np.mean([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else float("nan")
    return (f"| {title} | {len(rows)} | {100 * f('success'):.0f}% | {100 * f('sub'):.0f}% | {f('calls'):.0f} | {f('tokens') / 1000:.0f}k | "
            f"{f('minutes'):.0f} | {f('segments'):.1f} | {f('collisions'):.1f} | {f('blocked'):.1f} | {f('disturbed'):.1f} | {100 * f('skr'):.1f}% | {f('skate'):.2f} | "
            f"{100 * f('gmiss'):.0f}% | {100 * f('rmiss'):.0f}% |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--md"); A = ap.parse_args()
    out = []
    HEAD = ("| group | episodes | success | sub-goals | calls | tokens | min | segments | collisions | blocked segs | disturbed objects | skate ratio (GMD) | skate m/s | "
            "grasps missed (all tries) | releases missed |\n" + "|---" * 15 + "|")
    for tag in A.tags:
        td = Path(tag)
        rows = [episode(d, td) for d in sorted(td.iterdir()) if (d / "log.json").exists()]
        out += [f"\n## {td.name}\n", "### per episode\n",
                "| episode | level | success | sub-goals | first goal at try | calls | tokens | min | segments | collisions | blocked | disturbed | skate ratio | skate |",
                "|---" * 14 + "|"]
        for r in rows:
            out.append(f"| {r['id']} | {r['level']} | {'yes' if r['success'] else 'no'} | {100 * r['sub']:.0f}% | {r['first_try']} | {r['calls']} | "
                       f"{(r['tokens'] or 0) // 1000}k | {r['minutes']} | {r['segments']} | {r['collisions']} | {r['blocked']} | {r['disturbed']} | {r['skr']} | "
                       f"{r['skate']} |")
        out += ["\n### by level and category\n", HEAD, table(rows, "all")]
        out += [table([r for r in rows if r["level"] == lv], lv) for lv in LEVELS if any(r["level"] == lv for r in rows)]
        cats = sorted({c for r in rows for c in r["cats"]})
        out += [table([r for r in rows if c in r["cats"]], f"category {c}") for c in cats]
        ks = [1, 2, 5, 10, 20, 40]
        out += ["\n### success within k executed tries\n", "| k | " + " | ".join(map(str, ks)) + " |", "|---" * (len(ks) + 1) + "|",
                "| share | " + " | ".join(f"{100 * np.mean([r['first_try'] is not None and r['first_try'] <= k for r in rows]):.0f}%" for k in ks) + " |"]
    text = "\n".join(x for x in out if x is not None)
    print(text)
    if A.md:
        Path(A.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
