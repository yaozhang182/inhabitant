"""Regression test without an LLM: replay the tool calls of a finished run and compare the replies.

Takes a run directory whose log.json has a `tool_log` (runs of the built-in loop or of the MCP server), builds the episode of
the same task again and feeds the logged tool calls (name and arguments, in order) through `Episode.call_tool`, the entry
point that the built-in loop and the MCP server use. Each new reply is compared with the logged one:
  identical   the same text
  tolerant    the same text once numbers are left out, and every number within the tolerance (0.05 m; 5 deg for angles)
  differs     anything else; the first differing line is printed
Execution in the simulator is not exactly repeatable (positions differ by millimetres to a few centimetres between identical
tries, and a later segment starts from them), so `tolerant` is the expected result for most calls after the first try.
The outcome of every try is compared too: goal satisfied, each grasp / release (SUCCESS / MISSED), what is held. At the end:
whether the episode ended with success at the same call. The replayed run is written like any run (log.json, tries/), so
scripts/evaluate.sh works on it.

  python tests/replay_log.py <old run dir> [--out <new run dir>] [--tol 0.05]
Needs a GPU and the installed environment; exit code 0 if no call differs and the end is the same."""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUM = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")


def numbers(line):
    """[(value, is_angle)] of a line and the line with its numbers left out."""
    out = []
    for m in NUM.finditer(line):
        before = line[max(0, m.start() - 24):m.start()]
        ang = bool(re.match(r"\s*(deg|°)", line[m.end():])) or bool(re.search(r"(facing|heading|theta|turned)\s*[-+(]*$", before))
        out.append((float(m.group()), ang))
    return out, NUM.sub("#", line)


def compare(old, new, tol, tol_deg):
    """-> (verdict, detail): identical / tolerant / differs."""
    if old == new:
        return "identical", ""
    lo, ln = old.split("\n"), new.split("\n"); worst = 0.0
    if len(lo) != len(ln):
        return "differs", f"{len(lo)} lines before, {len(ln)} now"
    for a, b in zip(lo, ln):
        if a == b:
            continue
        (na, sa), (nb, sb) = numbers(a), numbers(b)
        if sa != sb or len(na) != len(nb):
            return "differs", f"\n      - {a[:240]}\n      + {b[:240]}"
        for (x, ang), (y, _) in zip(na, nb):
            d = abs(x - y)
            if ang:
                d = min(d % 360, 360 - d % 360)
            if d > (tol_deg if ang else tol) + 1e-9:
                return "differs", f"number {x:g} -> {y:g}\n      - {a[:240]}\n      + {b[:240]}"
            worst = max(worst, 0.0 if ang else d)
    return "tolerant", f"max difference {worst:.3f}"


def outcome(t):
    """What a try did, from its log summary: goal, grasp / release results, what is held."""
    fb = t.get("feedback", "")
    ev = re.findall(r"^- (grasp|release) (\w+)(?: (\S+))?.*?: (SUCCESS|MISSED)", fb, re.M)
    held = next((l for l in fb.split("\n") if l.startswith("- Holding now:")), "")
    return dict(goal=bool(t.get("goal")), events=[(k, h, r) for k, h, _, r in ev], holding=held)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a run directory with a log.json that has a tool_log")
    ap.add_argument("--out", default="", help="directory of the replayed run (default <runs>/replay/<tag>/<set>_<id>)")
    ap.add_argument("--tol", type=float, default=0.05, help="tolerance for numbers (metres)")
    ap.add_argument("--tol_deg", type=float, default=5.0, help="tolerance for angles (degrees)")
    A = ap.parse_args()
    old = json.load(open(Path(A.run) / "log.json"))
    if not old.get("tool_log"):
        sys.exit("this log has no tool_log")
    from inhabitant.episode import Episode
    from inhabitant.paths import BENCHMARK
    a = argparse.Namespace(task=str(old["id"]), spec="", out=A.out, model="replay", tag=Path(A.run).resolve().parent.name, max_calls=10 ** 6)
    if not (BENCHMARK / f"{old['id']}.json").exists():            # a task file of the user's own: it is saved in the log
        if not old.get("spec"):
            sys.exit(f"no task {old['id']} in the benchmark and no task saved in the log")
        tmp = Path(A.out or ".") / f"{old['id']}.json"; tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(old["spec"])); a.spec = str(tmp)
    ep = Episode(a)
    first = ep.task_message() + "\n\n" + f"# Current state\n{ep.state_text()}"
    ep.msgs = [{"role": "user", "content": first}]
    count = dict(identical=0, tolerant=0, differs=0); first_diff = None; end_new = None
    old_first = old["messages"][0]["content"] if old.get("messages") else None
    if isinstance(old_first, str) and "{STATE}" not in old_first:
        v, d = compare(old_first, first, A.tol, A.tol_deg); print(f"first message (task + initial state): {v} {d}", flush=True)
    n_try = len(ep.tries)
    for i, c in enumerate(old["tool_log"], 1):
        t0 = time.time()
        try:
            content, state = ep.call_tool(c["name"], json.loads(json.dumps(c["args"])))
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            text = "\n".join(x["text"] for x in parts if x["type"] == "text").replace("{STATE}", state or "")
        except Exception as e:                                    # as the agent loop: the error text is the reply
            text = f"Tool error: {type(e).__name__}: {e}"
        ep.calls += 1
        ep.msgs += [{"role": "assistant", "tool_call": {"name": c["name"], "args": c["args"]}}, {"role": "tool", "content": text}]
        ep.tool_log.append(dict(name=c["name"], args=c["args"], reply=text, t_think=0.0, t_run=time.time() - t0))
        v, d = compare(c["reply"], text, A.tol, A.tol_deg); count[v] += 1
        extra = ""
        if len(ep.tries) > n_try:                                 # a try was executed: compare what it did
            n_try = len(ep.tries); new_t = ep.tries[-1]
            old_t = old["tries"][n_try - 1] if n_try <= len(old["tries"]) else None
            same = old_t is not None and outcome(old_t) == outcome(new_t)
            extra = f" | try {n_try}: outcome {'same' if same else 'DIFFERENT'} ({new_t['summary']})"
            if not same:
                extra += f"\n      before: {outcome(old_t) if old_t else None}\n      now:    {outcome(new_t)}"
                if v != "differs":
                    count[v] -= 1; v = "differs"; count[v] += 1
        if v == "differs" and first_diff is None:
            first_diff = i
        print(f"[{i:3d}/{len(old['tool_log'])}] {c['name']:16s} {v:9s} {d}{extra}", flush=True)
        if ep.done:
            end_new = i; break
    best = ep.versions[0] if ep.versions else None
    ep.save_log(best)
    end_old = len(old["tool_log"]) if old.get("success") else None
    same_end = end_new == end_old and bool(ep.versions) == bool(old.get("success"))
    print(f"\n{old['id']}: {count['identical']} identical, {count['tolerant']} tolerant, {count['differs']} differ "
          f"(first difference at call {first_diff}); before: success={bool(old.get('success'))} at call {end_old}; "
          f"now: success={bool(ep.versions)} at call {end_new}; same end: {'YES' if same_end else 'NO'}; run: {ep.out}", flush=True)
    sys.exit(0 if same_end and not count["differs"] else 1)


if __name__ == "__main__":
    main()
