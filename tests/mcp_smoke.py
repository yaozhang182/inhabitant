"""Smoke test of the MCP server with the real simulator and generator, without an LLM.

Starts `python -m inhabitant.mcp_server` for one task as a subprocess and drives it with a scripted MCP client over stdio:
list the tools (compared with the built-in definitions), get the prompt `inhabitant_task`, then get_state, one
generate_segment (a simple text, no constraints), one inspect `where` query and commit. Checks that the try comes back as
text plus an image and that log.json holds the calls.

  python tests/mcp_smoke.py [--task low_14] [--out <run dir>]
Needs a GPU, the installed environment and the `mcp` package; exit code 0 if every check passes."""
import argparse, asyncio, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


async def run(A):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from inhabitant.prompt import build_prompt
    system, tools = build_prompt(); ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok = ok and bool(cond); print(f"{'ok  ' if cond else 'FAIL'} {name} {info}", flush=True)
    env = dict(os.environ); env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    params = StdioServerParameters(command=sys.executable, args=["-m", "inhabitant.mcp_server", "--task", A.task, "--out", A.out], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tl = (await s.list_tools()).tools
            check("tool list equals the built-in definitions",
                  [dict(name=t.name, description=t.description, parameters=t.input_schema) for t in tl] == [t["function"] for t in tools])
            check("prompt listed", [p.name for p in (await s.list_prompts()).prompts] == ["inhabitant_task"])
            t0 = time.time(); brief = (await s.get_prompt("inhabitant_task")).messages[0].content.text
            check("prompt = system prompt + task + state", brief.startswith(system) and "# Task\nInstruction: " in brief and "# Current state" in brief,
                  f"({len(brief)} characters, world built in {time.time() - t0:.0f} s)")
            calls = [("get_state", {}),
                     ("generate_segment", {"intent": "take a step forward", "text": "A person walks forward.", "seconds": 2}),
                     ("inspect", {"queries": [{"fn": "where", "id": "pelvis"}, {"fn": "where", "id": "pelvis", "t0": 1.0}]}),
                     ("commit", {})]
            for name, args in calls:
                t0 = time.time(); res = await s.call_tool(name, args, read_timeout_seconds=1800)
                text = "\n".join(c.text for c in res.content if c.type == "text"); imgs = [c for c in res.content if c.type == "image"]
                print(f"--- {name} ({time.time() - t0:.0f} s, {len(imgs)} image): {text[:300]!r}", flush=True)
                if name == "get_state":
                    check("get_state", text.startswith("# Current state (last committed)") and "## Top-down map" in text)
                if name == "generate_segment":
                    check("try executed", text.startswith("Try 1 of segment 1 executed (not committed). Goal satisfied:"))
                    check("image content", len(imgs) == 1 and imgs[0].mime_type == "image/jpeg" and len(imgs[0].data) > 10000, f"({len(imgs[0].data) if imgs else 0} base64 characters)")
                if name == "inspect":
                    check("inspect where", text.startswith("[1] pelvis now: map (") and "[2] pelvis at t = 1.0 s" in text)
                if name == "commit":
                    check("commit", text.startswith("Committed segment 1."))
    log = Path(A.out) / "log.json"
    check("log.json written", log.exists())
    if log.exists():
        d = json.load(open(log))
        check("log holds the calls", [c["name"] for c in d["tool_log"]] == [c[0] for c in calls] and len(d["program"]) == 1 and len(d["tries"]) == 1,
              f"(llm_calls {d['llm_calls']}, success {d['success']})")
    check("brief.md written", (Path(A.out) / "brief.md").exists())
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="low_14"); ap.add_argument("--out", default=str(REPO / "runs/mcp_smoke/low_14"))
    A = ap.parse_args(); A.out = str(Path(A.out).resolve())
    ok = asyncio.run(run(A))
    print("MCP smoke test:", "PASSED" if ok else "FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
