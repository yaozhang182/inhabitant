"""Example MCP client: a language model carries out one task through the Inhabitant MCP server.

The client starts the server as a stdio subprocess with the standard `mcp` client SDK, reads everything the model needs
from it (the tool list and the prompt `inhabitant_task`: how to act, the task, the initial state), and runs a plain agent
loop: model call -> its tool calls are forwarded with `call_tool` -> the returned text and image go back into the
conversation. It stops when the server says the episode is over, when the model stops calling tools, or at the budget.

  python examples/mcp_client.py --task low_01 --model claude-opus-5.5
  python examples/mcp_client.py --task low_01 --dry_run          # a scripted "model": checks the install without an API key

The server side (`python -m inhabitant.mcp_server`) needs the GPU machine with the simulator and Kimodo installed (README,
"Install"). This file is only an example: any MCP-capable client or agent framework can be used instead. To plug in your own
model, replace `cl.call(...)` below; the conversation is kept in OpenAI chat format (`inhabitant/llm.py` converts it for
Anthropic, OpenAI and Gemini; models and keys: configs/models.json, ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY)."""
import argparse, asyncio, base64, json, os, shlex, sys, time
from pathlib import Path
from types import SimpleNamespace

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SEP = "\n\n" + "=" * 80 + "\n\n"            # in the prompt, between the instructions (system prompt) and the task + state
KEEP_IMAGES = 3                             # models other than Claude: only the images of the last tries are sent again


class ScriptedModel:
    """--dry_run: a fixed list of tool calls instead of a model (same interface as inhabitant.llm.Client)."""
    script = [("get_state", {}), ("generate_segment", {"intent": "take a step forward", "text": "A person walks forward.", "seconds": 2}),
              ("inspect", {"queries": [{"fn": "where", "id": "pelvis"}]}), ("commit", {})]
    models = {"scripted": {"provider": "scripted"}}

    def call(self, req):
        n = sum(m["role"] == "assistant" for m in req.history)
        calls = [{"id": f"call_{n}", "type": "function", "function": {"name": self.script[n][0], "arguments": json.dumps(self.script[n][1])}}] \
            if n < len(self.script) else []
        return SimpleNamespace(text="" if calls else "(end of the script)", thoughts="", tool_calls=calls, raw_content=[], error="",
                               finish_reason="tool_calls" if calls else "stop", input_tokens=0, output_tokens=0)


def to_request(msgs, provider):
    """The conversation as it is sent: for models other than Claude, images older than the last KEEP_IMAGES are left out
    (Claude's thinking blocks need the earlier messages unchanged)."""
    has_image = [i for i, m in enumerate(msgs) if isinstance(m["content"], list)]
    old = set() if provider == "anthropic" else set(has_image[:-KEEP_IMAGES])
    return [dict(m, content=[p for p in m["content"] if p["type"] == "text"]) if i in old else m for i, m in enumerate(msgs)]


async def run(a):
    from inhabitant.llm import Client, Request
    out = Path(a.out or REPO / "runs/mcp_example" / a.task).resolve(); (out / "client_images").mkdir(parents=True, exist_ok=True)
    cl, model = (ScriptedModel(), "scripted") if a.dry_run else (Client(), a.model)
    provider = cl.models[model]["provider"]
    cmd = shlex.split(a.server) + ["--task", a.task, "--out", str(out), "--max_calls", str(a.max_calls)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    async with stdio_client(StdioServerParameters(command=cmd[0], args=cmd[1:], env=env)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 1. what the model is given: the tools (MCP schema -> function-calling format) and the prompt
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                     for t in (await session.list_tools()).tools]
            prompt = (await session.get_prompt("inhabitant_task")).messages[0].content.text       # waits until the world is built
            system, first = prompt.rsplit(SEP, 1)
            print(f"server ready; tools: {', '.join(t['function']['name'] for t in tools)}; model: {model}", flush=True)
            msgs = [{"role": "user", "content": first}]; tokens = 0; end = "budget of model calls"
            # 2. the loop
            for n in range(1, a.max_calls + 1):
                t0 = time.time()
                res = await asyncio.to_thread(cl.call, Request(model, prompt=None, system=system, history=to_request(msgs, provider),
                                                               params={"tools": tools}, tag=f"mcp_example/{a.task}/c{n}"))
                if res.error:
                    end = f"API error: {res.error[:200]}"; break
                tokens = res.input_tokens + res.output_tokens              # size of the conversation at the last call
                msg = {"role": "assistant", "content": res.text or "", "tool_calls": res.tool_calls}
                if res.raw_content:                                         # what the provider wants back unchanged (thinking blocks, ...)
                    msg["anthropic_content" if provider == "anthropic" else "raw_output"] = res.raw_content
                if not res.tool_calls:
                    del msg["tool_calls"]; msgs.append(msg); end = "the model stopped"; break
                msgs.append(msg); over = False
                for tc in res.tool_calls:                                   # 3. forward each tool call to the server
                    name = tc["function"]["name"]; t1 = time.time()
                    try:
                        result = await session.call_tool(name, json.loads(tc["function"]["arguments"] or "{}"), read_timeout_seconds=3600)
                        text = "\n".join(c.text for c in result.content if c.type == "text"); content = text
                        for c in result.content:
                            if c.type == "image":                           # the eight frames of a try
                                f = out / "client_images" / f"call{n:03d}.jpg"; f.write_bytes(base64.b64decode(c.data))
                                content = [{"type": "text", "text": text}, {"type": "image", "path": str(f)}]
                    except json.JSONDecodeError as e:
                        content = text = f"The arguments are not valid JSON ({e})."
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                    print(f"call {n:3d} {name:16s} model {t1 - t0:4.0f} s, tool {time.time() - t1:4.0f} s | {text[:100]!r}", flush=True)
                    over = "the episode ends" in text or text.startswith("The episode is over")
                    if over:
                        break
                if over:
                    end = "the server ended the episode"; break
    log = json.load(open(out / "log.json")) if (out / "log.json").exists() else {}
    print(f"\nsuccess: {'yes' if log.get('success') else 'no'} | end: {end} | tool calls: {log.get('llm_calls')} | tries: {len(log.get('tries', []))} | "
          f"committed segments: {len(log.get('program', []))} | conversation: {tokens} tokens | run: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="low_01", help="a task id of benchmark/inhabit")
    ap.add_argument("--model", default="claude-opus-5.5", help="a model alias of configs/models.json")
    ap.add_argument("--server", default=f"{shlex.quote(sys.executable)} -m inhabitant.mcp_server", help="command that starts the MCP server")
    ap.add_argument("--out", default="", help="run directory (default runs/mcp_example/<task>)")
    ap.add_argument("--max_calls", type=int, default=150, help="budget of model calls (and of tool calls on the server)")
    ap.add_argument("--dry_run", action="store_true", help="a scripted model instead of an API: four tool calls, no key needed")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
