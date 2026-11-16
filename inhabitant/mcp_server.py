"""The five tools of an episode as an MCP server (stdio), so that any MCP client can be the agent.

One server process = one episode. The server exposes exactly the tools of the built-in loop (generate_segment, inspect, commit,
get_state, retry_from) with the same descriptions, argument schemas and result texts, by calling the same `Episode` methods;
the eight frames of a try come back as an image next to the text. The prompt `inhabitant_task` holds what the built-in loop
sends before the first tool call: the system prompt, the task instruction and the initial state (also written to
<out>/brief.md). The episode ends when a committed segment completes the task or after --max_calls tool calls; log.json is
written as by the built-in loop (inhabitant/episode.py), so the evaluation scripts read both alike.

  python -m inhabitant.mcp_server --task low_01 --out runs/mcp/low_01
  python -m inhabitant.mcp_server --spec my_task.json --out runs/mcp/my_task

Written against the low-level server of the official `mcp` Python SDK, version 2 (the tool schemas are passed through as they
are, which the function-signature based MCPServer / FastMCP layer cannot do). The simulator and the generator write to stdout,
so the protocol keeps the original stdout and everything else goes to stderr. All work on the episode runs in one worker
thread (the simulator's GL context belongs to the thread that created it)."""
import argparse, asyncio, base64, io, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

from .prompt import build_prompt

PROMPT = "inhabitant_task"


class Session:
    """One episode behind the server; every method runs in the worker thread."""
    def __init__(self, a):
        self.a, self.ep, self.over, self.last = a, None, None, time.time()

    def build(self):
        from .episode import Episode            # imports the simulator
        ep = Episode(self.a)
        self.first = ep.task_message() + "\n\n" + f"# Current state\n{ep.state_text()}"
        self.brief = ep.system + "\n\n" + "=" * 80 + "\n\n" + self.first
        (ep.out / "brief.md").write_text(self.brief)
        ep.msgs = [{"role": "user", "content": self.first}]
        self.ep = ep; self.last = time.time()
        print(f"ready: {ep.out}", file=sys.stderr, flush=True)
        return self.brief

    def call(self, name, args):
        """One tool call -> (text, path of the try's image or None)."""
        ep = self.ep
        if self.over:
            return "The episode is over: " + self.over, None
        t0 = time.time(); text, img = "", None
        try:
            content, state = ep.call_tool(name, args)
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            text = "\n".join(x["text"] for x in parts if x["type"] == "text").replace("{STATE}", state or "")
            img = next((x["path"] for x in parts if x["type"] == "image"), None)        # the try's 8-frame sheet
            ep.msgs += [{"role": "assistant", "tool_call": {"name": name, "args": args}}, {"role": "tool", "content": text}]
        except Exception as e:                  # a bad input must not end the episode: the agent gets the error
            text = f"Tool error: {type(e).__name__}: {e}"; traceback.print_exc()
        ep.calls += 1
        ep.tool_log.append(dict(name=name, args=args, reply=text, t_think=t0 - self.last, t_run=time.time() - t0))
        if ep.done or ep.calls >= self.a.max_calls:
            best = ep.versions[0] if ep.versions else None          # the program that reached the goal (the episode ends there)
            ep.save_log(best); self.over = json.dumps(dict(success=bool(ep.versions), calls=ep.calls))
            print(f"\nsuccess={bool(ep.versions)} segments={len(best) if best else 0} calls={ep.calls} tries={len(ep.tries)}",
                  file=sys.stderr, flush=True)
        else:
            ep.save_log()
        self.last = time.time()
        return text, img


def make_server(session, worker, built):
    """The MCP server of a session. `built`: the future of session.build() in the worker thread."""
    import mcp.types as types
    from mcp.server.lowlevel import Server
    _, tools = build_prompt()
    run = lambda fn, *args: asyncio.get_running_loop().run_in_executor(worker, fn, *args)

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name=t["function"]["name"], description=t["function"]["description"],
                                                       input_schema=t["function"]["parameters"]) for t in tools])

    async def call_tool(ctx, params):
        await asyncio.wrap_future(built)
        text, img = await run(session.call, params.name, dict(params.arguments or {}))
        content = [types.TextContent(type="text", text=text)]
        if img:
            data = base64.b64encode(open(img, "rb").read()).decode()
            content.append(types.ImageContent(type="image", data=data, mime_type="image/jpeg"))
        return types.CallToolResult(content=content)

    async def list_prompts(ctx, params):
        return types.ListPromptsResult(prompts=[types.Prompt(name=PROMPT, description="How to act (the system prompt of the built-in "
                                                             "agent), the task instruction and the initial state of this episode.")])

    async def get_prompt(ctx, params):
        if params.name != PROMPT:
            raise ValueError(f"unknown prompt {params.name!r}")
        brief = await asyncio.wrap_future(built)
        return types.GetPromptResult(description="Inhabitant task", messages=[
            types.PromptMessage(role="user", content=types.TextContent(type="text", text=brief))])

    return Server("inhabitant", version="1.0.0",
                  instructions=f"One household task carried out with a human body. Read the prompt '{PROMPT}' first: it explains how to "
                               f"move, the tools and the world, and gives the task and the initial state.",
                  on_list_tools=list_tools, on_call_tool=call_tool, on_list_prompts=list_prompts, on_get_prompt=get_prompt)


async def serve(session, wire_in, wire_out):
    import anyio
    from mcp.server.stdio import stdio_server
    worker = ThreadPoolExecutor(max_workers=1)
    built = worker.submit(session.build)        # the world is built while the client connects
    server = make_server(session, worker, built)
    stdin = anyio.wrap_file(io.TextIOWrapper(wire_in, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(io.TextIOWrapper(wire_out, encoding="utf-8"))
    async with stdio_server(stdin=stdin, stdout=stdout) as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="low_01", help="a task id of benchmark/inhabit (low_01 ... body_10)")
    ap.add_argument("--spec", default="", help="a task file (format of benchmark/inhabit/*.json) instead of --task")
    ap.add_argument("--out", default="", help="run directory (default <runs>/<model>/<tag>/<set>_<id>)")
    ap.add_argument("--model", default="mcp-agent", help="name of the agent, for the log and the default run directory")
    ap.add_argument("--tag", default="mcp", help="name of this set of runs")
    ap.add_argument("--max_calls", type=int, default=150, help="budget of tool calls")
    a = ap.parse_args()
    # the protocol keeps the process's stdin / stdout; everything else that prints (simulator, generator, logs) goes to stderr
    wire_in, wire_out = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY); os.dup2(devnull, 0); os.close(devnull)
    os.dup2(2, 1); sys.stdout = sys.stderr
    asyncio.run(serve(Session(a), wire_in, wire_out))


if __name__ == "__main__":
    main()
