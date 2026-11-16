"""The built-in agent loop: an LLM drives one episode through its five tools.

One episode = one conversation. The model gets the system prompt (inhabitant/prompt.py), the task instruction and the full
state, then calls tools until a committed segment completes the task, it answers without a tool call, or the budget of LLM
calls (--max_calls) is used up. The harness only runs the tools and keeps the logs; the goal itself is hidden, but every try
says whether it is satisfied. The conversation only grows (every state, image and reasoning block stays, which prompt caching
and reasoning blocks need); each try returns the eight frames of its video.

  python -m inhabitant.agent --task low_01 --model claude-opus-5.5 --tag main
  python -m inhabitant.agent --spec my_task.json --model gpt-6-luna
  python -m inhabitant.agent --print_prompt        # the system prompt and the tool definitions

Outputs <runs>/<model>/<tag>/<set>_<id>/ (see inhabitant/episode.py). Models are configured in configs/models.json."""
import argparse, copy, json, re, time

from .episode import Episode, _fill
from .llm import Client, Request


class Agent(Episode):
    def __init__(self, a):
        self.cl = Client()                           # before the world is built: a missing model configuration fails at once
        if a.model not in self.cl.models:
            raise SystemExit(f"unknown model {a.model!r}; configured: {', '.join(self.cl.models)}")
        self.provider = self.cl.models[a.model].get("provider", "")
        super().__init__(a)

    KEEP_IMAGES, IMAGE_BATCH = 3, 3     # non-Anthropic models: the sheets of the last 3 tries stay; text, reasoning, states all stay

    def render(self):
        """Stored messages -> request messages: everything as it was sent, each state message with its own state (the conversation
        only grows)."""
        img = [i for i, m in enumerate(self.msgs) if isinstance(m.get("content"), list) and any(p["type"] == "image" for p in m["content"])]
        # Anthropic models: nothing is changed (a thinking block is bound to everything before it: editing an earlier message fails
        # the API's history check); other models: older sheets are dropped 3 at a time
        cut = set() if self.provider == "anthropic" else set(img[:max(0, (len(img) - self.KEEP_IMAGES) // self.IMAGE_BATCH * self.IMAGE_BATCH)])
        blocked = getattr(self, "_blocked", set())    # sheets an API's image filter rejected
        out = []
        for i, m in enumerate(self.msgs):
            r = {k: v for k, v in m.items() if not k.startswith("_")}
            if m.get("_state"):
                r["content"] = _fill(m["content"], m["_state"])
            if i in blocked and i not in cut:
                r["content"] = [dict(p, text=p["text"].replace("8 frames of the video of this try are attached", "8 frames of the video of this "
                                                               "try could not be shown (the API rejected the image)")) for p in m["content"] if p["type"] == "text"]
            elif i in cut:
                r["content"] = [dict(p, text=p["text"].replace("8 frames of the video of this try are attached", "8 frames of the video of this "
                                                               "try were attached (no longer shown)")) for p in m["content"] if p["type"] == "text"]
            out.append(r)
        return out

    def step(self):
        tools = self.tools          # one tool list for the whole conversation (for Anthropic models a changed list would break the
                                    # cache and Claude's thinking blocks: 400 "bound to a different conversation")
        t0 = time.time()
        for attempt in range(4):                          # a dropped connection must not end the episode
            try:
                res = self.cl.call(Request(self.a.model, prompt=None, system=self.system, history=copy.deepcopy(self.render()),
                                           params={"tools": tools}, tag=f"agent/{self.a.tag}/{self.a.eval_set}_{self.a.id}/c{self.calls + 1}"))
                break
            except Exception as e:                        # network / transport errors that the client did not retry
                print(f"LLM call failed ({type(e).__name__}: {str(e)[:200]}); retry {attempt + 1} in 60 s", flush=True)
                if attempt == 3:
                    self.done = True; self.end_reason = f"API error: {type(e).__name__}: {str(e)[:200]}"; return
                time.sleep(60)
        self.calls += 1; t_llm = time.time() - t0
        for k in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens", "cost_usd"):
            self.usage[k] += getattr(res, k) or 0
        if res.input_tokens:                               # size of the conversation at the last call (input + its answer)
            self.usage["context_tokens"] = res.input_tokens + res.output_tokens
        print(f"LLM call {self.calls} ({t_llm:.0f} s, in {res.input_tokens} (cache read {res.cached_tokens}, write {res.cache_write_tokens}), "
              f"out {res.output_tokens}, ${res.cost_usd or 0:.3f}, total ${self.usage['cost_usd']:.2f}): "
              f"{[tc['function']['name'] for tc in res.tool_calls] or (res.error[:200] or res.finish_reason)}", flush=True)
        limit = 20 if re.match(r"HTTP 429", res.error or "") else 5   # 429 = a token rate limit; 5xx = an overloaded server
        if res.error and re.match(r"HTTP (5\d\d|429)", res.error) and getattr(self, "_server_retries", 0) < limit:
            self._server_retries = getattr(self, "_server_retries", 0) + 1
            print(f"server error ({res.error[:120]}); waiting 120 s, retry {self._server_retries} of {limit}", flush=True)
            self.calls -= 1; time.sleep(120); return     # the same request again at the next step; not counted as a call
        if res.error and "content_policy_violation" in res.error and len(getattr(self, "_blocked", set())) < 5:
            # a provider's content filter rejected an image of the request (the sheets only show the body in the apartment): the
            # newest sheet still shown is replaced by a line saying it could not be shown, and the request is sent again (not
            # counted as a call); at most 5 sheets per episode.
            self._blocked = getattr(self, "_blocked", set())
            shown = [i for i, m in enumerate(self.render()) if isinstance(m.get("content"), list) and any(p["type"] == "image" for p in m["content"])]
            if shown:
                self._blocked.add(shown[-1]); self.calls -= 1
                print(f"image filter: the request was rejected ({res.error[:120]}); the sheet of message {shown[-1]} is replaced by a note "
                      f"({len(self._blocked)} so far)", flush=True)
                return
        if res.error:                                    # the client already retried transient errors; a 4xx will not go away
            self.done = True; self.end_reason = f"API error: {res.error[:300]}"; print(f"episode ends: {self.end_reason}", flush=True)
            return
        self._server_retries = 0
        msg = {"role": "assistant", "content": res.text or ""}
        if res.thoughts:
            msg["reasoning_content"] = res.thoughts
        if res.raw_content:                              # sent back unchanged: Anthropic thinking blocks with signatures; Responses API
            msg["anthropic_content" if self.provider == "anthropic" else "raw_output"] = res.raw_content   # / Gemini output items
        if res.tool_calls:
            msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                                 for tc in res.tool_calls]
        self.msgs.append(msg)
        if not res.tool_calls:                           # the model stopped: the episode ends
            self.done = True; self.end_reason = f"no tool call ({res.finish_reason})"; print(f"episode ends: {self.end_reason}", flush=True)
            return
        for tc in res.tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                content, state = f"The arguments are not valid JSON ({e}).", None
            else:
                t1 = time.time()
                try:
                    content, state = self.call_tool(tc["function"]["name"], args)
                except Exception as e:                   # a bad input must not end the episode: the model gets the error
                    import traceback; traceback.print_exc()  # (e.g. seconds sent as the string "[4, 2]")
                    content, state = f"Tool error: {type(e).__name__}: {e}", None
                txt = content if isinstance(content, str) else "\n".join(p["text"] for p in content if p["type"] == "text")
                self.tool_log.append(dict(name=tc["function"]["name"], args=args, reply=txt.replace("{STATE}", state or ""),
                                          t_think=t_llm, t_run=time.time() - t1)); t_llm = 0.0
            m = {"role": "tool", "tool_call_id": tc["id"], "content": content}
            if state:
                m["_state"] = state
            self.msgs.append(m)
            if self.done:
                break

    def run(self):
        self.msgs = [{"role": "user", "content": self.task_message() + "\n\n# Current state\n{STATE}", "_state": self.state_text()}]
        while not self.done and self.calls < self.a.max_calls:
            self.step(); self.save_log()
        best = self.versions[0] if self.versions else None
        print(f"\nsuccess={bool(self.versions)} segments={len(best) if best else 0} calls={self.calls} tries={len(self.tries)} end={self.end_reason}", flush=True)
        self.save_log(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="low_01", help="a task id of benchmark/inhabit (low_01 ... body_10)")
    ap.add_argument("--spec", default="", help="a task file (format of benchmark/inhabit/*.json) instead of --task; the run is "
                    "<tag>/spec_<its id>")
    ap.add_argument("--model", default="claude-opus-5.5", help="a model alias of configs/models.json")
    ap.add_argument("--tag", default="main", help="name of this set of runs")
    ap.add_argument("--out", default="", help="run directory (default <runs>/<model>/<tag>/<set>_<id>)")
    ap.add_argument("--max_calls", type=int, default=150, help="budget of LLM calls")
    ap.add_argument("--print_prompt", action="store_true", help="print the system prompt and the tool definitions and exit")
    a = ap.parse_args()
    if a.print_prompt:
        from .prompt import prompt_text
        print(prompt_text()); return
    del a.print_prompt
    Agent(a).run()


if __name__ == "__main__":
    main()
