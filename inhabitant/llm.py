"""LLM client of the agent loop: one interface over three kinds of API, all with tool calling and image input.

  provider "anthropic"          Anthropic Messages API, through the official `anthropic` SDK (imported on first use)
  provider "openai_responses"   OpenAI Responses API (plain HTTP)
  provider "openai_chat"        any OpenAI-compatible chat completions server, e.g. vLLM for open models (plain HTTP)
  provider "gemini"             Google Gemini generateContent with function calling (plain HTTP)

Models are aliases in configs/models.json (copy configs/models.example.json): provider, api_model, optional base_url, default
params (provider-native fields) and prices. The conversation is passed in OpenAI chat format (Request.history) and converted
for each provider; what a provider needs back unchanged (Anthropic thinking blocks, Responses / Gemini output items) is returned
in Result.raw_content and replayed by the agent.

  - Keys: the environment variable of the provider (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY; `key_env` in the model's
    entry names another one), or `key_file` in the model's entry: the path of a file that holds the key. Never logged or printed.
  - Cache: every successful response is stored under <repo>/cache/llm/ keyed by sha256 of (provider, api_model, base_url, system,
    conversation, params, sample_idx). Transport errors are not cached; a completed-but-bad response (max tokens hit, safety
    block) is cached, because it is the model's answer under those params.
  - Log: one line per call (cached or not) in <repo>/logs/llm_calls.jsonl: date, alias, api model, reported model version,
    tokens, cost, latency, finish reason. `python -m inhabitant.llm --spend` sums it.
  - Budget: Client(budget_usd=...) refuses new (uncached) calls once this process has spent that much (models need a price).

    from inhabitant.llm import Client, Request
    res = Client().call(Request("claude-opus-5.5", prompt=None, system=system, history=messages, params={"tools": tools}))
"""
import base64, hashlib, json, os, random, re, sys, threading, time, datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path
import httpx

from .paths import REPO, MODELS_JSON

CACHE_DIR = REPO / "cache/llm"
CALL_LOG = REPO / "logs/llm_calls.jsonl"
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai_responses": "OPENAI_API_KEY", "openai_chat": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_TRIES = 6


@dataclass
class Request:
    model: str                      # alias in configs/models.json
    prompt: str | list | None       # text, or content parts with local media (see _content); None: the conversation is all in
                                    # `history` (agent loops ending with a tool message)
    system: str = ""
    params: dict = field(default_factory=dict)   # merged over the alias's default params (provider-native fields)
    sample_idx: int = 0             # distinct cache entries for repeated samples of the same prompt
    tag: str = ""                   # free text for the log
    history: list = field(default_factory=list)  # earlier messages, OpenAI chat format: user / assistant
                                                 # (+ tool_calls, reasoning_content) / tool (+ tool_call_id); content str | parts.
                                                 # Tools are declared in params["tools"].


@dataclass
class Result:
    text: str = ""
    thoughts: str = ""              # only if the provider returns them (Gemini includeThoughts, vLLM reasoning_content)
    finish_reason: str = ""         # normalised: "stop", "max_tokens", "safety", "other:<raw>"
    model_version: str = ""         # as reported by the API
    input_tokens: int = 0
    cached_tokens: int = 0          # part of input_tokens served from the provider's prompt cache
    cache_write_tokens: int = 0     # part of input_tokens written to the prompt cache (Anthropic: billed at price cache_write)
    output_tokens: int = 0          # visible answer tokens
    thinking_tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    cached: bool = False
    key: str = ""
    error: str = ""                 # transport / HTTP error after retries (never cached)
    tool_calls: list = field(default_factory=list)   # [{"id", "type": "function", "function": {"name", "arguments"}}]
    raw_content: list = field(default_factory=list)  # the assistant turn as the provider returned it (Anthropic content blocks
                                                     # with thinking signatures; Responses / Gemini output items): pass it back
                                                     # unchanged as "anthropic_content" / "raw_output" of the assistant message

    @property
    def ok(self):
        return not self.error and self.finish_reason in ("stop", "tool_calls")


class BudgetExceeded(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """Daily quota used up: retrying is pointless until it resets."""


def load_models(path=MODELS_JSON):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: copy configs/models.example.json to configs/models.json (or set INHABITANT_MODELS)")
    return {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}


def _api_key(provider, cfg):
    env = cfg.get("key_env") or KEY_ENV.get(provider)
    if env and os.environ.get(env):
        return os.environ[env]
    if cfg.get("key_file"):
        f = Path(os.path.expandvars(cfg["key_file"])).expanduser()
        if f.exists():
            if f.stat().st_mode & 0o077:
                print(f"[llm] warning: {f} is readable by group/others; chmod 600 it", file=sys.stderr)
            return f.read_text().strip()
    if provider == "openai_chat":                  # a local server needs no key
        return "EMPTY"
    raise RuntimeError(f"no API key for {provider}: set ${env} or give key_file in the model's entry")


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---------------------------------------------------------------------------------------------------------------- providers
def _gemini_parse(j):
    c = (j.get("candidates") or [{}])[0]
    parts = (c.get("content") or {}).get("parts") or []
    u = j.get("usageMetadata") or {}
    fr = c.get("finishReason") or ("SAFETY" if (j.get("promptFeedback") or {}).get("blockReason") else "")
    return dict(text="".join(p.get("text", "") for p in parts if not p.get("thought")),
                thoughts="".join(p.get("text", "") for p in parts if p.get("thought")),
                finish_reason={"STOP": "stop", "MAX_TOKENS": "max_tokens", "SAFETY": "safety"}.get(fr, f"other:{fr}"),
                model_version=j.get("modelVersion", ""), input_tokens=u.get("promptTokenCount", 0),
                cached_tokens=u.get("cachedContentTokenCount", 0), output_tokens=u.get("candidatesTokenCount", 0), thinking_tokens=u.get("thoughtsTokenCount", 0))


def _openai_parse(j):
    text = "".join(c.get("text", "") for o in j.get("output", []) if o.get("type") == "message"
                   for c in o.get("content", []) if c.get("type") == "output_text")
    u = j.get("usage") or {}
    reas = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    st, why = j.get("status", ""), (j.get("incomplete_details") or {}).get("reason", "")
    fr = "stop" if st == "completed" else {"max_output_tokens": "max_tokens", "content_filter": "safety"}.get(why, f"other:{st}/{why}")
    return dict(text=text, finish_reason=fr, model_version=j.get("model", ""), input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0) - reas, thinking_tokens=reas)


MEDIA = {"video": ("video_url", "video/mp4"), "image": ("image_url", "image/png")}


def _mime(p):
    """MIME type of a media part from its file extension"""
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "mp4": "video/mp4"}.get(str(p["path"]).rsplit(".", 1)[-1].lower(), MEDIA[p["type"]][1])


def _content(c):
    """Message content: a string, or a list of parts {"type": "text", "text"} | {"type": "video" | "image", "path"} (local files,
    sent inline as data URLs)."""
    if c is None or isinstance(c, str):
        return c
    out = []
    for p in c:
        if p["type"] in MEDIA:
            kind, mime = MEDIA[p["type"]][0], _mime(p); b = base64.b64encode(open(p["path"], "rb").read()).decode()
            out.append({"type": kind, kind: {"url": f"data:{mime};base64,{b}"}})
        else:
            out.append(p)
    return out


def _keyable(c):
    """Content for the cache key: media parts by the sha256 of their file, not by path."""
    if c is None or isinstance(c, str):
        return c
    return [{"type": p["type"], "sha256": hashlib.sha256(open(p["path"], "rb").read()).hexdigest()} if p["type"] in MEDIA else p for p in c]


def _chat(cfg, req, params, key):
    msgs = ([{"role": "system", "content": req.system}] if req.system else []) + \
        [dict(m, content=_content(m.get("content"))) for m in req.history] + \
        ([{"role": "user", "content": _content(req.prompt)}] if req.prompt is not None else [])
    body = _merge({"model": cfg["api_model"], "messages": msgs}, params)
    return f"{os.path.expandvars(cfg['base_url'])}/chat/completions", {"Authorization": f"Bearer {key}"}, body


def _chat_parse(j):
    c = (j.get("choices") or [{}])[0]; m = c.get("message") or {}; u = j.get("usage") or {}
    fr = c.get("finish_reason", "")
    reas = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return dict(text=m.get("content") or "", thoughts=m.get("reasoning_content") or m.get("reasoning") or "",
                tool_calls=m.get("tool_calls") or [],
                finish_reason={"stop": "stop", "length": "max_tokens", "content_filter": "safety", "tool_calls": "tool_calls"}.get(fr, f"other:{fr}"),
                model_version=j.get("model", ""), input_tokens=u.get("prompt_tokens", 0),
                output_tokens=u.get("completion_tokens", 0) - reas, thinking_tokens=reas)


def _resp_parts(c, role="user"):
    """OpenAI-chat-style content -> Responses API input parts (text and inline images)."""
    if c is None or isinstance(c, str):
        return [{"type": "input_text" if role != "assistant" else "output_text", "text": c or "(empty)"}]
    out = []
    for p in c:
        if p["type"] == "image":
            b = base64.b64encode(open(p["path"], "rb").read()).decode()
            out.append({"type": "input_image", "image_url": f"data:{_mime(p)};base64,{b}"})
        elif p["type"] == "video":
            raise ValueError("openai_responses: video input is not supported")
        else:
            out.append({"type": "input_text", "text": p["text"]})
    return out


def _responses(cfg, req, params, key):
    """Multi-turn OpenAI Responses API. req.history is in OpenAI
    chat format: an assistant turn with "raw_output" (the output items of that response, reasoning with its encrypted content) is
    replayed unchanged, so the conversation only grows; a tool result becomes a function_call_output, and its images follow as a
    user message (function_call_output takes text)."""
    items = []
    for m in req.history + ([{"role": "user", "content": req.prompt}] if req.prompt is not None else []):
        if m["role"] in ("user", "system"):
            items.append({"role": "user" if m["role"] == "user" else "developer", "content": _resp_parts(m.get("content"))})
        elif m["role"] == "tool":
            c = m.get("content"); txt = c if isinstance(c, str) or c is None else "\n".join(p["text"] for p in c if p["type"] == "text")
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": txt or "(empty)"})
            if not isinstance(c, str) and c and any(p["type"] == "image" for p in c):
                items.append({"role": "user", "content": _resp_parts([p for p in c if p["type"] == "image"])})
        elif m["role"] == "assistant":
            if m.get("raw_output"):
                items += m["raw_output"]
            else:
                if m.get("content"):
                    items.append({"role": "assistant", "content": _resp_parts(m["content"], "assistant")})
                items += [{"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"], "arguments": tc["function"]["arguments"] or "{}"}
                          for tc in m.get("tool_calls") or []]
    params = dict(params); tools = params.pop("tools", None)
    body = _merge({"model": cfg["api_model"], "input": items, "store": False, "include": ["reasoning.encrypted_content"]}, params)
    if req.system:
        body["instructions"] = req.system
    if tools:
        body["tools"] = [{"type": "function", **{k: v for k, v in (t["function"] if t.get("type") == "function" else t).items() if k in ("name", "description", "parameters")}}
                         for t in tools]
    url = os.path.expandvars(cfg.get("base_url", "https://api.openai.com/v1/responses"))
    return url, {cfg.get("auth_header", "Authorization"): (key if cfg.get("auth_header") else f"Bearer {key}")}, body


def _responses_parse(j):
    r = _openai_parse(j); out = j.get("output") or []
    u = j.get("usage") or {}
    r["cached_tokens"] = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    r["tool_calls"] = [{"id": o["call_id"], "type": "function", "function": {"name": o["name"], "arguments": o.get("arguments") or "{}"}}
                       for o in out if o.get("type") == "function_call"]
    r["raw_content"] = out                     # stored by the agent as "raw_output" and replayed unchanged
    if r["tool_calls"] and r["finish_reason"] == "stop":
        r["finish_reason"] = "tool_calls"
    return r


def _gem_parts(c):
    """OpenAI-style content -> Gemini parts (images inline)."""
    if c is None or isinstance(c, str):
        return [{"text": c or "(empty)"}]
    out = []
    for p in c:
        if p["type"] == "image":
            out.append({"inlineData": {"mimeType": _mime(p), "data": base64.b64encode(open(p["path"], "rb").read()).decode()}})
        elif p["type"] == "video":
            raise ValueError("gemini: video input is not supported")
        else:
            out.append({"text": p["text"]})
    return out


def _gemini_tools(cfg, req, params, key):
    """Multi-turn Gemini generateContent with function calling.
    req.history in OpenAI chat format; an assistant turn with "raw_output" (the model's content parts, with their thought
    signatures) is replayed unchanged; a tool result becomes a functionResponse part, its images follow in a separate user turn."""
    contents, names, kinds = [], {}, []

    def add(role, parts, kind="msg"):
        # function responses must be a user turn of their own (a request where text or images share it can be rejected with
        # 400 "Requests ending with a model turn"): consecutive turns merge only if role and kind agree
        if contents and contents[-1]["role"] == role and kinds[-1] == kind:
            contents[-1]["parts"] += parts
        else:
            contents.append({"role": role, "parts": parts}); kinds.append(kind)
    for m in req.history + ([{"role": "user", "content": req.prompt}] if req.prompt is not None else []):
        if m["role"] in ("user", "system"):
            add("user", _gem_parts(m.get("content")))
        elif m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
            if m.get("raw_output"):
                add("model", list(m["raw_output"]))
            else:
                ps = [{"text": m["content"]}] if m.get("content") else []
                ps += [{"functionCall": {"name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"] or "{}")}}
                       for tc in m.get("tool_calls") or []]
                add("model", ps or [{"text": "(empty)"}])
        elif m["role"] == "tool":
            c = m.get("content"); txt = c if isinstance(c, str) or c is None else "\n".join(p["text"] for p in c if p["type"] == "text")
            fr = {"name": names.get(m["tool_call_id"], m.get("name", "tool")), "response": {"result": txt or "(empty)"}}
            if not str(m["tool_call_id"]).startswith("gem_"):
                fr["id"] = m["tool_call_id"]
            add("user", [{"functionResponse": fr}], "fr")
            if isinstance(c, list) and any(p["type"] == "image" for p in c):
                add("user", _gem_parts([p for p in c if p["type"] == "image"]), "img")
    params = dict(params); tools = params.pop("tools", None)
    body = _merge({"contents": contents}, params)
    if req.system:
        body["systemInstruction"] = {"parts": [{"text": req.system}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": f["name"], "description": f.get("description", ""), "parametersJsonSchema": f.get("parameters") or {"type": "object", "properties": {}}}
            for f in ((t["function"] if t.get("type") == "function" else t) for t in tools)]}]
    url = f"{os.path.expandvars(cfg.get('base_url', 'https://generativelanguage.googleapis.com/v1beta'))}/models/{cfg['api_model']}:generateContent"
    return url, {cfg.get("auth_header", "x-goog-api-key"): key}, body


def _gemini_tools_parse(j):
    r = _gemini_parse(j); c = (j.get("candidates") or [{}])[0]; parts = (c.get("content") or {}).get("parts") or []
    calls = [p["functionCall"] for p in parts if p.get("functionCall")]
    r["tool_calls"] = [{"id": fc.get("id") or f"gem_{random.getrandbits(48):012x}", "type": "function",
                        "function": {"name": fc["name"], "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}} for fc in calls]
    r["raw_content"] = parts
    if r["tool_calls"] and r["finish_reason"] == "stop":
        r["finish_reason"] = "tool_calls"
    return r


PROVIDERS = {"openai_chat": (_chat, _chat_parse), "openai_responses": (_responses, _responses_parse),
             "gemini": (_gemini_tools, _gemini_tools_parse)}


def _anthropic_blocks(c):
    """OpenAI-style content (str | parts with local image paths) -> Anthropic content blocks. Video is not accepted by Claude."""
    if c is None or isinstance(c, str):
        return [{"type": "text", "text": c or "(empty)"}]
    out = []
    for p in c:
        if p["type"] == "image":
            out.append({"type": "image", "source": {"type": "base64", "media_type": _mime(p),
                                                    "data": base64.standard_b64encode(open(p["path"], "rb").read()).decode()}})
        elif p["type"] == "video":
            raise ValueError("anthropic: video input is not supported")
        else:
            out.append({"type": "text", "text": p["text"]})
    return out


def _anthropic_messages(req):
    """req.history (+ req.prompt) in OpenAI chat format -> Anthropic messages. Tool results of consecutive tool messages go into one
    user turn (as the API wants); an assistant turn with "anthropic_content" is replayed verbatim."""
    out = []

    def add(role, blocks):
        if out and out[-1]["role"] == role and role == "user":
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})
    for m in req.history + ([{"role": "user", "content": req.prompt}] if req.prompt is not None else []):
        if m["role"] == "user":
            add("user", _anthropic_blocks(m.get("content")))
        elif m["role"] == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": _anthropic_blocks(m.get("content"))}])
        elif m["role"] == "assistant":
            if m.get("anthropic_content"):
                add("assistant", m["anthropic_content"])
            else:
                b = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                b += [{"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": json.loads(tc["function"]["arguments"] or "{}")}
                      for tc in m.get("tool_calls") or []]
                add("assistant", b)
    return out


def _anthropic_tool(t):
    """OpenAI function tool -> Anthropic tool definition."""
    f = t["function"] if t.get("type") == "function" else t
    return {"name": f["name"], "description": f.get("description", ""), "input_schema": f.get("parameters") or {"type": "object", "properties": {}}}


def _anthropic_parse(j):
    blocks = j.get("content") or []; u = j.get("usage") or {}
    rd, wr = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
    sr = j.get("stop_reason") or ""
    return dict(text="".join(b.get("text", "") for b in blocks if b["type"] == "text"),
                thoughts="\n".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
                tool_calls=[{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                            for b in blocks if b["type"] == "tool_use"],
                raw_content=blocks,
                finish_reason={"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "max_tokens", "refusal": "safety"}.get(sr, f"other:{sr}"),
                model_version=j.get("model", ""), input_tokens=(u.get("input_tokens") or 0) + rd + wr, cached_tokens=rd, cache_write_tokens=wr,
                output_tokens=u.get("output_tokens") or 0, thinking_tokens=0)


# ------------------------------------------------------------------------------------------------------------------- client
class Client:
    def __init__(self, budget_usd=None, models=None, cache_dir=CACHE_DIR, call_log=CALL_LOG, timeout_s=900, transport=None):
        self.models = models or load_models()
        self.budget, self.spent = budget_usd, 0.0
        self.cache_dir, self.call_log = Path(cache_dir), Path(call_log)
        self.http = httpx.Client(timeout=timeout_s, transport=transport)
        self.lock = threading.Lock()
        self.anthropic = None                   # SDK client, created on the first Anthropic call

    def cache_key(self, req):
        cfg = self.models[req.model]
        blob = json.dumps({"provider": cfg["provider"], "api_model": cfg["api_model"], "base_url": cfg.get("base_url", ""),
                           "system": req.system, "prompt": _keyable(req.prompt), "params": self._params(req), "sample_idx": req.sample_idx,
                           **({"history": [dict(m, content=_keyable(m.get("content"))) for m in req.history]} if req.history else {})},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _params(self, req):
        return _merge(self.models[req.model].get("params", {}), req.params)

    def _cost(self, cfg, r):
        p = cfg.get("price_usd_per_mtok") or {}
        if p.get("input") is None or p.get("output") is None:
            return None
        cached = r.get("cached_tokens", 0); pc = p.get("cached_input", p["input"])
        wr = r.get("cache_write_tokens", 0); pw = p.get("cache_write", p["input"])
        return ((r["input_tokens"] - cached - wr) * p["input"] + cached * pc + wr * pw + (r["output_tokens"] + r["thinking_tokens"]) * p["output"]) / 1e6

    def _log(self, req, res):
        rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "alias": req.model, "api_model": self.models[req.model]["api_model"],
               "tag": req.tag, **{k: v for k, v in asdict(res).items() if k not in ("text", "thoughts", "tool_calls", "raw_content")}, "n_tool_calls": len(res.tool_calls)}
        with self.lock:
            self.call_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.call_log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def call(self, req: Request) -> Result:
        cfg = self.models[req.model]
        if "FILL" in cfg["api_model"]:
            raise RuntimeError(f"{MODELS_JSON}: api_model of {req.model!r} is not set")
        key = self.cache_key(req)
        hit = self._cached(req, key)
        if hit:
            return hit
        with self.lock:
            if self.budget is not None:
                if self._cost(cfg, {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}) is None:
                    raise BudgetExceeded(f"{req.model}: no price in configs/models.json, cannot enforce budget")
                if self.spent >= self.budget:
                    raise BudgetExceeded(f"spent ${self.spent:.2f} >= budget ${self.budget:.2f}")
        if cfg["provider"] == "anthropic":
            return self._call_anthropic(req, key, cfg)
        if cfg["provider"] not in PROVIDERS:
            raise NotImplementedError(f"unknown provider {cfg['provider']!r} (anthropic, {', '.join(PROVIDERS)})")
        build, parse = PROVIDERS[cfg["provider"]]
        url, headers, body = build(cfg, req, self._params(req), _api_key(cfg["provider"], cfg))
        t0, err, j = time.time(), "", None
        for attempt in range(MAX_TRIES):
            try:
                r = self.http.post(url, headers=headers, json=body)
                if r.status_code == 200:
                    j = r.json(); break
                err = f"HTTP {r.status_code}: {r.text[:2000]}"
                if r.status_code == 429 and re.search(r"PerDay|per day|daily", r.text, re.I):
                    self._log(req, Result(error=err, key=key)); raise QuotaExhausted(err)
                if r.status_code not in RETRY_STATUS:
                    break
                m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', r.text)
                try:
                    wait = float(m.group(1)) + 1 if m else (float(r.headers.get("retry-after") or 0) or min(120, 2 ** attempt * 4))
                except ValueError:
                    wait = min(120, 2 ** attempt * 4)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                err, wait = f"{type(e).__name__}: {e}", min(120, 2 ** attempt * 4)
            if attempt < MAX_TRIES - 1:
                time.sleep(wait * (0.75 + 0.5 * random.random()))
        if j is None:
            res = Result(error=err, latency_s=time.time() - t0, key=key)
            self._log(req, res)
            return res
        return self._store(req, key, parse(j), j, time.time() - t0)

    def _call_anthropic(self, req, key, cfg):
        """Messages API, streamed (long thinking + large tool inputs), with automatic prompt caching of the whole prefix. The
        conversation is converted from the OpenAI chat format of req.history; assistant turns that carry "anthropic_content" are sent
        back exactly as returned (thinking blocks must not change). Retries: the SDK's (408 / 409 / 429 / 5xx / connection)."""
        import anthropic
        if self.anthropic is None:
            self.anthropic = anthropic.Anthropic(api_key=_api_key("anthropic", cfg), max_retries=MAX_TRIES, timeout=1800.0)
        params = self._params(req); tools = params.pop("tools", None)
        kw = dict(model=cfg["api_model"], messages=_anthropic_messages(req), cache_control={"type": "ephemeral"}, **params)
        if req.system:      # explicit breakpoint at the end of the static prefix (tools + system) + automatic one on the growing tail
            kw["system"] = [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            kw["tools"] = [_anthropic_tool(t) for t in tools]
        t0 = time.time()
        try:
            with self.anthropic.messages.stream(**kw) as st:
                msg = st.get_final_message()
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            res = Result(error=f"{type(e).__name__}: {str(e)[:2000]}", latency_s=time.time() - t0, key=key)
            self._log(req, res)
            return res
        j = msg.to_dict(); return self._store(req, key, _anthropic_parse(j), j, time.time() - t0)

    def _path(self, key):
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, req, key):
        path = self._path(key)
        if not path.exists():
            return None
        d = json.load(open(path))
        res = Result(**{**d["result"], "cached": True, "key": key, "latency_s": 0.0})
        self._log(req, res)
        return res

    def _store(self, req, key, p, j, latency):
        """Parsed provider answer -> Result, spend, cache file, log line."""
        cfg = self.models[req.model]; cost = self._cost(cfg, p)
        res = Result(**p, cost_usd=cost, latency_s=latency, key=key)
        with self.lock:
            self.spent += res.cost_usd or 0.0
        path = self._path(key); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        json.dump({"request": {**asdict(req), "params": self._params(req), "api_model": cfg["api_model"], "provider": cfg["provider"]},
                   "result": {k: v for k, v in asdict(res).items() if k not in ("cached", "key", "error")},
                   "date": dt.date.today().isoformat(), "batch": None, "raw": j}, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, path)
        self._log(req, res)
        return res


def spend_summary(call_log=CALL_LOG):
    if not Path(call_log).exists():
        print(f"no calls logged yet ({call_log})"); return
    agg = {}
    for l in open(call_log):
        r = json.loads(l)
        a = agg.setdefault((r["alias"], r.get("model_version") or r["api_model"]), dict(calls=0, cached=0, errors=0, inp=0, out=0, think=0, usd=0.0))
        a["calls"] += 1; a["cached"] += r["cached"]; a["errors"] += bool(r["error"])
        if not r["cached"]:
            a["inp"] += r["input_tokens"]; a["out"] += r["output_tokens"]; a["think"] += r["thinking_tokens"]; a["usd"] += r["cost_usd"] or 0.0
    print(f"{'alias':20s} {'version':32s} {'calls':>6s} {'cached':>6s} {'err':>4s} {'in_tok':>10s} {'out_tok':>9s} {'think_tok':>10s} {'usd':>8s}")
    for (al, ver), a in sorted(agg.items()):
        print(f"{al:20s} {ver:32s} {a['calls']:6d} {a['cached']:6d} {a['errors']:4d} {a['inp']:10d} {a['out']:9d} {a['think']:10d} {a['usd']:8.2f}")


if __name__ == "__main__":
    if "--spend" in sys.argv:
        spend_summary()
