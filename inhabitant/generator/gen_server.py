"""Motion generator service: one resident Kimodo model, file RPC.

The client (GenClient in inhabitant/episode.py) writes <rpc>/req_<n>.json (n = 1, 2, ...), the server answers
<rpc>/resp_<n>.json; the client deletes both after reading. Requests:
  {"cmd": "generate", text, seconds, constraints, seed, post, cfg, prev}   see gen_kimodo.py
  {"cmd": "fk" | "native", ...}                                            inspect fk / pose
  {"cmd": "quit"}
Response: {"joints": "<npy (T,22,3) segment frame>", "fps", "frame", "gen_s", "extra": "<npz of the other arrays (rot, root)>" or
null, ...} or {"error": ..., "tb": ...}.

  scripts/run_generator.sh --rpc <dir> [--shared]
  --shared: one model for several clients (episodes), each in its own subdirectory <rpc>/<client>/ with its own req / resp
  numbering; the server answers them in turn and runs until killed ("quit" only ends that client). Client side:
  GEN_RPC_SHARED=<rpc>."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

CWD = Path.cwd()                # relative "out" paths are resolved here


def load_kimodo(a):
    from .gen_kimodo import Kimodo
    return Kimodo(device=a.device)


def handle(model, r, rpc, n):
    if r.get("cmd") in ("fk", "native"):            # Kimodo: forward kinematics / executed pose as native pose (inspect fk, pose)
        return dict(result=getattr(model, r["cmd"])(**{k: v for k, v in r.items() if k != "cmd"}), joints="-")
    if r.get("cmd", "generate") != "generate":
        raise ValueError(f"unknown cmd {r.get('cmd')!r}")
    out = model.generate(**{k: v for k, v in r.items() if k not in ("cmd", "out")})
    path = CWD / (r.get("out") or rpc / f"gen_{n:05d}.npy"); path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(out["joints"], np.float32))
    extra = {k: v for k, v in out.items() if isinstance(v, np.ndarray) and k != "joints"}
    if extra:
        np.savez(path.with_suffix(".npz"), **extra)
    return dict(joints=str(path), extra=str(path.with_suffix(".npz")) if extra else None, fps=out["fps"], frame=out.get("frame", "segment"),
                gen_s=round(out.get("gen_s", 0.0), 2), T=int(len(out["joints"])))


def serve(model, rpc):
    for f in rpc.glob("*"):
        if f.is_file():
            f.unlink()
    (rpc / "READY").write_text("{}"); print("READY", flush=True); n = 0
    while True:
        req = rpc / f"req_{n + 1:05d}.json"
        if not req.exists():
            time.sleep(0.05); continue
        n += 1; r = json.loads(req.read_text())
        if r.get("cmd") == "quit":
            (rpc / f"resp_{n:05d}.json").write_text("{}"); return
        try:
            resp = handle(model, r, rpc, n)
        except Exception as ex:  # noqa: BLE001
            resp = dict(error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc())
        print(f"[{n}] {r.get('text')!r} -> {resp.get('error') or resp['joints']} ({resp.get('gen_s')} s)", flush=True)
        tmp = rpc / f"resp_{n:05d}.json.tmp"; tmp.write_text(json.dumps(resp)); tmp.rename(rpc / f"resp_{n:05d}.json")


def serve_shared(model, root):
    """several clients, one subdirectory each (created by the clients), numbered independently"""
    (root / "READY").write_text("{}"); print("READY (shared)", flush=True); nxt = {}
    while True:
        busy = False
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            if d.name not in nxt:  # a restarted server resumes at the oldest pending request
                reqs = sorted(d.glob("req_*.json"))
                nxt[d.name] = int(reqs[0].stem[4:]) if reqs else 1
            req = d / f"req_{nxt.get(d.name, 1):05d}.json"
            if not req.exists():
                continue
            try:                    # a client that ended may delete its last request while we look at it
                r = json.loads(req.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            n = nxt.get(d.name, 1); nxt[d.name] = n + 1; busy = True
            if r.get("cmd") == "quit":
                (d / f"resp_{n:05d}.json").write_text("{}"); nxt.pop(d.name); continue
            try:
                resp = handle(model, r, d, n)
            except Exception as ex:  # noqa: BLE001
                resp = dict(error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc())
            print(f"[{d.name} {n}] {r.get('text')!r} -> {resp.get('error') or resp['joints']} ({resp.get('gen_s')} s)", flush=True)
            tmp = d / f"resp_{n:05d}.json.tmp"; tmp.write_text(json.dumps(resp)); tmp.rename(d / f"resp_{n:05d}.json")
        if not busy:
            time.sleep(0.05)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rpc", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--shared", action="store_true", help="several clients, one subdirectory of --rpc each")
    a = ap.parse_args(); rpc = Path(a.rpc).resolve(); rpc.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); model = load_kimodo(a); print(f"loaded kimodo in {time.time() - t0:.1f} s", flush=True)
    serve_shared(model, rpc) if a.shared else serve(model, rpc)
