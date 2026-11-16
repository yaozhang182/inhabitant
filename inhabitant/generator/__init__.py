"""The motion generator service: Kimodo behind a small file-based RPC (gen_server.py), run as a separate process."""
import os

from ..paths import REPO, KIMODO_ROOT, KIMODO_WEIGHTS


def kimodo_env():
    """Environment of the generator process: Kimodo's code (and, if present, <repo>/third_party/kimodo_extra, where
    scripts/install_kimodo.sh puts Kimodo's extra Python packages) on PYTHONPATH; its checkpoint and text encoder read locally."""
    env = dict(os.environ)
    extra = os.environ.get("KIMODO_EXTRA") or str(REPO / "third_party/kimodo_extra")
    env["PYTHONPATH"] = os.pathsep.join([extra, str(KIMODO_ROOT), str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("CHECKPOINT_DIR", str(KIMODO_WEIGHTS))
    env.setdefault("HF_HUB_CACHE", str(KIMODO_WEIGHTS / "hf_hub"))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TEXT_ENCODER_MODE", "local")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env
