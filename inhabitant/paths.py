"""Where things are. Every location has a default inside the repository and can be moved with an environment variable.

  INHABITANT_DATA      data directory (default <repo>/data): eb_habitat/data (EB-Habitat assets), eb_meta/ (start poses,
                       written by scripts/export_eb_meta.py), kimodo/ (Kimodo checkpoint and its text encoder cache)
  INHABITANT_RUNS      where runs are written (default <repo>/runs)
  HUMANCLAW_ROOT       clone of HumanCLAW: half-physics controller and the body (default <repo>/third_party/HumanCLAW)
  EMBODIEDBENCH_ROOT   clone of EmbodiedBench: EB-Habitat episodes and configs (default <repo>/third_party/EmbodiedBench)
  KIMODO_ROOT          clone of Kimodo (default <repo>/third_party/kimodo)
  KIMODO_WEIGHTS       Kimodo checkpoint directory (default $INHABITANT_DATA/kimodo)
  INHABITANT_MODELS    model configuration of the LLM client (default <repo>/configs/models.json)"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dir(env, default):
    return Path(os.environ.get(env) or default).expanduser()


DATA = _dir("INHABITANT_DATA", REPO / "data")
RUNS = _dir("INHABITANT_RUNS", REPO / "runs")
HUMANCLAW_ROOT = _dir("HUMANCLAW_ROOT", REPO / "third_party/HumanCLAW")
EMBODIEDBENCH_ROOT = _dir("EMBODIEDBENCH_ROOT", REPO / "third_party/EmbodiedBench")
KIMODO_ROOT = _dir("KIMODO_ROOT", REPO / "third_party/kimodo")
KIMODO_WEIGHTS = _dir("KIMODO_WEIGHTS", DATA / "kimodo")
MODELS_JSON = _dir("INHABITANT_MODELS", REPO / "configs/models.json")
BENCHMARK = REPO / "benchmark/inhabit"
EB_HABITAT = EMBODIEDBENCH_ROOT / "embodiedbench/envs/eb_habitat"       # EB-Habitat package directory (datasets/, dataset/configs/)
EB_DATA = DATA / "eb_habitat/data"                                      # ReplicaCAD scenes and YCB objects
EB_META = DATA / "eb_meta"                                              # <subset>/<episode>.json
