#!/bin/bash
# Step 4 of 4 (inside the conda environment): Kimodo's code (unmodified, Apache-2.0) and the extra Python packages it needs,
# kept apart from the main environment in third_party/kimodo_extra (pip --no-deps --target) so that they do not change it.
#   bash scripts/install_kimodo.sh
# The weights are NOT downloaded here (gated repositories, see the README, "Kimodo weights"):
#   <data>/kimodo/Kimodo-SMPLX-RP-v1/     the checkpoint (NVIDIA R&D model licence, research use only)
#   <data>/kimodo/hf_hub/                 Hugging Face cache with the text encoder: meta-llama/Meta-Llama-3-8B-Instruct and
#                                         McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp (and -mntp-supervised)
set -ex
REPO="$(cd "$(dirname "$0")/.." && pwd)"; TP="$REPO/third_party"; mkdir -p "$TP"
KIMODO=${KIMODO_ROOT:-$TP/kimodo}; EXTRA=${KIMODO_EXTRA:-$TP/kimodo_extra}
[ -d "$KIMODO" ] || git clone https://github.com/nv-tlabs/kimodo.git "$KIMODO"
python -m pip install --no-deps --target "$EXTRA" peft accelerate psutil
python -m pip install --no-deps --target "$EXTRA" "$KIMODO/MotionCorrection"     # C++ extension of Kimodo's post-processing (needs CMake, gcc)
PYTHONPATH="$EXTRA:$KIMODO:$PYTHONPATH" python -c "import kimodo, peft, accelerate, motion_correction; print('kimodo ok')"
