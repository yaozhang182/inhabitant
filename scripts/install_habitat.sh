#!/bin/bash
# Step 2 of 4 (inside the conda environment): HumanCLAW (half-physics controller and the body), habitat-sim 0.3.3 built with
# HumanCLAW's half-physics patch, and habitat-lab 0.3.0 with the two patches of patches/.
#   bash scripts/install_habitat.sh
# Everything goes to third_party/ (override with HUMANCLAW_ROOT). Building habitat-sim takes a while and needs a C++ compiler,
# CMake, CUDA and the OpenGL / EGL development files; HumanCLAW's patches/habitat-sim/README.md lists the host prerequisites
# and fixes for common build problems. The build commands below are HumanCLAW's (its README, "Install").
set -ex
REPO="$(cd "$(dirname "$0")/.." && pwd)"; TP="$REPO/third_party"; mkdir -p "$TP"
HC=${HUMANCLAW_ROOT:-$TP/HumanCLAW}
[ -d "$HC" ] || git clone https://github.com/Ryenhails/HumanCLAW.git "$HC"

# habitat-sim at the revision HumanCLAW pins, with its half-physics patch
if ! python -c "import habitat_sim" 2>/dev/null; then
    [ -d "$TP/habitat-sim" ] || git clone https://github.com/facebookresearch/habitat-sim.git "$TP/habitat-sim"
    cd "$TP/habitat-sim"
    git checkout acbe6f4922e68145e401e55c30f9dfea460a3f24
    git submodule update --init --recursive
    git apply --check "$HC/patches/habitat-sim/humanclaw_halfphysics.patch"
    git apply "$HC/patches/habitat-sim/humanclaw_halfphysics.patch"
    python -m pip install -r requirements.txt
    python setup.py build_ext --inplace --headless --with-cuda --bullet --cache-args
    HEADLESS=True WITH_CUDA=True WITH_BULLET=True python -m pip install -e .
    python -m pip install -e build/deps/magnum-bindings/src/python
fi

# habitat-lab 0.3.0 (the version EmbodiedBench's EB-Habitat uses), without its dependencies (they would replace habitat-sim)
HL="$TP/habitat-lab-0.3.0"
[ -d "$HL" ] || git clone -b v0.3.0 --depth 1 https://github.com/facebookresearch/habitat-lab.git "$HL"
python "$REPO/patches/patch_py311_dataclass.py" "$HL/habitat-lab"                 # Python 3.11: dataclass defaults
git -C "$HL" apply --check "$REPO/patches/habitat_lab_0.3.0_rearrange_sim_reset.patch" 2>/dev/null \
    && git -C "$HL" apply "$REPO/patches/habitat_lab_0.3.0_rearrange_sim_reset.patch" \
    || echo "rearrange_sim reset patch not applied (already applied?)"
python -m pip install -e "$HL/habitat-lab" --no-deps

python - <<'PY'
import torch, habitat_sim, habitat
print("torch", torch.__version__, torch.version.cuda)
print("habitat_sim", habitat_sim.__version__, "bullet", habitat_sim.built_with_bullet)
print("habitat-lab", habitat.__version__)
PY
