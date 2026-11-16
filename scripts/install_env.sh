#!/bin/bash
# Step 1 of 4: the conda environment (Python 3.11, torch 2.6 / CUDA 12.4) and this package.
#   bash scripts/install_env.sh [env name or prefix, default: inhabitant]
# Then: conda activate inhabitant; bash scripts/install_habitat.sh; bash scripts/install_eb_habitat.sh; bash scripts/install_kimodo.sh
set -ex
REPO="$(cd "$(dirname "$0")/.." && pwd)"; ENV=${1:-inhabitant}
case "$ENV" in */*) SEL=(-p "$ENV");; *) SEL=(-n "$ENV");; esac
conda create -y "${SEL[@]}" -c conda-forge python=3.11 pip
# GL / EGL runtime libraries for habitat-sim's headless renderer, git-lfs for EmbodiedBench, and a libstdc++ new enough for a
# habitat-sim built with gcc 13 (GLIBCXX_3.4.32)
conda install -y "${SEL[@]}" -c conda-forge libglvnd libegl libopengl libgl git-lfs "libstdcxx-ng>=13" "libgcc-ng>=13"
RUN=(conda run --no-capture-output "${SEL[@]}")
"${RUN[@]}" python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"${RUN[@]}" python -m pip install "numpy==1.26.4" "pillow==10.4.0" gym==0.23.1 hydra-core omegaconf smplx trimesh rtree \
    pyyaml tqdm matplotlib einops scikit-learn opencv-python-headless transformers
"${RUN[@]}" python -m pip install -e "$REPO"
