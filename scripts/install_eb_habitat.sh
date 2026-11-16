#!/bin/bash
# Step 3 of 4 (inside the conda environment): EmbodiedBench (the EB-Habitat episodes and configs the tasks are built on) and
# the EB-Habitat assets (ReplicaCAD scenes, YCB objects), then the start poses of the episodes.
#   bash scripts/install_eb_habitat.sh
# EmbodiedBench goes to third_party/EmbodiedBench (override with EMBODIEDBENCH_ROOT), the assets to data/eb_habitat
# (override the data directory with INHABITANT_DATA). The last step needs a GPU.
set -ex
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EB=${EMBODIEDBENCH_ROOT:-$REPO/third_party/EmbodiedBench}; DATA=${INHABITANT_DATA:-$REPO/data}
git lfs install
[ -d "$EB" ] || git clone https://github.com/EmbodiedBench/EmbodiedBench.git "$EB"
(cd "$EB" && git lfs pull)                                    # the episode pickles in embodiedbench/envs/eb_habitat/datasets
ls -la "$EB/embodiedbench/envs/eb_habitat/datasets"
[ -f "$EB/setup.py" ] && python -m pip install -e "$EB" --no-deps

mkdir -p "$DATA/eb_habitat" && cd "$DATA/eb_habitat"
[ -d data/replica_cad ] || python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets --data-path data
ln -sfn "$DATA/eb_habitat/data" "$EB/embodiedbench/envs/eb_habitat/data"       # EB-Habitat reads ./data from its package directory

# start poses of the EB episodes (data/eb_meta/<subset>/<id>.json); needs a GPU
cd "$REPO"
for s in base common_sense complex_instruction spatial_relationship visual_appearance long_horizon; do
    python scripts/export_eb_meta.py --eval_set $s --n 50
done
