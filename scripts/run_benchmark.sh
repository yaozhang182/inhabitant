#!/bin/bash
# The Inhabit benchmark (or some of its tasks) with the built-in agent loop, on one GPU, with one shared Kimodo.
#   scripts/run_benchmark.sh <model alias> <tag> [task ids ...]        (no ids = all 75 tasks)
#   JOBS=3 scripts/run_benchmark.sh claude-opus-5.5 main               (3 episodes at a time; default 1)
# Each episode holds a simulator (a few GB of GPU memory) next to the shared generator (about 19 GB). With more than about 3
# episodes on one generator, generation becomes the bottleneck. Episodes that already have a log.json are skipped.
# Runs: runs/<model>/<tag>/inhabit_<id>/ (or under $INHABITANT_RUNS); logs: logs/<tag>/. Afterwards: scripts/evaluate.sh runs/<model>/<tag>
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${1:?model alias of configs/models.json}; TAG=${2:?tag}; shift 2
IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=($(ls "$REPO"/benchmark/inhabit/*.json | xargs -n1 basename | sed 's/\.json$//'))
JOBS=${JOBS:-1}; RUNS=${INHABITANT_RUNS:-$REPO/runs}; LOGS=$REPO/logs/$TAG; mkdir -p "$LOGS"
export GEN_RPC_SHARED="$LOGS/gen_rpc_$$"; rm -rf "$GEN_RPC_SHARED"; mkdir -p "$GEN_RPC_SHARED"
python -m inhabitant.generator --rpc "$GEN_RPC_SHARED" --shared > "$LOGS/generator.log" 2>&1 &
G=$!; trap 'kill $G 2>/dev/null; rm -rf "$GEN_RPC_SHARED"' EXIT
until [ -f "$GEN_RPC_SHARED/READY" ]; do sleep 5; kill -0 $G 2>/dev/null || { echo "generator failed: $LOGS/generator.log"; exit 1; }; done
for id in "${IDS[@]}"; do
    [ -f "$RUNS/$MODEL/$TAG/inhabit_$id/log.json" ] && { echo "skip $id (has a log)"; continue; }
    while [ "$(jobs -rp | grep -vc "^$G\$")" -ge "$JOBS" ]; do sleep 2; done
    echo "start $id"
    python -m inhabitant.agent --task "$id" --model "$MODEL" --tag "$TAG" > "$LOGS/$id.log" 2>&1 &
done
for p in $(jobs -rp | grep -v "^$G\$"); do wait "$p"; done
echo "done: $RUNS/$MODEL/$TAG"
