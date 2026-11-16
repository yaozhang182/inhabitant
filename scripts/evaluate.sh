#!/bin/bash
# Evaluation of one or more sets of runs (CPU only): a report per episode (report.md, final.mp4, final_motion.npz, stats.json),
# then the tables of each set in <tag dir>/summary.md.
#   scripts/evaluate.sh runs/claude-opus-5.5/main [more tag dirs ...]
# The package must be installed (pip install -e .).
set -e
for TAG in "$@"; do
    TAG=${TAG%/}; RUNS=()
    for d in "$TAG"/*/; do [ -f "$d/log.json" ] && RUNS+=("${d%/}"); done
    [ ${#RUNS[@]} -gt 0 ] || { echo "no runs with a log.json in $TAG"; continue; }
    python -m inhabitant.eval.episode_report "${RUNS[@]}"
    python -m inhabitant.eval.summarize "$TAG" --md "$TAG/summary.md" > /dev/null
    echo "summary -> $TAG/summary.md"
done
