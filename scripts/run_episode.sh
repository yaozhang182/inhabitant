#!/bin/bash
# One task with the built-in agent loop.
#   scripts/run_episode.sh <task id | task file.json> [model alias] [tag]
#   scripts/run_episode.sh low_01 claude-opus-5.5 main
# Run directory: runs/<model>/<tag>/inhabit_<task id>/ (log.json, tries/). Needs a GPU, configs/models.json, the model's API key
# and the installed package (pip install -e .).
TASK=${1:?task id (e.g. low_01) or a task file}; MODEL=${2:-claude-opus-5.5}; TAG=${3:-main}
if [ -f "$TASK" ]; then ARG=(--spec "$TASK"); else ARG=(--task "$TASK"); fi
exec python -m inhabitant.agent "${ARG[@]}" --model "$MODEL" --tag "$TAG"
