#!/bin/bash
# Start one Kimodo generator service that several episodes share (about 19 GB of GPU memory with the text encoder on the GPU;
# TEXT_ENCODER_DEVICE=cpu keeps the text encoder off the GPU).
#   scripts/run_generator.sh --rpc runs/gen_rpc --shared &
#   GEN_RPC_SHARED=runs/gen_rpc python -m inhabitant.agent --task low_01 ...
# Without a shared service every episode starts its own generator.
exec python -m inhabitant.generator "$@"
