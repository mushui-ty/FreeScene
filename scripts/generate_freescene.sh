#!/usr/bin/env bash
# Full FreeScene pipeline: VLM Graph Designer + MGDiT
DEVICE=${1:-0}
EPOCH=${2:-1099}

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m src.generate_freescene \
  --checkpoint_epoch $EPOCH \
  --device 0 \
  --draw_scene_graph \
  "${@:3}"
