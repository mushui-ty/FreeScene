#!/usr/bin/env bash
DEVICE=${1:-0}
EPOCH=${2:-1099}
MODE=${3:-scratch}

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m src.generate_mgdit \
  --checkpoint_epoch $EPOCH \
  --mode $MODE \
  --device 0 \
  --n_scenes 0 \
  --draw_scene_graph \
  --visualize \
  "$@"
