#!/usr/bin/env bash
DEVICE=${1:-0}

CUDA_VISIBLE_DEVICES=$DEVICE python3 -m src.train_mgdit --device 0 "$@"
