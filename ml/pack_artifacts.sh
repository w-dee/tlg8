#!/bin/bash

THIS_DIR=$(dirname $0)

  python $THIS_DIR/pack_artifacts.py \
    --training-json-root $THIS_DIR/runs/training-json \
    --label-cache-root   $THIS_DIR/runs/label-cache \
    --out-dir            $THIS_DIR/runs/packed