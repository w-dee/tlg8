#!/bin/bash

THIS_DIR=$(dirname $0)

  python $THIS_DIR/pack_artifacts.py \
    --training-json-root $THIS_DIR/runs/training-json_tv1y_tv2y \
    --label-cache-root   $THIS_DIR/runs/label-cache_tv1y_tv2y \
    --out-dir            $THIS_DIR/runs/packed_tv1y_tv2y