#!/bin/bash

THIS_DIR=$(dirname $0)

  python $THIS_DIR/make_dataset.py \
    --in-dir             $THIS_DIR/runs/packed_tv1y_tv2y \
    --out-dir            $THIS_DIR/runs/dataset_tv1y_tv2y \
    --feature-set        raw_plus_stats_v2_tv1y_tv2y
