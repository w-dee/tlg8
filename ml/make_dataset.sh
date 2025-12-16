#!/bin/bash

THIS_DIR=$(dirname $0)

  python $THIS_DIR/make_dataset.py \
    --in-dir             $THIS_DIR/runs/packed \
    --out-dir            $THIS_DIR/runs/dataset