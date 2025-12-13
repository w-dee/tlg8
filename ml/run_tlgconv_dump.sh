#!/bin/bash

THIS_DIR=$(dirname $0)

python3 $THIS_DIR/run_tlgconv_dump.py \
    --input-dir $THIS_DIR/../test/copyrighted/ \
    --tlgconv $THIS_DIR/../build/tlgconv \
    --out-training-json $THIS_DIR/runs/training-json \
    --out-label-cache $THIS_DIR/runs/label-cache \
    --tlg8-temp-dir=/tmp/tlg8ml

