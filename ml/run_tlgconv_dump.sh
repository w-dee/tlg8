#!/bin/bash

THIS_DIR=$(dirname $0)

python3 $THIS_DIR/run_tlgconv_dump.py \
    --jobs 15 \
    --input-dir $THIS_DIR/../test/copyrighted/ \
    --tlgconv $THIS_DIR/../build/tlgconv \
    --out-training-json $THIS_DIR/source/training-json \
    --out-label-cache $THIS_DIR/source/label-cache \
    --tlg8-temp-dir=/tmp/tlg8ml

