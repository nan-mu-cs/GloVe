#!/usr/bin/env bash

SAVE_PATH=./data/
DATA_PATH=./data/text8
WINDOW_SIZE=5
MIN_COUNT=50

python clean_data.py \
    --save_path=$SAVE_PATH \
    --data_file=$DATA_PATH \
    --window_size=$WINDOW_SIZE \
    --min_count=$MIN_COUNT

