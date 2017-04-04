#!/usr/bin/env bash

SAVE_PATH=./data/
TRAIN_DATA=./data/cooccur_matrix.csv
EMBEDDING_SIZE=200
EPOCHS=2
LEARNING_RATE=0.05
BATCH_SIZE=1000
MATRIX_SIZE=17724348
CONCURRENT_STEPS=2
LOAD_DATA_PER_TIME=100
VOCAB_DATA=./data/vocab.txt
EVAL_DATA=./data/questions-words.txt

python glove.py \
    --save_path=$SAVE_PATH \
    --train_data=$TRAIN_DATA \
    --vocab_data=$VOCAB_DATA \
    --eval_data=$EVAL_DATA \
    --embedding_size=$EMBEDDING_SIZE \
    --epochs_to_train=$EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE \
    --matrix_size=$MATRIX_SIZE \
    --concurrent_steps=$CONCURRENT_STEPS \
    --load_data_per_time=$LOAD_DATA_PER_TIME
