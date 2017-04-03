#!/usr/bin/env bash

SAVE_PATH=./data/
TRAIN_DATA=./data/cooccur_matrix.tfrecord
EMBEDDING_SIZE=200
EPOCHS=15
LEARNING_RATE=0.05
BATCH_SIZE=5000
VOCAB_SIZE=18497
MATRIX_SIZE=17724348

python glove.py \
    --save_path=$SAVE_PATH \
    --train_data=$TRAIN_DATA \
    --embedding_size=$EMBEDDING_SIZE \
    --epochs_to_train=$EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE \
    --vocab_size=$VOCAB_SIZE \
    --matrix_size=$MATRIX_SIZE
