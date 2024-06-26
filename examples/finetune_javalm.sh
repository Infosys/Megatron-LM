# Copyright 2024 Infosys Ltd.
# Use of this source code is governed by BSD-3 license that can be found in the LICENSE file or at
# https://opensource.org/license/bsd-3-clause

#!/bin/bash
# +
# !nvidia-smi
# -
# Runs the "pylm" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=6
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

SAVED_CHECKPOINT_PATH=<specify path to model checkpoint>


NEW_CHECKPOINT_PATH=<Path to save the new checkpoints>
# NEW_CHECKPOINT_PATH=../../checkpoints/temp

VOCAB_FILE=<Specify path to vocab.json>
MERGE_FILE=<Specify path to merges.txt>
DATA_PATH=<Specify path to processed data>
TOKENIZER_FILE=<Specify path to tokenizer.json>
# -

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 2048 \
    --attention-head-type multiquery \
    --num-attention-heads 16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --attention-dropout 0.1 \
    --hidden-dropout 0.1 \
    --micro-batch-size 1 \
    --global-batch-size 180 \
    --lr 0.0004 \
    --min-lr 0.000004 \
    --train-iters 100000 \
    --lr-decay-iters 100000 \
    --lr-decay-style cosine \
    --lr-warmup-iters 1000 \
    --weight-decay .1 \
    --adam-beta2 .95 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --init-method-std 0.02209 \
    --no-load-optim \
    --no-load-rng \
    --finetune"


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --tokenizer-file $TOKENIZER_FILE \
    --data-impl mmap \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 5000 \
    --eval-interval 2000 \
    --eval-iters 10000 \
"

torchrun $DISTRIBUTED_ARGS ../pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $NEW_CHECKPOINT_PATH \
    --load $SAVED_CHECKPOINT_PATH \
