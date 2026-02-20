#!/bin/bash

MODEL='DLinear'
TASK='LTSF'
DATASET='PEMS08'

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2024

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2025

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2026


python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT24.yaml \
    -sd 2024

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT24.yaml \
    -sd 2025

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT24.yaml \
    -sd 2026


python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT48.yaml \
    -sd 2024

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT48.yaml \
    -sd 2025

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT48.yaml \
    -sd 2026


python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT96.yaml \
    -sd 2024

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT96.yaml \
    -sd 2025

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/$MODEL/configs/PEMS08_IN96_OUT96.yaml \
    -sd 2026