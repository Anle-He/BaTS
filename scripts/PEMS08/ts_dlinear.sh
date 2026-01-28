#!/bin/bash

MODEL = 'DLinear'
TASK = 'LTSF'
DATASET = 'PEMS08'

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/DLinear/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2024

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/DLinear/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2025

python -u main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg models/DLinear/configs/PEMS08_IN96_OUT12.yaml \
    -sd 2026