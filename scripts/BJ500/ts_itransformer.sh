#!/bin/bash

MODEL='iTransformer'
TASK='LTSF'
DATASET='BJ500'

# 设置 PYTHONPATH 以支持 bats 模块导入
export PYTHONPATH="$(cd "$(dirname "$0")/../.." && pwd):$PYTHONPATH"

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT12.yaml \
    -sd 2024

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT12.yaml \
    -sd 2025

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT12.yaml \
    -sd 2026


python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT24.yaml \
    -sd 2024

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT24.yaml \
    -sd 2025

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT24.yaml \
    -sd 2026


python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT48.yaml \
    -sd 2024

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT48.yaml \
    -sd 2025

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT48.yaml \
    -sd 2026


python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT96.yaml \
    -sd 2024

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT96.yaml \
    -sd 2025

python -u bats/main.py \
    -m $MODEL \
    -t $TASK \
    -d $DATASET \
    -cfg bats/models/$MODEL/configs/BJ500_IN96_OUT96.yaml \
    -sd 2026