#!/usr/bin/env bash

python train.py \
    --data-dir 'data/' \
    --model-dir 'saved-models/' \
    --model-name 'keras.model' \
    --num-epochs 50 \
    --label-bin 'data/data.pickle' \
    --plot 'images/loss_acc.png'