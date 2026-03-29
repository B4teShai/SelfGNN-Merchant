#!/bin/bash
# Run SelfGNN baseline on Yelp dataset
# Matches original TF hyperparameters from yelp.sh

python train.py \
    --data yelp \
    --reg 1e-2 \
    --lr 1e-3 \
    --temp 0.1 \
    --ssl_reg 1e-7 \
    --save_path yelp_baseline \
    --epoch 150 \
    --batch 512 \
    --sslNum 40 \
    --graphNum 12 \
    --gnn_layer 3 \
    --att_layer 2 \
    --testSize 1000 \
    --ssldim 32 \
    --sampNum 40 \
    --keepRate 0.5 \
    --leaky 0.5 \
    --tstEpoch 3 \
    --device mps \
    2>&1 | tee yelp_baseline.log
