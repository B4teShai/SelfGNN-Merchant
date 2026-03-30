#!/bin/bash
# SelfGNN baseline on Yelp-Merchant dataset
# graphNum=5 matches preprocessing notebook GRAPH_NUM

python train.py \
    --data yelp-merchant \
    --reg 1e-2 \
    --lr 1e-3 \
    --temp 0.1 \
    --ssl_reg 1e-7 \
    --save_path yelp_merchant_baseline \
    --epoch 150 \
    --batch 512 \
    --sslNum 40 \
    --graphNum 5 \
    --gnn_layer 3 \
    --att_layer 2 \
    --testSize 1000 \
    --ssldim 32 \
    --sampNum 40 \
    --keepRate 0.5 \
    --leaky 0.5 \
    --tstEpoch 3 \
    --patience 20 \
    --device cuda \
    2>&1 | tee yelp_merchant_baseline.log
