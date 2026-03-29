# SelfGNN PyTorch Port

PyTorch implementation of SelfGNN (Self-Supervised Graph Neural Networks for Sequential Recommendation), ported from the [original TensorFlow code](https://github.com/HKUDS/SelfGNN).

## Project Structure

```
selfgnn_pytorch/
├── config.py          # Hyperparameters (matches original Params.py)
├── data_handler.py    # Data loading & sampling (matches DataHandler.py)
├── model.py           # SelfGNN model in PyTorch (matches model.py)
├── train.py           # Training + evaluation loop
├── run_yelp.sh        # Run Yelp experiment with original hyperparameters
├── requirements.txt
└── Datasets/
    └── Yelp/
        ├── trn_mat_time   # Pickled: [overall_mat, sub_mats, time_mat]
        ├── tst_int        # Pickled: test items per user
        ├── sequence       # Pickled: user behavior sequences
        └── test_dict      # Pickled: negative items for evaluation
```

## Setup

```bash
# 1. Install dependencies
pip install torch numpy scipy

# 2. Copy Yelp dataset from original SelfGNN repo
#    The dataset files should be in ./Datasets/Yelp/
cp -r /path/to/original/SelfGNN/Datasets/Yelp ./Datasets/

# 3. Run
chmod +x run_yelp.sh
./run_yelp.sh
```

## Expected Results (Yelp)

From the original SelfGNN paper (Table 2):

| Metric   | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|----------|-------|---------|-------|---------|
| SelfGNN  | 0.365 | 0.201   | 0.509 | 0.237   |

## Key Differences from TF Version

1. **Proper edge dropout**: Original TF code had a subtle bug where edge dropout had no effect (values cast to int32 became 0, and `segment_sum` ignored values). This port implements proper edge dropout.

2. **PyTorch sparse ops**: Uses `torch.sparse.mm` for graph message passing.

3. **Modern PyTorch**: Uses `nn.Module`, `nn.Parameter`, built-in LSTM, and standard training loop.

## Run on Other Datasets

```bash
# Amazon
python train.py --data amazon --graphNum 5 --gnn_layer 3 --att_layer 4 \
    --sslNum 80 --ssl_reg 1e-6 --pred_num 0 --pos_length 200

# Gowalla  
python train.py --data gowalla --graphNum 3 --gnn_layer 2 --att_layer 1 \
    --lr 2e-3 --sslNum 40 --ssl_reg 1e-6 --ssldim 48

# MovieLens
python train.py --data movielens --graphNum 6 --gnn_layer 2 --att_layer 3 \
    --sslNum 90 --ssl_reg 1e-6 --ssldim 48
```
