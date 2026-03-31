import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SelfGNN PyTorch with Features')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch', default=512, type=int)
    parser.add_argument('--reg', default=1e-2, type=float, help='L2 regularization weight')
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--graphNum', default=5, type=int, help='number of time-interval sub-graphs')
    parser.add_argument('--decay', default=0.96, type=float, help='lr decay rate')
    parser.add_argument('--save_path', default='yelp_merchant_feature', type=str)
    parser.add_argument('--latdim', default=64, type=int, help='embedding dimension')
    parser.add_argument('--ssldim', default=32, type=int, help='SAL weight MLP hidden dim')
    parser.add_argument('--sampNum', default=40, type=int, help='negative samples per user in train')
    parser.add_argument('--testSize', default=1000, type=int, help='neg + 1 pos for evaluation')
    parser.add_argument('--sslNum', default=40, type=int, help='SSL sample size')
    parser.add_argument('--num_attention_heads', default=16, type=int)
    parser.add_argument('--gnn_layer', default=3, type=int)
    parser.add_argument('--trnNum', default=10000, type=int, help='users sampled per epoch')
    parser.add_argument('--data', default='yelp-merchant', type=str)
    parser.add_argument('--keepRate', default=0.5, type=float, help='edge keep rate (1 - dropout)')
    parser.add_argument('--pos_length', default=200, type=int, help='max sequence length')
    parser.add_argument('--att_layer', default=2, type=int, help='sequence self-attention layers')
    parser.add_argument('--pred_num', default=5, type=int)
    parser.add_argument('--temp', default=0.1, type=float)
    parser.add_argument('--ssl_reg', default=1e-7, type=float, help='SAL loss weight')
    parser.add_argument('--leaky', default=0.5, type=float, help='leaky relu slope')
    parser.add_argument('--tstEpoch', default=3, type=int, help='evaluate every N epochs')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--patience', default=20, type=int,
                        help='early stopping patience (counted in evaluation epochs)')
    # Feature flags
    parser.add_argument('--use_edge_features', action='store_true', default=False,
                        help='use edge weights (star ratings) in adjacency')
    parser.add_argument('--use_node_features', action='store_true', default=False,
                        help='use user/merchant features via MLP projection')
    parser.add_argument('--node_mlp_hidden', type=int, default=64,
                        help='hidden dimension of feature projection MLP')
    return parser.parse_args()

args = parse_args()
