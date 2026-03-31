import pickle
import ast
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
import os
import torch


def build_sparse_adj(rows, cols, vals, shape):
    """Build PyTorch sparse tensor from COO arrays."""
    idx = torch.stack([torch.LongTensor(rows), torch.LongTensor(cols)])
    v = torch.FloatTensor(vals)
    return torch.sparse_coo_tensor(idx, v, size=shape).coalesce()


def build_binary_adj(mat, shape):
    """Convert scipy sparse matrix to binary PyTorch sparse tensor."""
    coo = sp.coo_matrix(mat)
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    vals = np.ones(len(coo.data), dtype=np.float32)
    return build_sparse_adj(rows, cols, vals, shape)


def build_weighted_adj(mat, edge_weights_arr, shape):
    """Build weighted normalized adjacency from scipy sparse + weight array.

    Weights are log-sigmoid transformed, then symmetric-normalized.
    edge_weights_arr: numpy array of shape (num_users, num_items) or dict.
    Accepts either a pre-built CSR weight matrix or a dict.
    """
    coo = sp.coo_matrix((mat != 0).astype(np.float32))
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)

    # Vectorized weight lookup via CSR matrix
    if sp.issparse(edge_weights_arr):
        # Fast path: edge_weights_arr is a CSR matrix with rating values
        raw = np.asarray(edge_weights_arr[rows, cols]).flatten()
        raw = np.where(raw > 0, raw, 1.0)
    else:
        # Fallback: dict lookup (slow, only for small matrices)
        edge_weights = edge_weights_arr
        raw = np.ones(len(rows), dtype=np.float64)
        for idx in range(len(rows)):
            key = (int(rows[idx]), int(cols[idx]))
            if key in edge_weights:
                raw[idx] = edge_weights[key]

    # log-sigmoid: w_hat = sigmoid(log(1 + w))
    w_hat = 1.0 / (1.0 + np.exp(-np.log1p(raw)))

    # Symmetric normalization: w_norm[i,j] = w_hat[i,j] / sqrt(D_row[i]) / sqrt(D_col[j])
    num_rows, num_cols = shape
    D_row = np.zeros(num_rows, dtype=np.float64)
    D_col = np.zeros(num_cols, dtype=np.float64)
    np.add.at(D_row, rows, w_hat)
    np.add.at(D_col, cols, w_hat)

    D_row_inv_sqrt = np.power(D_row + 1e-8, -0.5)
    D_col_inv_sqrt = np.power(D_col + 1e-8, -0.5)
    norm_vals = (w_hat * D_row_inv_sqrt[rows] * D_col_inv_sqrt[cols]).astype(np.float32)

    adj = build_sparse_adj(rows, cols, norm_vals, shape)
    adj_t = build_sparse_adj(cols, rows, norm_vals, (num_cols, num_rows))
    return adj, adj_t


class DataHandler:
    def __init__(self, args):
        self.args = args
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self.predir = os.path.join(_script_dir, '..', 'Datasets', args.data) + os.sep

    def load_data(self):
        args = self.args

        # ---- trn_mat_time: [global_mat, subMat_list, timeMat] ----
        with open(self.predir + 'trn_mat_time', 'rb') as f:
            trnMat = pickle.load(f)
        args.user, args.item = trnMat[0].shape
        print(f'Users: {args.user}, Items(Merchants): {args.item}')

        self.subMat = trnMat[1]

        # ---- sequence ----
        with open(self.predir + 'sequence', 'rb') as f:
            self.sequence = pickle.load(f)

        # Build overall binary train matrix from sequences
        row, col, data = [], [], []
        for uid, item_list in enumerate(self.sequence):
            for iid in item_list:
                row.append(uid)
                col.append(iid)
                data.append(1)
        self.trnMat = csr_matrix(
            (np.array(data), (np.array(row), np.array(col))),
            shape=(args.user, args.item)
        )

        # ---- tst_int ----
        with open(self.predir + 'tst_int', 'rb') as f:
            tstInt = pickle.load(f)
        self.tstInt = np.array(tstInt, dtype=object)
        tstStat = np.array([x is not None for x in tstInt])
        self.tstUsrs = np.where(tstStat)[0]
        print(f'Test users: {len(self.tstUsrs)}')

        # ---- val_int ----
        val_path = self.predir + 'val_int'
        if os.path.isfile(val_path):
            with open(val_path, 'rb') as f:
                valInt = pickle.load(f)
            self.valInt = np.array(valInt, dtype=object)
            valStat = np.array([x is not None for x in valInt])
            self.valUsrs = np.where(valStat)[0]
            print(f'Val users: {len(self.valUsrs)}')
        else:
            self.valInt = None
            self.valUsrs = np.array([], dtype=int)
            print('No val_int found, validation disabled.')

        # ---- test_dict ----
        if os.path.isfile(self.predir + 'test_dict'):
            with open(self.predir + 'test_dict', 'rb') as f:
                self.test_dict = pickle.load(f)
        else:
            self.test_dict = {}

        # ---- val_dict ----
        val_dict_path = self.predir + 'val_dict'
        val_csv_path = self.predir + 'val_yelp_merchant.csv'
        if os.path.isfile(val_dict_path):
            with open(val_dict_path, 'rb') as f:
                self.val_dict = pickle.load(f)
            print(f'Val dict loaded: {len(self.val_dict)} users')
        elif os.path.isfile(val_csv_path):
            self.val_dict = {}
            df_val = pd.read_csv(val_csv_path, sep='\t')
            for _, row in df_val.iterrows():
                uid = int(row['user_id']) - 1
                negs = ast.literal_eval(str(row['neg_merchants']))
                self.val_dict[uid] = negs
            print(f'Val dict from CSV: {len(self.val_dict)} users')
        else:
            self.val_dict = {}

        # ---- Load edge weights (optional) ----
        ew_csr = None
        if args.use_edge_features:
            ew_path = self.predir + 'edge_weights.pkl'
            if os.path.isfile(ew_path):
                with open(ew_path, 'rb') as f:
                    edge_weights = pickle.load(f)
                print(f'Edge weights loaded: {len(edge_weights):,} pairs — building CSR...')
                ew_rows = np.array([k[0] for k in edge_weights], dtype=np.int32)
                ew_cols = np.array([k[1] for k in edge_weights], dtype=np.int32)
                ew_vals = np.array(list(edge_weights.values()), dtype=np.float32)
                ew_csr = csr_matrix((ew_vals, (ew_rows, ew_cols)),
                                    shape=(args.user, args.item))
                del edge_weights
                print('  CSR weight matrix built.')
            else:
                print('WARNING: edge_weights.pkl not found, using binary adjacency')

        # ---- Build PyTorch sparse adjacency ----
        self.sub_adj = []
        self.sub_adj_t = []
        for i in range(len(self.subMat)):
            mat = self.subMat[i]
            if args.use_edge_features and ew_csr is not None:
                adj, adj_t = build_weighted_adj(
                    mat, ew_csr, (args.user, args.item))
            else:
                adj = build_binary_adj(mat, (args.user, args.item))
                adj_t = build_binary_adj(mat.T, (args.item, args.user))
            self.sub_adj.append(adj)
            self.sub_adj_t.append(adj_t)

        actual_graphs = len(self.subMat)
        if args.graphNum > actual_graphs:
            print(f'Warning: graphNum={args.graphNum} > sub-graphs={actual_graphs}, clamping')
            args.graphNum = actual_graphs
        print(f'Sub-graphs: {args.graphNum}')

        # ---- Load node features (optional) ----
        self.user_features = None
        self.merchant_features = None
        if args.use_node_features:
            uf_path = self.predir + 'user_features.npy'
            mf_path = self.predir + 'merchant_features.npy'
            if os.path.isfile(uf_path) and os.path.isfile(mf_path):
                self.user_features = torch.FloatTensor(np.load(uf_path))
                self.merchant_features = torch.FloatTensor(np.load(mf_path))
                args.d_u = self.user_features.shape[1]
                args.d_v = self.merchant_features.shape[1]
                print(f'User features: {self.user_features.shape}, '
                      f'Merchant features: {self.merchant_features.shape}')
            else:
                print('WARNING: feature files not found, disabling node features')
                args.use_node_features = False

    # ------------------------------------------------------------------ #
    #  Sampling helpers (identical to selfGNN-Base)                       #
    # ------------------------------------------------------------------ #

    def neg_sample(self, label_row, samp_size, num_items, exclude):
        negs = []
        max_tries = num_items * 3
        tries = 0
        while len(negs) < samp_size and tries < max_tries:
            r = np.random.randint(num_items)
            if label_row[r] == 0 and r not in exclude:
                negs.append(r)
            tries += 1
        # Pad with last found item if sampling exhausted (very dense users)
        if negs:
            while len(negs) < samp_size:
                negs.append(negs[-1])
        else:
            negs = [np.random.randint(num_items)] * samp_size
        return negs

    def sample_train_batch(self, bat_ids):
        args = self.args
        label_mat = self.trnMat[bat_ids].toarray()
        batch = len(bat_ids)

        pos_u, pos_i, pos_seq = [], [], []
        neg_u, neg_i, neg_seq = [], [], []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            posset = list(self.sequence[uid])
            tst_item = self.tstInt[uid]
            samp_num = min(args.sampNum, len(posset))
            choose = 1

            if samp_num == 0:
                pos_items = [np.random.randint(args.item)]
                neg_items = [pos_items[0]]
                samp_num = 1
            else:
                choose = np.random.randint(
                    1, max(min(args.pred_num + 1, len(posset) - 3), 1) + 1)
                pos_items = [posset[-choose]] * samp_num
                exclude = set()
                last_item = self.sequence[uid][-1]
                exclude.add(last_item)
                if tst_item is not None:
                    exclude.add(int(tst_item))
                neg_items = self.neg_sample(
                    label_mat[i], samp_num, args.item, exclude)

            for j in range(samp_num):
                pos_u.append(uid);  pos_i.append(pos_items[j]);  pos_seq.append(i)
                neg_u.append(uid);  neg_i.append(neg_items[j]);  neg_seq.append(i)

            seq = posset[:-choose] if choose < len(posset) else posset
            if len(seq) == 0:
                seq = [0]
            if len(seq) <= args.pos_length:
                sequences[i, -len(seq):] = seq
                masks[i, -len(seq):] = 1.0
            else:
                sequences[i] = seq[-args.pos_length:]
                masks[i] = 1.0

        all_u = pos_u + neg_u
        all_i = pos_i + neg_i
        all_seq = pos_seq + neg_seq
        return (np.array(all_u), np.array(all_i),
                sequences, masks, np.array(all_seq))

    def sample_ssl_batch(self, bat_ids):
        args = self.args
        su_locs = [[] for _ in range(args.graphNum)]
        si_locs = [[] for _ in range(args.graphNum)]

        for k in range(args.graphNum):
            label = self.subMat[k][bat_ids].toarray()
            label_binary = (label != 0).astype(np.float32)
            for i, uid in enumerate(bat_ids):
                pos_items = np.where(label_binary[i] != 0)[0]
                ssl_num = min(args.sslNum, len(pos_items) // 2)
                if ssl_num == 0:
                    rand_item = np.random.randint(args.item)
                    su_locs[k].extend([uid, uid])
                    si_locs[k].extend([rand_item, rand_item])
                else:
                    chosen = np.random.choice(pos_items, ssl_num * 2, replace=False)
                    for j in range(ssl_num):
                        su_locs[k].extend([uid, uid])
                        si_locs[k].extend([int(chosen[j]), int(chosen[ssl_num + j])])

        return ([np.array(s) for s in su_locs],
                [np.array(s) for s in si_locs])

    def sample_eval_batch(self, bat_ids, mode='test'):
        args = self.args
        batch = len(bat_ids)

        if mode == 'val':
            eval_int = self.valInt
            eval_dict = self.val_dict
        else:
            eval_int = self.tstInt
            eval_dict = self.test_dict

        u_locs, i_locs, u_locs_seq = [], [], []
        tst_locs = []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            pos_item = eval_int[uid]

            neg_items = np.array(eval_dict.get(uid, [])[:args.testSize - 1])
            if len(neg_items) < args.testSize - 1:
                extra = args.testSize - 1 - len(neg_items)
                pad = np.random.randint(0, args.item, extra)
                neg_items = np.concatenate([neg_items, pad])
            loc_set = np.concatenate([neg_items, np.array([pos_item])])
            tst_locs.append(loc_set)

            for j in range(len(loc_set)):
                u_locs.append(uid)
                i_locs.append(int(loc_set[j]))
                u_locs_seq.append(i)

            posset = list(self.sequence[uid])
            if len(posset) <= args.pos_length:
                sequences[i, -len(posset):] = posset
                masks[i, -len(posset):] = 1.0
            else:
                sequences[i] = posset[-args.pos_length:]
                masks[i] = 1.0

        return (np.array(u_locs), np.array(i_locs),
                sequences, masks, np.array(u_locs_seq),
                tst_locs)
