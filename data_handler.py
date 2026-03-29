import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import os
import torch


def build_sparse_adj(mat, shape):
    """Convert scipy sparse matrix to PyTorch sparse tensor (binary, values=1)."""
    coo = sp.coo_matrix(mat)
    row = torch.LongTensor(coo.row)
    col = torch.LongTensor(coo.col)
    idx = torch.stack([row, col])
    vals = torch.ones(len(coo.data), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()


class DataHandler:
    def __init__(self, args):
        self.args = args
        if args.data == 'yelp':
            predir = './Datasets/Yelp/'
        elif args.data == 'amazon':
            predir = './Datasets/amazon/'
        elif args.data == 'gowalla':
            predir = './Datasets/gowalla/'
        elif args.data == 'movielens':
            predir = './Datasets/movielens/'
        else:
            predir = './Datasets/' + args.data + '/'
        self.predir = predir

    def load_data(self):
        args = self.args

        # --- Load training data ---
        with open(self.predir + 'trn_mat_time', 'rb') as f:
            trnMat = pickle.load(f)

        # trnMat[0]: overall sparse matrix (user x item)
        # trnMat[1]: list of sparse matrices per time interval
        # trnMat[2]: time-interval index matrix
        args.user, args.item = trnMat[0].shape
        print(f'Users: {args.user}, Items: {args.item}')

        self.subMat = trnMat[1]  # list of scipy sparse matrices

        # --- Build overall binary train matrix from sequences ---
        with open(self.predir + 'sequence', 'rb') as f:
            self.sequence = pickle.load(f)

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

        # --- Test data ---
        with open(self.predir + 'tst_int', 'rb') as f:
            tstInt = pickle.load(f)
        self.tstInt = np.array(tstInt, dtype=object)

        tstStat = np.array([x is not None for x in tstInt])
        self.tstUsrs = np.where(tstStat)[0]
        print(f'Test users: {len(self.tstUsrs)}')

        if os.path.isfile(self.predir + 'test_dict'):
            with open(self.predir + 'test_dict', 'rb') as f:
                self.test_dict = pickle.load(f)

        # --- Build PyTorch sparse adjacency tensors ---
        self.sub_adj = []      # user->item for each time interval
        self.sub_adj_t = []    # item->user for each time interval
        for i in range(len(self.subMat)):
            mat = self.subMat[i]
            # Binary adjacency (ignore timestamp values)
            binary = (mat != 0).astype(np.float32)
            adj = build_sparse_adj(binary, (args.user, args.item))
            adj_t = build_sparse_adj(binary.T, (args.item, args.user))
            self.sub_adj.append(adj)
            self.sub_adj_t.append(adj_t)

        # Trim graphNum to actual number of sub-graphs
        actual_graphs = len(self.subMat)
        if args.graphNum > actual_graphs:
            print(f'Warning: graphNum={args.graphNum} > available sub-graphs={actual_graphs}, using {actual_graphs}')
            args.graphNum = actual_graphs

        print(f'Sub-graphs: {args.graphNum}')

    def neg_sample(self, label_row, samp_size, num_items, exclude):
        """Sample negative items not in label_row and not in exclude."""
        negs = []
        while len(negs) < samp_size:
            r = np.random.randint(num_items)
            if label_row[r] == 0 and r not in exclude:
                negs.append(r)
        return negs

    def sample_train_batch(self, bat_ids):
        """Sample positive and negative items for training.
        Returns arrays arranged as [all_pos, all_neg] (first half pos, second half neg).
        """
        args = self.args
        label_mat = self.trnMat[bat_ids].toarray()
        batch = len(bat_ids)
        train_sample_num = args.sampNum

        # Separate lists for pos and neg (matching original TF arrangement)
        pos_u, pos_i, pos_seq = [], [], []
        neg_u, neg_i, neg_seq = [], [], []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            posset = list(self.sequence[uid][:-1])
            tst_item = self.tstInt[uid]
            samp_num = min(train_sample_num, len(posset))

            if samp_num == 0:
                pos_items = [np.random.randint(args.item)]
                neg_items = [pos_items[0]]
                choose = 1
                samp_num = 1
            else:
                choose = np.random.randint(1, max(min(args.pred_num + 1, len(posset) - 3), 1) + 1)
                pos_items = [posset[-choose]] * samp_num
                exclude = set()
                last_item = self.sequence[uid][-1]
                exclude.add(last_item)
                if tst_item is not None:
                    exclude.add(int(tst_item))
                neg_items = self.neg_sample(label_mat[i], samp_num, args.item, exclude)

            for j in range(samp_num):
                pos_u.append(uid)
                pos_i.append(pos_items[j])
                pos_seq.append(i)
                neg_u.append(uid)
                neg_i.append(neg_items[j])
                neg_seq.append(i)

            # Build sequence (exclude the chosen target)
            seq = posset[:-choose] if choose < len(posset) else posset
            if len(seq) == 0:
                seq = [0]
            if len(seq) <= args.pos_length:
                sequences[i, -len(seq):] = seq
                masks[i, -len(seq):] = 1.0
            else:
                sequences[i] = seq[-args.pos_length:]
                masks[i] = 1.0

        # Concatenate: [all_pos, all_neg]
        all_u = pos_u + neg_u
        all_i = pos_i + neg_i
        all_seq = pos_seq + neg_seq

        return (np.array(all_u), np.array(all_i),
                sequences, masks, np.array(all_seq))

    def sample_ssl_batch(self, bat_ids):
        """Sample positive/negative pairs per sub-graph for SAL loss."""
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
                        si_locs[k].extend([chosen[j], chosen[ssl_num + j]])

        return ([np.array(s) for s in su_locs],
                [np.array(s) for s in si_locs])

    def sample_test_batch(self, bat_ids):
        """Sample test items (1 positive + testSize-1 negatives) per user."""
        args = self.args
        batch = len(bat_ids)

        u_locs, i_locs, u_locs_seq = [], [], []
        tst_locs = []
        sequences = np.zeros((args.batch, args.pos_length), dtype=np.int64)
        masks = np.zeros((args.batch, args.pos_length), dtype=np.float32)

        for i in range(batch):
            uid = bat_ids[i]
            pos_item = self.tstInt[uid]

            neg_items = np.array(self.test_dict[uid + 1][:args.testSize - 1]) - 1
            loc_set = np.concatenate([neg_items, np.array([pos_item])])
            tst_locs.append(loc_set)

            for j in range(len(loc_set)):
                u_locs.append(uid)
                i_locs.append(loc_set[j])
                u_locs_seq.append(i)

            posset = self.sequence[uid]
            if len(posset) <= args.pos_length:
                sequences[i, -len(posset):] = posset
                masks[i, -len(posset):] = 1.0
            else:
                sequences[i] = posset[-args.pos_length:]
                masks[i] = 1.0

        return (np.array(u_locs), np.array(i_locs),
                sequences, masks, np.array(u_locs_seq),
                tst_locs)
