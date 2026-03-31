import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.W_Q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = torch.exp(scores)
        attn = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return context


def edge_dropout(adj, keep_rate, training):
    if not training or keep_rate >= 1.0:
        return adj
    vals = adj.values()
    mask = (torch.rand_like(vals) < keep_rate).float()
    new_vals = vals * mask / keep_rate
    return torch.sparse_coo_tensor(adj.indices(), new_vals, adj.shape).coalesce()


class SelfGNN(nn.Module):
    def __init__(self, args, sub_adj_list, sub_adj_t_list,
                 user_features=None, merchant_features=None):
        super().__init__()
        self.args = args
        self.num_users = args.user
        self.num_items = args.item
        self.latdim = args.latdim
        self.graph_num = args.graphNum
        self.gnn_layers = args.gnn_layer
        self.att_layers = args.att_layer
        self.num_heads = args.num_attention_heads
        self.leaky = args.leaky
        self.use_node_features = args.use_node_features

        for k in range(self.graph_num):
            self.register_buffer(f'sub_adj_{k}', sub_adj_list[k])
            self.register_buffer(f'sub_adj_t_{k}', sub_adj_t_list[k])

        self.user_embeds = nn.Parameter(
            torch.empty(self.graph_num, self.num_users, self.latdim))
        self.item_embeds = nn.Parameter(
            torch.empty(self.graph_num, self.num_items, self.latdim))
        self.pos_embed = nn.Parameter(
            torch.empty(args.pos_length, self.latdim))
        nn.init.xavier_uniform_(self.user_embeds)
        nn.init.xavier_uniform_(self.item_embeds)
        nn.init.xavier_uniform_(self.pos_embed)

        self.user_lstm = nn.LSTM(self.latdim, self.latdim, batch_first=True)
        self.item_lstm = nn.LSTM(self.latdim, self.latdim, batch_first=True)

        self.user_mhsa = MultiHeadSelfAttention(self.latdim, self.num_heads)
        self.item_mhsa = MultiHeadSelfAttention(self.latdim, self.num_heads)

        self.seq_mhsa = nn.ModuleList([
            MultiHeadSelfAttention(self.latdim, self.num_heads)
            for _ in range(self.att_layers)
        ])

        self.sal_fc1 = nn.Linear(self.latdim * 3, args.ssldim)
        self.sal_fc2 = nn.Linear(args.ssldim, 1)

        self.ln_user = nn.LayerNorm(self.latdim)
        self.ln_item = nn.LayerNorm(self.latdim)
        self.ln_seq = nn.LayerNorm(self.latdim)
        self.ln_seq_pos = nn.LayerNorm(self.latdim)
        self.ln_seq_layers = nn.ModuleList([
            nn.LayerNorm(self.latdim) for _ in range(self.att_layers)
        ])

        # Node feature MLPs (optional)
        if self.use_node_features and user_features is not None:
            d_u = user_features.shape[1]
            d_v = merchant_features.shape[1]
            hidden = args.node_mlp_hidden
            self.user_mlp = nn.Sequential(
                nn.Linear(d_u, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.latdim)
            )
            self.merchant_mlp = nn.Sequential(
                nn.Linear(d_v, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.latdim)
            )
            # Xavier init to keep projection magnitudes comparable to embeddings
            for mlp in [self.user_mlp, self.merchant_mlp]:
                for layer in mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            self.register_buffer('user_feat', user_features)
            self.register_buffer('merchant_feat', merchant_features)
        else:
            self.use_node_features = False

    def _get_adj(self, k):
        return getattr(self, f'sub_adj_{k}')

    def _get_adj_t(self, k):
        return getattr(self, f'sub_adj_t_{k}')

    def leaky_relu(self, x):
        return torch.where(x > 0, x, self.leaky * x)

    def graph_encode(self, keep_rate):
        # Compute feature projections once if using node features
        f_u = None
        f_v = None
        if self.use_node_features:
            f_u = self.user_mlp(self.user_feat)       # (num_users, latdim)
            f_v = self.merchant_mlp(self.merchant_feat)  # (num_items, latdim)

        user_vectors, item_vectors = [], []
        for k in range(self.graph_num):
            adj = edge_dropout(self._get_adj(k), keep_rate, self.training)
            adj_t = edge_dropout(self._get_adj_t(k), keep_rate, self.training)

            # Layer 0: base embedding + optional feature fusion
            u_init = self.user_embeds[k]
            i_init = self.item_embeds[k]
            if self.use_node_features:
                u_init = u_init + f_u
                i_init = i_init + f_v

            u_embs = [u_init]
            i_embs = [i_init]
            for _ in range(self.gnn_layers):
                u_new = self.leaky_relu(torch.sparse.mm(adj, i_embs[-1]))
                i_new = self.leaky_relu(torch.sparse.mm(adj_t, u_embs[-1]))
                u_embs.append(u_new + u_embs[-1])
                i_embs.append(i_new + i_embs[-1])
            user_vectors.append(sum(u_embs))
            item_vectors.append(sum(i_embs))
        user_stack = torch.stack(user_vectors, dim=1)
        item_stack = torch.stack(item_vectors, dim=1)
        return user_stack, item_stack, user_vectors, item_vectors

    def temporal_encode(self, user_stack, item_stack, keep_rate):
        user_rnn, _ = self.user_lstm(user_stack)
        item_rnn, _ = self.item_lstm(item_stack)
        if self.training and keep_rate < 1.0:
            user_rnn = F.dropout(user_rnn, p=1.0 - keep_rate, training=True)
            item_rnn = F.dropout(item_rnn, p=1.0 - keep_rate, training=True)
        user_att = self.user_mhsa(self.ln_user(user_rnn))
        item_att = self.item_mhsa(self.ln_item(item_rnn))
        final_user = user_att.mean(dim=1)
        final_item = item_att.mean(dim=1)
        return final_user, final_item

    def sequence_encode(self, final_item, sequences, masks, keep_rate):
        B = sequences.shape[0]
        seq_emb = final_item[sequences]                          # (B, L, d)
        pos_emb = self.pos_embed.unsqueeze(0).expand(B, -1, -1) # (B, L, d)
        mask_exp = masks.unsqueeze(-1)                           # (B, L, 1)
        # Normalize per-position and zero out padding
        att = (self.ln_seq(seq_emb) + self.ln_seq_pos(pos_emb)) * mask_exp
        # Multi-head self-attention over the full sequence, re-mask after each layer
        for i in range(self.att_layers):
            att_new = self.seq_mhsa[i](self.ln_seq_layers[i](att))
            att = (self.leaky_relu(att_new) + att) * mask_exp
        # Masked sum pooling -> (B, d)
        return att.sum(dim=1)

    def forward(self, uids, iids, sequences, masks, u_locs_seq, keep_rate,
                su_locs=None, si_locs=None):
        user_stack, item_stack, user_vecs, item_vecs = self.graph_encode(keep_rate)
        final_user, final_item = self.temporal_encode(user_stack, item_stack, keep_rate)
        att_user = self.sequence_encode(final_item, sequences, masks, keep_rate)

        u_emb = final_user[uids]
        i_emb = final_item[iids]
        preds = (u_emb * i_emb).sum(dim=-1)
        att_u = att_user[u_locs_seq]
        i_emb_att = final_item[iids]
        preds = preds + (self.leaky_relu(att_u) * i_emb_att).sum(dim=-1)

        ssl_loss = torch.tensor(0.0, device=preds.device)
        if su_locs is not None and si_locs is not None:
            ssl_loss = self.compute_sal_loss(
                final_user, final_item, user_vecs, item_vecs,
                su_locs, si_locs)
        return preds, ssl_loss

    def compute_sal_loss(self, final_user, final_item, user_vecs, item_vecs,
                         su_locs, si_locs):
        ssl_loss = torch.tensor(0.0, device=final_user.device)
        for k in range(self.graph_num):
            su = su_locs[k]
            si = si_locs[k]
            if len(su) < 2:
                continue
            uv_short = user_vecs[k]
            meta_input = torch.cat([
                final_user * uv_short, final_user, uv_short], dim=-1)
            weights = torch.sigmoid(
                self.sal_fc2(self.leaky_relu(self.sal_fc1(meta_input)))
            ).squeeze(-1)
            samp_num = len(su) // 2
            u_long = final_user[su]
            i_long = final_item[si]
            s_long = (self.leaky_relu(u_long * i_long)).sum(dim=-1)
            pos_long = s_long[:samp_num].detach()
            neg_long = s_long[samp_num:].detach()
            w_pos = weights[su[:samp_num]]
            w_neg = weights[su[samp_num:]]
            s_final = w_pos * pos_long - w_neg * neg_long
            u_short = uv_short[su]
            i_short = item_vecs[k][si]
            s_short = (self.leaky_relu(u_short * i_short)).sum(dim=-1)
            pos_short = s_short[:samp_num]
            neg_short = s_short[samp_num:]
            ssl_loss = ssl_loss + torch.clamp(
                1.0 - s_final * (pos_short - neg_short), min=0.0).sum()
        return ssl_loss

    def get_reg_loss(self):
        loss = (self.user_embeds.norm(2).pow(2) +
                self.item_embeds.norm(2).pow(2) +
                self.pos_embed.norm(2).pow(2))
        if self.use_node_features:
            for p in self.user_mlp.parameters():
                loss = loss + p.norm(2).pow(2)
            for p in self.merchant_mlp.parameters():
                loss = loss + p.norm(2).pow(2)
        return loss
