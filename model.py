import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention matching the original TF implementation."""
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
        """x: [batch, seq_len, d_model] -> [batch, seq_len, d_model]"""
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
    """Drop edges from sparse adjacency tensor during training."""
    if not training or keep_rate >= 1.0:
        return adj
    vals = adj.values()
    mask = (torch.rand_like(vals) < keep_rate).float()
    new_vals = vals * mask / keep_rate  # inverted dropout
    return torch.sparse_coo_tensor(adj.indices(), new_vals, adj.shape).coalesce()


class SelfGNN(nn.Module):
    def __init__(self, args, sub_adj_list, sub_adj_t_list):
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

        # Register sparse adjacency matrices as buffers
        for k in range(self.graph_num):
            self.register_buffer(f'sub_adj_{k}', sub_adj_list[k])
            self.register_buffer(f'sub_adj_t_{k}', sub_adj_t_list[k])

        # ---------- Embeddings ----------
        # Per-graph learnable embeddings (matching original uEmbed/iEmbed)
        self.user_embeds = nn.Parameter(
            torch.empty(self.graph_num, self.num_users, self.latdim))
        self.item_embeds = nn.Parameter(
            torch.empty(self.graph_num, self.num_items, self.latdim))
        self.pos_embed = nn.Parameter(
            torch.empty(args.pos_length, self.latdim))

        nn.init.xavier_uniform_(self.user_embeds)
        nn.init.xavier_uniform_(self.item_embeds)
        nn.init.xavier_uniform_(self.pos_embed)

        # ---------- Interval-level LSTM ----------
        self.user_lstm = nn.LSTM(self.latdim, self.latdim, batch_first=True)
        self.item_lstm = nn.LSTM(self.latdim, self.latdim, batch_first=True)

        # ---------- Interval-level multi-head self-attention ----------
        self.user_mhsa = MultiHeadSelfAttention(self.latdim, self.num_heads)
        self.item_mhsa = MultiHeadSelfAttention(self.latdim, self.num_heads)

        # ---------- Instance-level sequence self-attention ----------
        self.seq_mhsa = nn.ModuleList([
            MultiHeadSelfAttention(self.latdim, self.num_heads)
            for _ in range(self.att_layers)
        ])

        # ---------- SAL weight network ----------
        self.sal_fc1 = nn.Linear(self.latdim * 3, args.ssldim)
        self.sal_fc2 = nn.Linear(args.ssldim, 1)

        # Layer norms (matching original tf.contrib.layers.layer_norm)
        self.ln_user = nn.LayerNorm(self.latdim)
        self.ln_item = nn.LayerNorm(self.latdim)
        self.ln_seq = nn.LayerNorm(self.latdim)
        self.ln_seq_pos = nn.LayerNorm(self.latdim)
        self.ln_seq_layers = nn.ModuleList([
            nn.LayerNorm(self.latdim) for _ in range(self.att_layers)
        ])

    def _get_adj(self, k):
        return getattr(self, f'sub_adj_{k}')

    def _get_adj_t(self, k):
        return getattr(self, f'sub_adj_t_{k}')

    def leaky_relu(self, x):
        return torch.where(x > 0, x, self.leaky * x)

    def graph_encode(self, keep_rate):
        """Short-term graph encoding for all time intervals."""
        user_vectors = []
        item_vectors = []

        for k in range(self.graph_num):
            adj = edge_dropout(self._get_adj(k), keep_rate, self.training)
            adj_t = edge_dropout(self._get_adj_t(k), keep_rate, self.training)

            u_embs = [self.user_embeds[k]]  # layer 0
            i_embs = [self.item_embeds[k]]  # layer 0

            for _ in range(self.gnn_layers):
                # user <- aggregate item neighbors
                u_new = self.leaky_relu(torch.sparse.mm(adj, i_embs[-1]))
                # item <- aggregate user neighbors
                i_new = self.leaky_relu(torch.sparse.mm(adj_t, u_embs[-1]))
                # residual connection
                u_embs.append(u_new + u_embs[-1])
                i_embs.append(i_new + i_embs[-1])

            # Sum all layers (LightGCN style)
            user_vec = sum(u_embs)  # [num_users, latdim]
            item_vec = sum(i_embs)  # [num_items, latdim]
            user_vectors.append(user_vec)
            item_vectors.append(item_vec)

        # Stack: [num_users, graphNum, latdim]
        user_stack = torch.stack(user_vectors, dim=1)
        item_stack = torch.stack(item_vectors, dim=1)

        return user_stack, item_stack, user_vectors, item_vectors

    def temporal_encode(self, user_stack, item_stack, keep_rate):
        """Long-term temporal modeling: LSTM + self-attention over intervals."""
        # LSTM
        user_rnn, _ = self.user_lstm(user_stack)
        item_rnn, _ = self.item_lstm(item_stack)

        if self.training and keep_rate < 1.0:
            user_rnn = F.dropout(user_rnn, p=1.0 - keep_rate, training=True)
            item_rnn = F.dropout(item_rnn, p=1.0 - keep_rate, training=True)

        # Multi-head self-attention
        user_att = self.user_mhsa(self.ln_user(user_rnn))
        item_att = self.item_mhsa(self.ln_item(item_rnn))

        # Mean pooling over time intervals
        final_user = user_att.mean(dim=1)  # [num_users, latdim]
        final_item = item_att.mean(dim=1)  # [num_items, latdim]

        return final_user, final_item

    def sequence_encode(self, final_item, sequences, masks, keep_rate):
        """Instance-level sequence modeling with self-attention."""
        # sequences: [batch, pos_length] (item indices)
        # masks: [batch, pos_length]
        B = sequences.shape[0]

        # Look up item embeddings from final_item
        seq_emb = final_item[sequences]  # [batch, pos_length, latdim]
        pos_emb = self.pos_embed.unsqueeze(0).expand(B, -1, -1)

        # Apply mask: weighted combination
        mask_3d = masks.unsqueeze(1)  # [batch, 1, pos_length]
        seq_out = self.ln_seq(torch.bmm(mask_3d, seq_emb))  # [batch, 1, latdim]
        seq_out = seq_out + self.ln_seq_pos(torch.bmm(mask_3d, pos_emb))

        # Self-attention layers with residual
        att = seq_out
        for i in range(self.att_layers):
            att_new = self.seq_mhsa[i](self.ln_seq_layers[i](att))
            att = self.leaky_relu(att_new) + att

        att_user = att.squeeze(1)  # [batch, latdim]
        return att_user

    def forward(self, uids, iids, sequences, masks, u_locs_seq, keep_rate,
                su_locs=None, si_locs=None):
        """
        Full forward pass.

        Args:
            uids: user indices [N]
            iids: item indices [N]
            sequences: [batch, pos_length]
            masks: [batch, pos_length]
            u_locs_seq: maps each uid in uids to its position in sequences [N]
            keep_rate: edge keep rate for dropout
            su_locs: list of user arrays for SAL, one per graph
            si_locs: list of item arrays for SAL, one per graph
        """
        # 1. Graph encoding
        user_stack, item_stack, user_vecs, item_vecs = self.graph_encode(keep_rate)

        # 2. Temporal encoding
        final_user, final_item = self.temporal_encode(user_stack, item_stack, keep_rate)

        # 3. Sequence encoding
        att_user = self.sequence_encode(final_item, sequences, masks, keep_rate)

        # 4. Prediction
        u_emb = final_user[uids]
        i_emb = final_item[iids]
        preds = (u_emb * i_emb).sum(dim=-1)

        # Add sequence-level prediction
        att_u = att_user[u_locs_seq]
        i_emb_att = final_item[iids]
        preds = preds + (self.leaky_relu(att_u) * i_emb_att).sum(dim=-1)

        # 5. SAL loss
        ssl_loss = torch.tensor(0.0, device=preds.device)
        if su_locs is not None and si_locs is not None:
            ssl_loss = self.compute_sal_loss(
                final_user, final_item, user_vecs, item_vecs,
                su_locs, si_locs
            )

        return preds, ssl_loss

    def compute_sal_loss(self, final_user, final_item, user_vecs, item_vecs,
                         su_locs, si_locs):
        """Self-Augmented Learning loss."""
        ssl_loss = torch.tensor(0.0, device=final_user.device)

        for k in range(self.graph_num):
            su = su_locs[k]  # user indices
            si = si_locs[k]  # item indices
            if len(su) < 2:
                continue

            # Compute personalized weights
            uv_short = user_vecs[k]  # [num_users, latdim]
            meta_input = torch.cat([
                final_user * uv_short,
                final_user,
                uv_short
            ], dim=-1)  # [num_users, 3*latdim]
            weights = torch.sigmoid(
                self.sal_fc2(self.leaky_relu(self.sal_fc1(meta_input)))
            ).squeeze(-1)  # [num_users]

            samp_num = len(su) // 2

            # Long-term view scores
            u_long = final_user[su]
            i_long = final_item[si]
            s_long = (self.leaky_relu(u_long * i_long)).sum(dim=-1)
            pos_long = s_long[:samp_num].detach()
            neg_long = s_long[samp_num:].detach()

            w_pos = weights[su[:samp_num]]
            w_neg = weights[su[samp_num:]]
            s_final = w_pos * pos_long - w_neg * neg_long

            # Short-term view scores
            u_short = uv_short[su]
            i_short = item_vecs[k][si]
            s_short = (self.leaky_relu(u_short * i_short)).sum(dim=-1)
            pos_short = s_short[:samp_num]
            neg_short = s_short[samp_num:]

            ssl_loss = ssl_loss + torch.clamp(
                1.0 - s_final * (pos_short - neg_short), min=0.0
            ).sum()

        return ssl_loss

    def get_reg_loss(self):
        """L2 regularization on embedding parameters."""
        reg = (self.user_embeds.norm(2).pow(2) +
               self.item_embeds.norm(2).pow(2) +
               self.pos_embed.norm(2).pow(2))
        return reg
