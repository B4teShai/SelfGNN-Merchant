import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
import random

from config import args
from data_handler import DataHandler
from model import SelfGNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calc_metrics(preds, tst_item, tst_locs, k_list=[10, 20]):
    """Calculate HR@K and NDCG@K."""
    results = {f'HR@{k}': 0.0 for k in k_list}
    results.update({f'NDCG@{k}': 0.0 for k in k_list})
    num = preds.shape[0]

    for j in range(num):
        pred_vals = list(zip(preds[j], tst_locs[j]))
        pred_vals.sort(key=lambda x: x[0], reverse=True)

        for k in k_list:
            top_k = [x[1] for x in pred_vals[:k]]
            if tst_item[j] in top_k:
                results[f'HR@{k}'] += 1
                rank = top_k.index(tst_item[j])
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)

    for key in results:
        results[key] /= num
    return results


def train_epoch(model, handler, optimizer, device):
    model.train()
    num = args.user
    sf_ids = np.random.permutation(num)[:args.trnNum]
    num = len(sf_ids)
    steps = int(np.ceil(num / args.batch))

    epoch_loss = 0.0
    epoch_pre_loss = 0.0

    for i in range(steps):
        st = i * args.batch
        ed = min((i + 1) * args.batch, num)
        bat_ids = sf_ids[st:ed]

        # Sample training data
        uids, iids, sequences, masks, u_locs_seq = handler.sample_train_batch(bat_ids)
        su_locs, si_locs = handler.sample_ssl_batch(bat_ids)

        # To tensors
        uids = torch.LongTensor(uids).to(device)
        iids = torch.LongTensor(iids).to(device)
        sequences = torch.LongTensor(sequences).to(device)
        masks = torch.FloatTensor(masks).to(device)
        u_locs_seq = torch.LongTensor(u_locs_seq).to(device)
        su_locs_t = [torch.LongTensor(s).to(device) for s in su_locs]
        si_locs_t = [torch.LongTensor(s).to(device) for s in si_locs]

        # Forward
        preds, ssl_loss = model(
            uids, iids, sequences, masks, u_locs_seq,
            keep_rate=args.keepRate,
            su_locs=su_locs_t, si_locs=si_locs_t
        )

        # BPR margin loss
        samp_num = len(preds) // 2
        pos_pred = preds[:samp_num]
        neg_pred = preds[samp_num:]
        pre_loss = torch.clamp(1.0 - (pos_pred - neg_pred), min=0.0).mean()

        # Regularization
        reg_loss = args.reg * model.get_reg_loss()
        sal_loss = args.ssl_reg * ssl_loss

        loss = pre_loss + reg_loss + sal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_pre_loss += pre_loss.item()

        if (i + 1) % 5 == 0 or i == steps - 1:
            print(f'\r  Step {i+1}/{steps}: preLoss={pre_loss.item():.4f} '
                  f'regLoss={reg_loss.item():.4f} salLoss={sal_loss.item():.4f}',
                  end='', flush=True)

    print()
    return epoch_loss / steps, epoch_pre_loss / steps


@torch.no_grad()
def test_epoch(model, handler, device):
    model.eval()
    ids = handler.tstUsrs
    num = len(ids)
    tstBat = args.batch
    steps = int(np.ceil(num / tstBat))

    all_preds = []
    all_tst_items = []
    all_tst_locs = []

    for i in range(steps):
        st = i * tstBat
        ed = min((i + 1) * tstBat, num)
        bat_ids = ids[st:ed]
        batch_size = len(bat_ids)

        uids, iids, sequences, masks, u_locs_seq, tst_locs = \
            handler.sample_test_batch(bat_ids)

        tst_items = [handler.tstInt[uid] for uid in bat_ids]

        uids = torch.LongTensor(uids).to(device)
        iids = torch.LongTensor(iids).to(device)
        sequences = torch.LongTensor(sequences).to(device)
        masks = torch.FloatTensor(masks).to(device)
        u_locs_seq = torch.LongTensor(u_locs_seq).to(device)

        preds, _ = model(
            uids, iids, sequences, masks, u_locs_seq,
            keep_rate=1.0
        )

        preds_np = preds.cpu().numpy().reshape(batch_size, args.testSize)
        all_preds.append(preds_np)
        all_tst_items.extend(tst_items)
        all_tst_locs.extend(tst_locs)

        if (i + 1) % 10 == 0:
            print(f'\r  Test step {i+1}/{steps}', end='', flush=True)

    print()
    all_preds = np.concatenate(all_preds, axis=0)
    results = calc_metrics(all_preds, all_tst_items, all_tst_locs, k_list=[10, 20])
    return results


def main():
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.data}')

    # Load data
    handler = DataHandler(args)
    handler.load_data()

    # Move sparse adjacency to device
    sub_adj = [handler.sub_adj[k].to(device) for k in range(args.graphNum)]
    sub_adj_t = [handler.sub_adj_t[k].to(device) for k in range(args.graphNum)]

    # Create model
    model = SelfGNN(args, sub_adj, sub_adj_t).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    # Optimizer with lr decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)

    # Training
    best_ndcg = 0.0
    best_results = {}
    best_epoch = 0
    os.makedirs('History', exist_ok=True)
    os.makedirs('Models', exist_ok=True)

    for ep in range(args.epoch):
        t0 = time.time()

        # Train
        loss, pre_loss = train_epoch(model, handler, optimizer, device)
        scheduler.step()

        t1 = time.time()
        print(f'Epoch {ep}/{args.epoch} | Loss={loss:.4f} preLoss={pre_loss:.4f} '
              f'| time={t1-t0:.1f}s | lr={scheduler.get_last_lr()[0]:.6f}')

        # Test
        if ep % args.tstEpoch == 0:
            results = test_epoch(model, handler, device)
            result_str = ' | '.join([f'{k}={v:.4f}' for k, v in results.items()])
            print(f'  Test: {result_str}')

            if results['NDCG@10'] > best_ndcg:
                best_ndcg = results['NDCG@10']
                best_results = results.copy()
                best_epoch = ep
                torch.save(model.state_dict(),
                           f'Models/{args.save_path}.pt')
                print(f'  >>> New best! Saved model.')

        print()

    # Final test
    print('=' * 60)
    print('Loading best model for final evaluation...')
    model.load_state_dict(torch.load(f'Models/{args.save_path}.pt',
                                      map_location=device))
    final_results = test_epoch(model, handler, device)
    result_str = ' | '.join([f'{k}={v:.4f}' for k, v in final_results.items()])
    print(f'Final Test (from epoch {best_epoch}): {result_str}')
    print(f'Best during training (epoch {best_epoch}):')
    print(' | '.join([f'{k}={v:.4f}' for k, v in best_results.items()]))


if __name__ == '__main__':
    main()
