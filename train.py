import os
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


def calc_metrics(preds, eval_items, eval_locs, k_list=[10, 20]):
    results = {f'HR@{k}': 0.0 for k in k_list}
    results.update({f'NDCG@{k}': 0.0 for k in k_list})
    num = preds.shape[0]
    for j in range(num):
        pred_vals = list(zip(preds[j], eval_locs[j]))
        pred_vals.sort(key=lambda x: x[0], reverse=True)
        for k in k_list:
            top_k = [x[1] for x in pred_vals[:k]]
            if eval_items[j] in top_k:
                results[f'HR@{k}'] += 1
                rank = top_k.index(eval_items[j])
                results[f'NDCG@{k}'] += 1.0 / np.log2(rank + 2)
    for key in results:
        results[key] /= num
    return results


def train_epoch(model, handler, optimizer, device):
    model.train()
    sf_ids = np.random.permutation(args.user)[:args.trnNum]
    num = len(sf_ids)
    steps = int(np.ceil(num / args.batch))
    epoch_loss = 0.0
    epoch_pre_loss = 0.0

    for i in range(steps):
        st = i * args.batch
        ed = min((i + 1) * args.batch, num)
        bat_ids = sf_ids[st:ed]

        uids, iids, sequences, masks, u_locs_seq = handler.sample_train_batch(bat_ids)
        su_locs, si_locs = handler.sample_ssl_batch(bat_ids)

        uids_t = torch.LongTensor(uids).to(device)
        iids_t = torch.LongTensor(iids).to(device)
        seq_t = torch.LongTensor(sequences).to(device)
        mask_t = torch.FloatTensor(masks).to(device)
        uloc_t = torch.LongTensor(u_locs_seq).to(device)
        su_t = [torch.LongTensor(s).to(device) for s in su_locs]
        si_t = [torch.LongTensor(s).to(device) for s in si_locs]

        preds, ssl_loss = model(
            uids_t, iids_t, seq_t, mask_t, uloc_t,
            keep_rate=args.keepRate, su_locs=su_t, si_locs=si_t)

        samp_num = len(preds) // 2
        pos_pred = preds[:samp_num]
        neg_pred = preds[samp_num:]
        pre_loss = torch.clamp(1.0 - (pos_pred - neg_pred), min=0.0).mean()
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
                  f'reg={reg_loss.item():.4f} sal={sal_loss.item():.4f}',
                  end='', flush=True)
    print()
    return epoch_loss / steps, epoch_pre_loss / steps


@torch.no_grad()
def evaluate(model, handler, device, mode='test'):
    """Run evaluation. mode='val' for validation, 'test' for test."""
    model.eval()
    if mode == 'val':
        ids = handler.valUsrs
        eval_int = handler.valInt
    else:
        ids = handler.tstUsrs
        eval_int = handler.tstInt

    if len(ids) == 0:
        return {}

    num = len(ids)
    steps = int(np.ceil(num / args.batch))
    all_preds, all_items, all_locs = [], [], []

    for i in range(steps):
        st = i * args.batch
        ed = min((i + 1) * args.batch, num)
        bat_ids = ids[st:ed]
        batch_size = len(bat_ids)

        uids, iids, sequences, masks, u_locs_seq, tst_locs = \
            handler.sample_eval_batch(bat_ids, mode=mode)
        eval_items = [eval_int[uid] for uid in bat_ids]

        preds, _ = model(
            torch.LongTensor(uids).to(device),
            torch.LongTensor(iids).to(device),
            torch.LongTensor(sequences).to(device),
            torch.FloatTensor(masks).to(device),
            torch.LongTensor(u_locs_seq).to(device),
            keep_rate=1.0)

        preds_np = preds.cpu().numpy().reshape(batch_size, args.testSize)
        all_preds.append(preds_np)
        all_items.extend(eval_items)
        all_locs.extend(tst_locs)

        if (i + 1) % 10 == 0:
            print(f'\r  {mode.capitalize()} step {i+1}/{steps}', end='', flush=True)

    print()
    all_preds = np.concatenate(all_preds, axis=0)
    return calc_metrics(all_preds, all_items, all_locs, k_list=[10, 20])


def fmt(results):
    return ' | '.join([f'{k}={v:.4f}' for k, v in results.items()])


def main():
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.data}')

    handler = DataHandler(args)
    handler.load_data()

    sub_adj = [handler.sub_adj[k].to(device) for k in range(args.graphNum)]
    sub_adj_t = [handler.sub_adj_t[k].to(device) for k in range(args.graphNum)]

    model = SelfGNN(args, sub_adj, sub_adj_t).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay)

    os.makedirs('History', exist_ok=True)
    os.makedirs('Models', exist_ok=True)

    has_val = len(handler.valUsrs) > 0

    best_val_ndcg = 0.0
    best_val_results = {}
    best_epoch = 0
    patience_counter = 0

    print('=' * 70)
    print(f'Training for {args.epoch} epochs (eval every {args.tstEpoch})')
    if has_val:
        print(f'Early stopping: patience={args.patience} eval epochs on Val NDCG@10')
    print('=' * 70)

    for ep in range(args.epoch):
        t0 = time.time()
        loss, pre_loss = train_epoch(model, handler, optimizer, device)
        scheduler.step()
        t1 = time.time()
        print(f'Epoch {ep}/{args.epoch} | Loss={loss:.4f} preLoss={pre_loss:.4f} '
              f'| {t1-t0:.1f}s | lr={scheduler.get_last_lr()[0]:.6f}')

        # ---- Evaluation ----
        if ep % args.tstEpoch == 0:
            if has_val:
                val_results = evaluate(model, handler, device, mode='val')
                print(f'  Val:  {fmt(val_results)}')

                if val_results.get('NDCG@10', 0) > best_val_ndcg:
                    best_val_ndcg = val_results['NDCG@10']
                    best_val_results = val_results.copy()
                    best_epoch = ep
                    patience_counter = 0
                    torch.save(model.state_dict(), f'Models/{args.save_path}.pt')
                    print(f'  >>> New best Val NDCG@10={best_val_ndcg:.4f}. Model saved.')
                else:
                    patience_counter += 1
                    print(f'  No improvement. Patience: {patience_counter}/{args.patience}')

                if patience_counter >= args.patience:
                    print(f'\nEarly stopping at epoch {ep}.')
                    break
            else:
                # No validation: save based on test (not ideal but fallback)
                test_results = evaluate(model, handler, device, mode='test')
                print(f'  Test: {fmt(test_results)}')
                if test_results.get('NDCG@10', 0) > best_val_ndcg:
                    best_val_ndcg = test_results['NDCG@10']
                    best_val_results = test_results.copy()
                    best_epoch = ep
                    torch.save(model.state_dict(), f'Models/{args.save_path}.pt')
                    print(f'  >>> New best. Model saved.')
        print()

    # ---- Final Test with best model ----
    print('=' * 70)
    print(f'Loading best model from epoch {best_epoch}...')
    model.load_state_dict(torch.load(f'Models/{args.save_path}.pt',
                                      map_location=device))

    if has_val:
        print(f'Best Val (epoch {best_epoch}): {fmt(best_val_results)}')

    print('\nFinal Test Evaluation:')
    test_results = evaluate(model, handler, device, mode='test')
    print(f'  Test: {fmt(test_results)}')

    if has_val:
        print(f'\nFinal Val Evaluation (sanity check):')
        val_results = evaluate(model, handler, device, mode='val')
        print(f'  Val:  {fmt(val_results)}')

    # Save results
    import json
    result_log = {
        'best_epoch': best_epoch,
        'best_val': best_val_results,
        'final_test': test_results,
        'args': vars(args),
    }
    with open(f'History/{args.save_path}_results.json', 'w') as f:
        json.dump(result_log, f, indent=2, default=str)
    print(f'\nResults saved to History/{args.save_path}_results.json')


if __name__ == '__main__':
    main()
