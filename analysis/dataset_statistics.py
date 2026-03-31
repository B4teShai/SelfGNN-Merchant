"""
Phase 0.2 — Dataset statistics for the paper.

Usage:
    cd /path/to/SelfGNN-Merchant
    python analysis/dataset_statistics.py
"""
import os
import json
import pickle
import numpy as np

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(_root, 'Datasets', 'yelp-merchant')
RESULTS_DIR = os.path.join(_root, 'Results')
FIGURES_DIR = os.path.join(_root, 'paper_figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print('Loading preprocessed data...')
    with open(os.path.join(DATA_DIR, 'trn_mat_time'), 'rb') as f:
        trnMat = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'sequence'), 'rb') as f:
        sequence = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'tst_int'), 'rb') as f:
        tstInt = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'val_int'), 'rb') as f:
        valInt = pickle.load(f)

    global_mat = trnMat[0]
    subMats = trnMat[1]
    num_users, num_merchants = global_mat.shape
    time_intervals = len(subMats)

    total_train_events = int(global_mat.sum())
    unique_pairs = global_mat.nnz
    density = unique_pairs / (num_users * num_merchants) * 100

    # User degree: unique merchants per user (CSR row counts)
    user_degrees = np.diff(global_mat.indptr)
    # Merchant degree: unique users per merchant (CSC column counts)
    global_mat_csc = global_mat.tocsc()
    merchant_deg = np.diff(global_mat_csc.indptr).astype(float)

    def stats_dict(arr):
        arr = arr[arr > 0]
        return {
            'min': int(arr.min()), 'max': int(arr.max()),
            'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'std': float(arr.std()),
        }

    # Sequence lengths
    seq_lengths = np.array([len(s) for s in sequence])

    # Per-interval edge counts
    per_interval = [int(m.nnz) for m in subMats]

    # Test / val user counts
    test_users = int(sum(x is not None for x in tstInt))
    val_users = int(sum(x is not None for x in valInt))

    stats = {
        'users': num_users,
        'merchants': num_merchants,
        'total_train_events': total_train_events,
        'unique_pairs': unique_pairs,
        'density': density,
        'user_degree': stats_dict(user_degrees),
        'merchant_degree': stats_dict(merchant_deg),
        'sequence_length': stats_dict(seq_lengths),
        'per_interval_edges': per_interval,
        'test_users': test_users,
        'val_users': val_users,
        'time_intervals': time_intervals,
    }

    out_path = os.path.join(RESULTS_DIR, 'dataset_statistics.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved: {out_path}')

    # Print summary
    print('\n' + '=' * 50)
    print('Yelp-Merchant Dataset Statistics')
    print('=' * 50)
    print(f'{"Users:":<30} {num_users:>12,}')
    print(f'{"Merchants:":<30} {num_merchants:>12,}')
    print(f'{"Train events:":<30} {total_train_events:>12,}')
    print(f'{"Unique pairs:":<30} {unique_pairs:>12,}')
    print(f'{"Density:":<30} {density:>12.4f}%')
    print(f'{"Test users:":<30} {test_users:>12,}')
    print(f'{"Val users:":<30} {val_users:>12,}')
    print(f'{"Time intervals:":<30} {time_intervals:>12}')
    print(f'{"Per-interval edges:":<30} {per_interval}')
    ud = stats['user_degree']
    md = stats['merchant_degree']
    sl = stats['sequence_length']
    print(f'\nUser degree  — mean={ud["mean"]:.1f}, median={ud["median"]:.0f}, '
          f'min={ud["min"]}, max={ud["max"]}, std={ud["std"]:.1f}')
    print(f'Merchant deg — mean={md["mean"]:.1f}, median={md["median"]:.0f}, '
          f'min={md["min"]}, max={md["max"]}, std={md["std"]:.1f}')
    print(f'Seq length   — mean={sl["mean"]:.1f}, median={sl["median"]:.0f}, '
          f'min={sl["min"]}, max={sl["max"]}, std={sl["std"]:.1f}')

    # Distribution figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Serif'

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        ud_vals = user_degrees[user_degrees > 0]
        md_vals = merchant_deg[merchant_deg > 0].astype(int)

        axes[0].hist(ud_vals, bins=50, color='#1976D2', alpha=0.8)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Degree', fontsize=9)
        axes[0].set_ylabel('Count', fontsize=9)
        axes[0].set_title('User Degree Distribution', fontsize=9)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].hist(md_vals, bins=50, color='#F57C00', alpha=0.8)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Degree', fontsize=9)
        axes[1].set_ylabel('Count', fontsize=9)
        axes[1].set_title('Merchant Degree Distribution', fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, 'dataset_distributions.pdf')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'\nFigure saved: {fig_path}')
    except ImportError:
        print('matplotlib not available, skipping figure')


if __name__ == '__main__':
    main()
