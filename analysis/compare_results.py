"""
Phase 3.1 — Comprehensive results comparison.

Usage:
    cd /path/to/SelfGNN-Merchant
    python analysis/compare_results.py

Requires:
    Results/config1_baseline.json
    Results/config2_edge.json
    Results/config3_node.json
    Results/config4_both.json
"""
import os
import json
import numpy as np

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(_root, 'Results')
FIGURES_DIR = os.path.join(_root, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

CONFIG_FILES = {
    'C1 Baseline': 'config1_baseline.json',
    'C2 Edge':     'config2_edge.json',
    'C3 Node':     'config3_node.json',
    'C4 Both':     'config4_both.json',
}
METRICS = ['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']


def load_results():
    results = {}
    missing = []
    for name, fname in CONFIG_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            results[name] = data
        else:
            missing.append(fname)
    if missing:
        print(f'WARNING: missing result files: {missing}')
    return results


def get_test_metrics(data):
    return data.get('test_results', data.get('final_test', {}))


def print_comparison_table(results):
    print('\n' + '=' * 80)
    print('Test Set Results Comparison')
    print('=' * 80)
    header = f"{'Config':<16}" + ''.join(f"{'  ' + m:<18}" for m in METRICS)
    print(header)
    print('-' * 80)

    baseline_metrics = None
    for name, data in results.items():
        test = get_test_metrics(data)
        row = f'{name:<16}'
        for m in METRICS:
            val = test.get(m, 0.0)
            if name == 'C1 Baseline' or baseline_metrics is None:
                row += f'  {val:.4f}          '
            else:
                delta = (val - baseline_metrics.get(m, 0)) / (baseline_metrics.get(m, 1e-8)) * 100
                sign = '+' if delta >= 0 else ''
                row += f'  {val:.4f} ({sign}{delta:.1f}%)'
        print(row)
        if baseline_metrics is None:
            baseline_metrics = {m: test.get(m, 0) for m in METRICS}
    print('=' * 80)


def check_additivity(results):
    if not all(k in results for k in ['C1 Baseline', 'C2 Edge', 'C3 Node', 'C4 Both']):
        return {}
    b = get_test_metrics(results['C1 Baseline'])
    e = get_test_metrics(results['C2 Edge'])
    n = get_test_metrics(results['C3 Node'])
    both = get_test_metrics(results['C4 Both'])

    print('\nAdditivity Check (C4 vs C1 + delta_C2 + delta_C3)')
    print('-' * 60)
    synergy = {}
    for m in METRICS:
        b_val = b.get(m, 0)
        e_delta = e.get(m, 0) - b_val
        n_delta = n.get(m, 0) - b_val
        expected = b_val + e_delta + n_delta
        actual = both.get(m, 0)
        diff = actual - expected
        synergy[m] = diff
        verdict = 'SYNERGISTIC' if diff > 0 else 'DIMINISHING'
        print(f'  {m}: expected={expected:.4f}, actual={actual:.4f}, '
              f'diff={diff:+.4f} → {verdict}')
    return synergy


def create_bar_chart(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Serif'
    except ImportError:
        print('matplotlib not available, skipping bar chart')
        return

    configs = list(results.keys())
    colors = ['#808080', '#FF8C00', '#1E90FF', '#2E8B57']

    fig, ax = plt.subplots(figsize=(7, 4))
    n_metrics = len(METRICS)
    n_configs = len(configs)
    x = np.arange(n_metrics)
    width = 0.18
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, (name, data) in enumerate(results.items()):
        test = get_test_metrics(data)
        vals = [test.get(m, 0) for m in METRICS]
        bars = ax.bar(x + offsets[i], vals, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6)

    # Y-axis floor
    all_vals = [get_test_metrics(d).get(m, 0) for d in results.values() for m in METRICS]
    if all_vals:
        ymin = max(0, min(all_vals) - 0.05)
        ax.set_ylim(bottom=ymin)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('SelfGNN Configuration Comparison', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'results_bar_chart.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def create_improvement_chart(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Serif'
    except ImportError:
        return

    if 'C1 Baseline' not in results:
        return

    b = get_test_metrics(results['C1 Baseline'])
    compare_configs = {k: v for k, v in results.items() if k != 'C1 Baseline'}
    if not compare_configs:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    n_metrics = len(METRICS)
    n_configs = len(compare_configs)
    x = np.arange(n_metrics)
    width = 0.22
    colors = ['#FF8C00', '#1E90FF', '#2E8B57']
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

    for i, (name, data) in enumerate(compare_configs.items()):
        test = get_test_metrics(data)
        improvements = [(test.get(m, 0) - b.get(m, 1e-8)) / b.get(m, 1e-8) * 100
                        for m in METRICS]
        bars = ax.bar(x + offsets[i], improvements, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.1 if val >= 0 else -0.4),
                    f'{val:+.1f}%', ha='center', va='bottom', fontsize=6)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=9)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=9)
    ax.set_title('Feature Impact Analysis', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'improvement_chart.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def create_convergence_plot(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams['font.family'] = 'DejaVu Serif'
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#808080', '#FF8C00', '#1E90FF', '#2E8B57']
    markers = ['o', 's', '^', 'D']

    for idx, (name, data) in enumerate(results.items()):
        history = data.get('train_history', [])
        if not history:
            continue
        epochs = [h['epoch'] for h in history]
        ndcg = [h.get('val_NDCG10', 0) for h in history]
        best_epoch = data.get('best_epoch', None)

        ax.plot(epochs, ndcg, label=name, color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)], markersize=3, linewidth=1.5)

        if best_epoch is not None:
            best_ndcg = next((h.get('val_NDCG10', 0) for h in history
                              if h['epoch'] == best_epoch), None)
            if best_ndcg is not None:
                ax.plot(best_epoch, best_ndcg, '*', color=colors[idx % len(colors)],
                        markersize=12, zorder=5)

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Val NDCG@10', fontsize=9)
    ax.set_title('Training Convergence', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'convergence.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def save_summary(results, synergy):
    summary = {}
    baseline_metrics = None
    for name, data in results.items():
        test = get_test_metrics(data)
        entry = {
            'best_epoch': data.get('best_epoch', 0),
            'val_results': data.get('val_results', data.get('best_val', {})),
            'test_results': test,
            'improvements': {},
        }
        if baseline_metrics is not None:
            for m in METRICS:
                b_val = baseline_metrics.get(m, 1e-8)
                entry['improvements'][m] = (test.get(m, 0) - b_val) / b_val * 100
        summary[name] = entry
        if baseline_metrics is None:
            baseline_metrics = test

    summary['additivity'] = synergy
    path = os.path.join(RESULTS_DIR, 'comparison_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Summary saved: {path}')


def main():
    results = load_results()
    if not results:
        print('No result files found. Run training first.')
        return

    print_comparison_table(results)
    synergy = check_additivity(results)

    print('\nGenerating figures...')
    create_bar_chart(results)
    create_improvement_chart(results)
    create_convergence_plot(results)

    save_summary(results, synergy)

    # Best config
    best_name = None
    best_ndcg = -1
    for name, data in results.items():
        test = get_test_metrics(data)
        ndcg = test.get('NDCG@10', 0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_name = name

    if best_name and 'C1 Baseline' in results:
        b_ndcg = get_test_metrics(results['C1 Baseline']).get('NDCG@10', 1e-8)
        b_hr = get_test_metrics(results['C1 Baseline']).get('HR@10', 1e-8)
        best_hr = get_test_metrics(results[best_name]).get('HR@10', 0)
        print(f'\nBest config: {best_name}')
        print(f'  NDCG@10 improvement over baseline: '
              f'{(best_ndcg - b_ndcg) / b_ndcg * 100:+.1f}%')
        print(f'  HR@10 improvement over baseline: '
              f'{(best_hr - b_hr) / b_hr * 100:+.1f}%')


if __name__ == '__main__':
    main()
