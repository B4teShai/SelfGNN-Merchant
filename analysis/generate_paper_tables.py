"""
Phase 3.2 — Generate LaTeX tables for the paper.

Usage:
    cd /path/to/SelfGNN-Merchant
    python analysis/generate_paper_tables.py

Requires:
    Results/comparison_summary.json
    Results/dataset_statistics.json  (optional, uses defaults if missing)
"""
import os
import json

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(_root, 'Results')
FIGURES_DIR = os.path.join(_root, 'paper_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS = ['HR@10', 'NDCG@10', 'HR@20', 'NDCG@20']
CONFIG_NAMES = ['C1 Baseline', 'C2 Edge', 'C3 Node', 'C4 Both']


def load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def best_in_col(summary, metric):
    best = -1
    for name in CONFIG_NAMES:
        if name in summary:
            val = summary[name].get('test_results', {}).get(metric, 0)
            if val > best:
                best = val
    return best


def table_dataset(stats):
    users = stats.get('users', 'N/A')
    merchants = stats.get('merchants', 'N/A')
    train_events = stats.get('total_train_events', 'N/A')
    density = stats.get('density', 'N/A')
    ud = stats.get('user_degree', {})
    md = stats.get('merchant_degree', {})
    test_users = stats.get('test_users', 'N/A')
    val_users = stats.get('val_users', 'N/A')
    intervals = stats.get('time_intervals', 5)

    def fmt(v, d=4):
        if isinstance(v, float):
            return f'{v:.{d}f}'
        return str(v)

    rows = [
        ('Хэрэглэгч (Users)', fmt(users, 0) if isinstance(users, (int, float)) else users),
        ('Борлуулагч (Merchants)', fmt(merchants, 0) if isinstance(merchants, (int, float)) else merchants),
        ('Сургалтын харилцан үйлдэл (Train interactions)',
         f'{train_events:,}' if isinstance(train_events, (int, float)) else train_events),
        ('Нягтрал (Density)', f'{density:.4f}%' if isinstance(density, float) else density),
        ('Хэрэглэгч/у дундаж зэрэг (Avg merchants/user)',
         fmt(ud.get('mean', 'N/A'))),
        ('Борлуулагч/у дундаж зэрэг (Avg users/merchant)',
         fmt(md.get('mean', 'N/A'))),
        ('Тест хэрэглэгч (Test users)', str(test_users)),
        ('Баталгаажуулалт хэрэглэгч (Val users)', str(val_users)),
        ('Цагийн интервал (T)', str(intervals)),
    ]

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Yelp-Merchant өгөгдлийн олонлогийн статистик}',
        r'\label{tab:dataset}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Үзүүлэлт} & \textbf{Утга} \\',
        r'\midrule',
    ]
    for label, val in rows:
        lines.append(f'{label} & {val} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_configs():
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Туршилтын 4 тохиргоо}',
        r'\label{tab:configs}',
        r'\begin{tabular}{llll}',
        r'\toprule',
        r'\textbf{Тохиргоо} & \textbf{Граф} & \textbf{Оройн шинж} & \textbf{Ирмэгийн шинж} \\',
        r'\midrule',
        r'C1 Baseline & Хоёртын & --- & --- \\',
        r'C2 Edge & Жинт & --- & Үнэлгээ (log-sigmoid + нормчлол) \\',
        r'C3 Node & Хоёртын & MLP проекц (нэмэх) & --- \\',
        r'C4 Both & Жинт & MLP проекц (нэмэх) & Үнэлгээ (log-sigmoid + нормчлол) \\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def table_results(summary):
    best = {m: best_in_col(summary, m) for m in METRICS}

    def fmt_cell(name, metric):
        if name not in summary:
            return '---'
        val = summary[name].get('test_results', {}).get(metric, 0)
        imp = summary[name].get('improvements', {}).get(metric, None)
        bold = (abs(val - best[metric]) < 1e-6)
        cell = f'{val:.4f}'
        if bold:
            cell = r'\textbf{' + cell + '}'
        if imp is not None and name != 'C1 Baseline':
            sign = '+' if imp >= 0 else ''
            cell += f' ($\\uparrow${sign}{imp:.1f}\\%)'
        return cell

    header_cols = ' & '.join([r'\textbf{' + m + '}' for m in METRICS])
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{4 тохиргооны туршилтын үр дүнгийн харьцуулалт}',
        r'\label{tab:results}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Тохиргоо} & ' + header_cols + r' \\',
        r'\midrule',
    ]
    display = {
        'C1 Baseline': 'C1 Baseline',
        'C2 Edge': 'C2 Edge',
        'C3 Node': 'C3 Node',
        'C4 Both': 'C4 Both',
    }
    for name, label in display.items():
        cells = ' & '.join([fmt_cell(name, m) for m in METRICS])
        lines.append(f'{label} & {cells} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def table_hyperparams():
    rows = [
        ('Embedding dim ($d$)', '64'),
        ('GNN давхарга ($L$)', '3'),
        ('Анхаарлын давхарга', '2'),
        ('Batch хэмжээ', '512'),
        ('Сургалтын хурд ($\\eta$)', '$10^{-3}$'),
        ('L2 регуляризаци ($\\lambda$)', '$10^{-2}$'),
        ('SAL жин ($\\lambda_{ssl}$)', '$10^{-7}$'),
        ('Ирмэгийн dropout хурд', '0.5'),
        ('LSTM далд хэмжээ', '64'),
        ('Анхаарлын толгой', '16'),
        ('Дарааллын урт', '200'),
        ('Цагийн интервал ($T$)', '5'),
    ]
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{Сургалтын гиперпараметрүүд}',
        r'\label{tab:hyperparams}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        r'\textbf{Параметр} & \textbf{Утга} \\',
        r'\midrule',
    ]
    for param, val in rows:
        lines.append(f'{param} & {val} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    summary = load_json(os.path.join(RESULTS_DIR, 'comparison_summary.json'))
    stats = load_json(os.path.join(RESULTS_DIR, 'dataset_statistics.json'))

    tables = {
        'table_dataset.tex': table_dataset(stats),
        'table_configs.tex': table_configs(),
        'table_results.tex': table_results(summary),
        'table_hyperparams.tex': table_hyperparams(),
    }

    for fname, tex in tables.items():
        path = os.path.join(FIGURES_DIR, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(tex)
        print(f'\n=== {fname} ===')
        print(tex)

    print(f'\nAll tables saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
