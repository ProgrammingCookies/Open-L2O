"""Read training_profiles.jsonl and print a scaling comparison table.

Each row is one training run. Rows are sorted by model, then number of layers,
so scaling behaviour is easy to read down each model's block.

Outputs:
  profiles/scaling_comparison.csv   -- for import into Excel / pandas
  profiles/scaling_comparison.txt   -- plain-text copy of the printed table
"""

import csv
import json
import os

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '.', 'Base directory containing the profiles/ folder.')


COLUMNS = [
    ('model_name',        'Model'),
    ('num_layers',        'Layers'),
    ('m',                 'M'),
    ('n',                 'N'),
    ('lasso_lam',         'Lambda'),
    ('noise_std',         'Noise'),
    ('training_time_s',   'Train time (s)'),
    ('gpu_mem_mean_mb',   'GPU mem mean (MB)'),
    ('gpu_mem_median_mb', 'GPU mem median (MB)'),
    ('gpu_mem_n_samples', 'Mem samples'),
    ('timestamp',         'Timestamp'),
]


def _load(base_dir):
    path = os.path.join(base_dir, 'profiles', 'training_profiles.jsonl')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sort_key(p):
    return (
        p.get('model_name', ''),
        p.get('m', 0) or 0,
        p.get('n', 0) or 0,
        p.get('lasso_lam', 0) or 0,
        p.get('num_layers', 0) or 0,
    )


def _build_table(profiles):
    profiles = sorted(profiles, key=_sort_key)
    headers = [label for _, label in COLUMNS]
    rows = []
    for p in profiles:
        row = []
        for key, _ in COLUMNS:
            val = p.get(key, '')
            row.append('' if val is None else str(val))
        rows.append(row)
    return headers, rows


def _render(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    fmt_row = lambda cells: '|' + '|'.join(f' {c:{w}} ' for c, w in zip(cells, widths)) + '|'

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return '\n'.join(lines)


def main(_):
    profiles = _load(FLAGS.base_dir)
    if not profiles:
        logging.warning('No profiles found. Run some training first.')
        return

    headers, rows = _build_table(profiles)
    table = _render(headers, rows)
    print(table)

    out_dir = os.path.join(FLAGS.base_dir, 'profiles')

    csv_path = os.path.join(out_dir, 'scaling_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
        csv.writer(f).writerows(rows)

    txt_path = os.path.join(out_dir, 'scaling_comparison.txt')
    with open(txt_path, 'w') as f:
        f.write(table + '\n')

    logging.info('Table saved to %s and %s', csv_path, txt_path)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
