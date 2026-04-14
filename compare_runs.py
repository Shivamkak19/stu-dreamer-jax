"""Quick baseline-vs-stuB comparison plot.

Usage: python /tmp/compare_runs.py [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    'baseline (use_stu=False)': '/home/ubuntu/dreamer_runs/baseline_proprio_20260411T173021/scores.jsonl',
    'stuB (spectral filters)':  None,
    'stuRand (random filters)': None,
}


def latest(prefix):
    matches = sorted(Path('/home/ubuntu/dreamer_runs').glob(prefix + '*'))
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        return None
    return str(matches[-1] / 'scores.jsonl')


RUNS['stuB (spectral filters)'] = latest('stuB_proprio_')
RUNS['stuRand (random filters)'] = latest('stuRand_proprio_')


def load(path):
    if path is None or not Path(path).exists():
        return np.array([]), np.array([])
    xs, ys = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'episode/score' in r and 'step' in r:
            xs.append(r['step'])
            ys.append(r['episode/score'])
    return np.array(xs), np.array(ys)


def smooth(ys, k=20):
    if len(ys) < k:
        return ys
    return np.convolve(ys, np.ones(k) / k, mode='valid')


fig, ax = plt.subplots(figsize=(9, 5))
for label, path in RUNS.items():
    xs, ys = load(path)
    if len(ys) == 0:
        print(f'  [skip] {label}: no scores at {path}')
        continue
    print(f'  [{label}]: {len(ys)} eps, last 50 mean = {ys[-50:].mean():.1f}, max = {ys.max():.1f}')
    ax.plot(xs, ys, alpha=0.25)
    sm = smooth(ys)
    if len(sm):
        ax.plot(xs[-len(sm):], sm, label=label, linewidth=2)

ax.set_xlabel('env step')
ax.set_ylabel('episode/score')
ax.set_title('dmc_walker_walk: baseline vs STU integrations')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/compare_runs.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'wrote {out}')
