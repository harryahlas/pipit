"""
compare_memory.py — compare EMA vs episodic memory modes.

Runs 5 configurations at 50k rounds each, all same seed:
    1. ema  alpha=0.97  (original — long-term average)
    2. ema  alpha=0.80  (medium memory)
    3. ema  alpha=0.60  (short memory)
    4. episodic buf=32   (last 32 observations)
    5. episodic buf=64   (last 64 observations)

Prints a summary table at the end comparing:
    - per-class CE (below 0.693 = learned)
    - free-running CE (below tf_ce = no exposure bias)
    - babble samples from each class prompt

    python compare_memory.py
    python compare_memory.py --rounds 20000    # faster test
"""

from __future__ import annotations
import argparse
import time

import numpy as np

from pipit import Pipit
from world import make_corpus, bits_to_str
from probe_pipit import probe_report, teacher_force_ce, free_running_ce, per_class_ce, UNIFORM


PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
]

CONFIGS = [
    {'label': 'ema_a97',    'memory_mode': 'ema',      'mem_alpha': 0.97, 'mem_alpha_short': 0.5, 'buf_size': 32},
    {'label': 'ema_a80',    'memory_mode': 'ema',      'mem_alpha': 0.80, 'mem_alpha_short': 0.5, 'buf_size': 32},
    {'label': 'dual_s50',   'memory_mode': 'dual',     'mem_alpha': 0.97, 'mem_alpha_short': 0.5, 'buf_size': 32},
    {'label': 'dual_s70',   'memory_mode': 'dual',     'mem_alpha': 0.97, 'mem_alpha_short': 0.7, 'buf_size': 32},
    {'label': 'dual_s30',   'memory_mode': 'dual',     'mem_alpha': 0.97, 'mem_alpha_short': 0.3, 'buf_size': 32},
]


def run_one(cfg, train_corpus, eval_corpus, rounds, seed, reset_phases,
            n_osc=32):
    label = cfg['label']
    pipit = Pipit(
        name=label, n_osc=n_osc, seed=seed,
        memory_mode=cfg['memory_mode'],
        mem_alpha=cfg['mem_alpha'],
        mem_alpha_short=cfg['mem_alpha_short'],
        buf_size=cfg['buf_size'],
    )

    t0 = time.time()
    for r in range(rounds):
        item = train_corpus[r % len(train_corpus)]
        if reset_phases:
            pipit.brain.reset_phases(pipit.rng)
        for bit in item['bits']:
            pipit.tick(int(bit), learn=True, emit=False)
        pipit.battery.replenish(0.02)
    elapsed = time.time() - t0

    # Evaluate
    tf = teacher_force_ce(pipit, eval_corpus)
    fr = free_running_ce(pipit, PROBE_PROMPTS, n_per_prompt=32)
    by_class = per_class_ce(pipit, eval_corpus)

    # Babble samples
    babbles = {}
    for prompt_info in PROBE_PROMPTS:
        cls = prompt_info['class']
        bb = pipit.babble_snapshot(prompt_info['bits'], n=32)
        babbles[cls] = bits_to_str(bb)

    return {
        'label': label,
        'tf_ce': tf,
        'fr_ce': fr,
        'per_class': by_class,
        'babbles': babbles,
        'elapsed': elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n-osc', type=int, default=32)
    p.add_argument('--no-reset', action='store_true')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    train_corpus = make_corpus(per_class=200, length=64, rng=rng)
    eval_corpus = make_corpus(per_class=20, length=64, rng=rng)
    reset = not args.no_reset

    print(f"=== memory mode comparison ===")
    print(f"  rounds={args.rounds}  seed={args.seed}  "
          f"n_osc={args.n_osc}  reset_phases={reset}")
    print(f"  train: {len(train_corpus)} seqs  eval: {len(eval_corpus)} seqs")
    print()

    results = []
    for cfg in CONFIGS:
        label = cfg['label']
        mm = cfg['memory_mode']
        if mm == 'ema':
            mode_desc = f"ema(α={cfg['mem_alpha']})"
        elif mm == 'dual':
            mode_desc = f"dual(long={cfg['mem_alpha']}, short={cfg['mem_alpha_short']})"
        else:
            mode_desc = f"episodic(buf={cfg['buf_size']})"
        print(f"--- training {label} [{mode_desc}] ---")
        r = run_one(cfg, train_corpus, eval_corpus, args.rounds,
                    args.seed, reset, n_osc=args.n_osc)
        results.append(r)
        print(f"  done in {r['elapsed']:.1f}s  "
              f"tf_ce={r['tf_ce']:.3f}  fr_ce={r['fr_ce']:.3f}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY  (ln2 = {UNIFORM:.3f},  below = learned,  "
          f"above = worse than random)")
    print(f"{'='*80}")
    print()

    classes = ['steady_2', 'swelling', 'staccato']

    # Header
    print(f"  {'config':<12s}  {'tf_ce':>6s}  {'fr_ce':>6s}  "
          f"{'gap':>6s}  ", end='')
    for cls in classes:
        print(f"  {cls:>10s}", end='')
    print()
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  ", end='')
    for _ in classes:
        print(f"  {'─'*10}", end='')
    print()

    # Rows
    for r in results:
        gap = r['fr_ce'] - r['tf_ce']
        print(f"  {r['label']:<12s}  {r['tf_ce']:6.3f}  {r['fr_ce']:6.3f}  "
              f"{gap:+6.3f}  ", end='')
        for cls in classes:
            ce = r['per_class'].get(cls, 0)
            marker = '✓' if ce < UNIFORM * 0.7 else '·' if ce < UNIFORM else '✗'
            print(f"  {marker}{ce:9.3f}", end='')
        print()

    # Babbles
    print(f"\n  babble samples (32 bits from class prompts):")
    for r in results:
        print(f"\n  {r['label']}:")
        for cls in classes:
            print(f"    {cls:10s}  {r['babbles'].get(cls, '???')}")


if __name__ == '__main__':
    main()
