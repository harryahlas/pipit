"""
train_pipit.py — train a pipit on bit rhythms.

    python train_pipit.py                     # birth 'p1', 5000 rounds
    python train_pipit.py --name p2 --n-osc 16

The pipit experiences sequences tick-by-tick, learning from each
prediction error as it arrives. No batching, no teacher forcing,
no listen/babble split. Each "round" is one full sequence experienced
online.

Training loop:
    for each round:
        pick a sequence from the corpus
        feed it to the pipit bit by bit (pipit.tick with lr > 0)
        battery management
        probe at checkpoints
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

from pipit import Pipit
from world import make_corpus, bits_to_str
from probe_pipit import probe_report, UNIFORM


PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
    {'bits': [0, 1, 1, 0, 0, 1],          'class': 'mixed'},
    {'bits': [1, 0, 0, 0, 1, 1, 1, 0],    'class': 'mixed'},
]


def show_babbles(pipit, prompts, n=24):
    print(f"    babbles ({n} bits each):")
    for prompt in prompts:
        bb = pipit.babble_snapshot(prompt['bits'], n=n)
        print(f"      {prompt['class']:10s}  "
              f"prompt={bits_to_str(prompt['bits']):10s}  "
              f"babble={bits_to_str(bb)}")


def show_phases(pipit):
    """Show entrainment strengths — which oscillators are phase-locked."""
    brain = pipit.brain
    if brain.memory_mode in ('ema', 'dual'):
        R_long = np.sqrt(brain.mem_cos**2 + brain.mem_sin**2)
        if brain.memory_mode == 'dual':
            R_short = np.sqrt(brain.short_cos**2 + brain.short_sin**2)
            confidence = R_short.sum(axis=1)
            order = np.argsort(confidence)[::-1]
            parts = []
            for i in order[:6]:
                period = abs(2 * np.pi / brain.omega[i]) if abs(brain.omega[i]) > 1e-6 else float('inf')
                parts.append(f"osc[{i}] T≈{period:.1f} "
                             f"Rlong=[{R_long[i,0]:.2f},{R_long[i,1]:.2f}] "
                             f"Rshort=[{R_short[i,0]:.2f},{R_short[i,1]:.2f}]")
            print(f"    entrainment (sorted by short-term confidence):")
            for part in parts:
                print(f"      {part}")
        else:
            total_R = R_long.sum(axis=1)
            order = np.argsort(total_R)[::-1]
            parts = []
            for i in order[:6]:
                period = abs(2 * np.pi / brain.omega[i]) if abs(brain.omega[i]) > 1e-6 else float('inf')
                parts.append(f"osc[{i}] T≈{period:.1f} R=[{R_long[i,0]:.2f},{R_long[i,1]:.2f}]")
            print(f"    entrainment: {', '.join(parts)}")
    else:
        valid = min(brain.buf_count, brain.buf_size)
        bits_0 = int((brain.buf_bit[:valid] == 0).sum())
        bits_1 = int((brain.buf_bit[:valid] == 1).sum())
        print(f"    buffer: {valid}/{brain.buf_size} filled, "
              f"0s={bits_0} 1s={bits_1}")


def train(name='p1', seed=0, rounds=5000, out='babies',
          n_osc=32, per_class=200, length=64, eval_per_class=20,
          lr=0.005, reset_phases=True,
          memory_mode='ema', mem_alpha=0.97, mem_alpha_short=0.5,
          buf_size=32):
    rng = np.random.default_rng(seed)

    print(f"\n=== building corpora (seed={seed}) ===")
    train_corpus = make_corpus(per_class=per_class, length=length, rng=rng)
    eval_corpus = make_corpus(per_class=eval_per_class, length=length, rng=rng)
    print(f"  train: {len(train_corpus)} sequences, {length} bits each")
    print(f"  eval:  {len(eval_corpus)} sequences, {length} bits each")

    print(f"\n=== birthing pipit '{name}' ===")
    pipit = Pipit(name=name, n_osc=n_osc, memory_mode=memory_mode,
                  mem_alpha=mem_alpha, mem_alpha_short=mem_alpha_short,
                  buf_size=buf_size, seed=seed)
    if memory_mode == 'ema':
        mode_str = f"ema(alpha={mem_alpha})"
    elif memory_mode == 'dual':
        mode_str = f"dual(long={mem_alpha}, short={mem_alpha_short})"
    else:
        mode_str = f"episodic(buf={buf_size})"
    print(f"  n_osc={pipit.brain.n_osc}  memory={mode_str}")
    print(f"  reset_phases={reset_phases}")

    # Untrained baseline
    probe_report(pipit, eval_corpus, PROBE_PROMPTS, label='untrained')
    show_babbles(pipit, PROBE_PROMPTS)

    # Training
    print(f"\n=== experiencing {rounds} sequences ===")
    checkpoints = sorted(set([
        max(1, rounds // 4),
        max(1, rounds // 2),
        max(1, 3 * rounds // 4),
        rounds,
    ]))
    chk = 0
    t0 = time.time()
    all_losses = []

    # Track per-class CE trajectory for early stopping
    best_ce = {cls: UNIFORM for cls in ['steady_2', 'swelling', 'staccato']}
    best_round = 0

    for r in range(rounds):
        item = train_corpus[r % len(train_corpus)]
        bits = item['bits']

        if reset_phases:
            pipit.brain.reset_phases(pipit.rng)

        # Experience the sequence tick by tick
        round_losses = []
        for bit in bits:
            _, _, loss = pipit.tick(int(bit), learn=True, emit=False)
            round_losses.append(loss)
        avg_loss = float(np.mean(round_losses))
        all_losses.append(avg_loss)

        pipit.battery.replenish(0.02)

        if (r + 1) == checkpoints[chk]:
            elapsed = time.time() - t0
            recent = all_losses[-200:]
            recent_avg = float(np.mean(recent))
            print(f"\n  --- checkpoint @ round {r+1}  "
                  f"(elapsed {elapsed:.1f}s, "
                  f"recent avg loss {recent_avg:.3f}) ---")

            report = probe_report(pipit, eval_corpus, PROBE_PROMPTS,
                                  label=f'{name} @ round {r+1}')
            show_babbles(pipit, PROBE_PROMPTS)
            show_phases(pipit)

            # Check for regression
            current_ce = report['per_class']
            regressing = False
            for cls in ['steady_2', 'swelling', 'staccato']:
                if cls in current_ce:
                    if current_ce[cls] < best_ce.get(cls, UNIFORM):
                        best_ce[cls] = current_ce[cls]
                        best_round = r + 1
                    elif (current_ce[cls] > UNIFORM and
                          best_ce.get(cls, UNIFORM) < UNIFORM * 0.9):
                        regressing = True
            if regressing:
                print(f"    ⚠ REGRESSION DETECTED — a class has risen "
                      f"past uniform after being below it.")
                print(f"      best CEs were at round {best_round}: "
                      f"{best_ce}")

            chk = min(chk + 1, len(checkpoints) - 1)

    # Save
    os.makedirs(out, exist_ok=True)
    save_path = os.path.join(out, name)
    pipit.save(save_path)
    print(f"\n=== saved {name} -> {save_path}.{{npz,json}} "
          f"(tick {pipit.round}) ===")
    return pipit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='p1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=5000)
    p.add_argument('--out', default='babies')
    p.add_argument('--n-osc', type=int, default=32)
    p.add_argument('--per-class', type=int, default=200)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--eval-per-class', type=int, default=20)
    p.add_argument('--no-reset', action='store_true',
                   help='don\'t reset phases between sequences')
    p.add_argument('--memory', choices=['ema', 'dual', 'episodic'], default='ema',
                   help='memory mode: ema, dual, or episodic')
    p.add_argument('--alpha', type=float, default=0.97,
                   help='long-term EMA decay. For ema and dual modes')
    p.add_argument('--alpha-short', type=float, default=0.5,
                   help='short-term EMA decay (dual mode confidence window)')
    p.add_argument('--buf-size', type=int, default=32,
                   help='episodic buffer size. Only for --memory episodic')
    args = p.parse_args()
    train(args.name, args.seed, args.rounds, args.out,
          n_osc=args.n_osc, per_class=args.per_class,
          length=args.length, eval_per_class=args.eval_per_class,
          reset_phases=not args.no_reset,
          memory_mode=args.memory, mem_alpha=args.alpha,
          mem_alpha_short=args.alpha_short,
          buf_size=args.buf_size)


if __name__ == '__main__':
    main()
