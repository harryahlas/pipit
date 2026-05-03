"""
duet.py — two pipits hearing each other.

    python duet.py

A's output is B's input. B's output is A's input.
Each tick:
    1. A predicts B's last output → A's loss
    2. B predicts A's last output → B's loss
    3. A emits a bit (from its prediction of what comes next)
    4. B emits a bit
    5. A hears B's emission; B hears A's emission

They start hearing a seed rhythm from the world, then the world
goes silent and they only hear each other. What happens?

Possible outcomes:
    - Silence: both collapse to 0000... or 1111...
    - Lock: both produce the same constant pattern
    - Sync: both produce the same rhythm (phase lock)
    - Call-and-response: they alternate or complement
    - Chaos: no stable pattern

The cross-creature prediction loss is the cleanest possible
peer-learning metric: can A predict what B will say?
"""

from __future__ import annotations
import argparse
import numpy as np

from pipit import Pipit
from world import bits_to_str, make_corpus


def duet(n_osc=16, seed=0, warmup_rounds=2000, duet_ticks=200,
         show_every=20):
    rng = np.random.default_rng(seed)

    # Birth two pipits
    a = Pipit(name='A', n_osc=n_osc, seed=seed)
    b = Pipit(name='B', n_osc=n_osc, seed=seed + 1)

    # --- Phase 1: both listen to the world separately ---
    print(f"=== phase 1: listening to the world ({warmup_rounds} sequences) ===")
    corpus = make_corpus(per_class=200, length=64, rng=rng)
    stream = []
    for item in corpus:
        stream.extend(item['bits'])

    # Each pipit hears the same stream
    for bit in stream[:warmup_rounds * 64]:
        a.brain.tick(bit, learn=True)
        b.brain.tick(bit, learn=True)

    print(f"  A entrainment: ", end='')
    R_a = a.brain.entrainment()
    top_a = np.argsort(R_a.sum(axis=1))[::-1][:3]
    for i in top_a:
        T = abs(2 * np.pi / a.brain.omega[i])
        print(f"T≈{T:.1f} R=[{R_a[i,0]:.2f},{R_a[i,1]:.2f}]  ", end='')
    print()
    print(f"  B entrainment: ", end='')
    R_b = b.brain.entrainment()
    top_b = np.argsort(R_b.sum(axis=1))[::-1][:3]
    for i in top_b:
        T = abs(2 * np.pi / b.brain.omega[i])
        print(f"T≈{T:.1f} R=[{R_b[i,0]:.2f},{R_b[i,1]:.2f}]  ", end='')
    print()

    # --- Phase 2: they hear each other ---
    print(f"\n=== phase 2: hearing each other ({duet_ticks} ticks) ===")
    print(f"  {'tick':>5s}  {'A→':>5s}  {'B→':>5s}  "
          f"{'A_loss':>7s}  {'B_loss':>7s}  "
          f"{'A_out':>30s}  {'B_out':>30s}")
    print(f"  {'─'*5}  {'─'*5}  {'─'*5}  "
          f"{'─'*7}  {'─'*7}  "
          f"{'─'*30}  {'─'*30}")

    # Seed: A hears 0, B hears 1 (different starting positions)
    a_hears = 0
    b_hears = 1
    a_out_history = []
    b_out_history = []
    a_losses = []
    b_losses = []

    for t in range(duet_ticks):
        # A processes what B said; B processes what A said
        a_bit, a_probs, a_loss = a.tick(a_hears, learn=True, emit=True)
        b_bit, b_probs, b_loss = b.tick(b_hears, learn=True, emit=True)

        a_out_history.append(a_bit)
        b_out_history.append(b_bit)
        a_losses.append(a_loss)
        b_losses.append(b_loss)

        # Cross: A's output becomes B's next input, and vice versa
        a_hears = b_bit
        b_hears = a_bit

        if (t + 1) % show_every == 0 or t < 5:
            recent_a = bits_to_str(a_out_history[-show_every:])
            recent_b = bits_to_str(b_out_history[-show_every:])
            avg_a = np.mean(a_losses[-show_every:])
            avg_b = np.mean(b_losses[-show_every:])
            print(f"  {t+1:5d}  {a_bit:5d}  {b_bit:5d}  "
                  f"{avg_a:7.3f}  {avg_b:7.3f}  "
                  f"{recent_a:>30s}  {recent_b:>30s}")

    # --- Summary ---
    print(f"\n=== summary ===")
    a_str = bits_to_str(a_out_history)
    b_str = bits_to_str(b_out_history)

    # Agreement: how often do they say the same thing?
    agree = sum(1 for x, y in zip(a_out_history, b_out_history) if x == y)
    print(f"  agreement: {agree}/{duet_ticks} = {agree/duet_ticks:.1%} "
          f"(50% = random)")

    # Complementary: how often do they say opposite things?
    opp = sum(1 for x, y in zip(a_out_history, b_out_history) if x != y)
    print(f"  complement: {opp}/{duet_ticks} = {opp/duet_ticks:.1%}")

    # Auto-correlation: does each pipit produce a pattern?
    for name, hist in [('A', a_out_history), ('B', b_out_history)]:
        arr = np.array(hist, dtype=float)
        if len(arr) > 10:
            # Period-2 autocorrelation
            ac2 = np.corrcoef(arr[:-2], arr[2:])[0, 1] if len(arr) > 4 else 0
            # Period-4 autocorrelation
            ac4 = np.corrcoef(arr[:-4], arr[4:])[0, 1] if len(arr) > 8 else 0
            # Bit balance
            frac_1 = arr.mean()
            print(f"  {name}: frac_1={frac_1:.2f}, "
                  f"autocorr(lag=2)={ac2:+.2f}, "
                  f"autocorr(lag=4)={ac4:+.2f}")

    # Cross-correlation: does A predict B?
    a_arr = np.array(a_out_history, dtype=float)
    b_arr = np.array(b_out_history, dtype=float)
    if len(a_arr) > 2:
        # A at time t vs B at time t+1 (does A's output predict B's next?)
        xc = np.corrcoef(a_arr[:-1], b_arr[1:])[0, 1]
        print(f"  cross-corr A(t) → B(t+1): {xc:+.3f}")
        xc2 = np.corrcoef(b_arr[:-1], a_arr[1:])[0, 1]
        print(f"  cross-corr B(t) → A(t+1): {xc2:+.3f}")

    # Final cross-prediction losses
    print(f"\n  A's avg loss (predicting B): {np.mean(a_losses):.3f} "
          f"(ln2={np.log(2):.3f})")
    print(f"  B's avg loss (predicting A): {np.mean(b_losses):.3f}")

    # Show last 64 bits of each
    print(f"\n  A's last 64: {bits_to_str(a_out_history[-64:])}")
    print(f"  B's last 64: {bits_to_str(b_out_history[-64:])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-osc', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--ticks', type=int, default=200)
    args = p.parse_args()
    duet(n_osc=args.n_osc, seed=args.seed,
         warmup_rounds=args.warmup, duet_ticks=args.ticks)


if __name__ == '__main__':
    main()
