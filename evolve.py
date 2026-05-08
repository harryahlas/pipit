"""
evolve.py — natural selection for bitterns.

Two levels of pressure:
    1. WITHIN a generation: competitive habitat (winner eats, losers bleed)
    2. BETWEEN generations: the weakest die, the strongest reproduce

Each generation:
    - Population of N creatures lives together in a competitive habitat
    - After training, evaluate each creature on a mixed eval corpus
    - Bottom half dies
    - Top half reproduces: clone brain weights + Gaussian mutation
    - Offspring get fresh organs, fresh battery, mutated brain

The mutation is the only source of variation after generation 0.
Selection is the only thing shaping the population. No intelligent
design, no hyperparameter tuning mid-run. Whatever wins, wins.

    python evolve.py                              # 8 creatures, 10 generations
    python evolve.py --pop 12 --generations 15    # bigger tournament
    python evolve.py --rounds 8000 --sigma 0.03   # tune pressure

Architecture:

    Generation 0:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
                    |    |    |    |    |    |    |    |
                    ---- compete in habitat for N rounds ----
                    |    |    |    |    |    |    |    |
                    evaluate fitness (teacher-force CE)
                    |    |    |    |    |    |    |    |
                    rank: keep top 4, kill bottom 4
                    |    |    |    |
                    reproduce with mutation
                    |  ↘  |  ↘  |  ↘  |  ↘
    Generation 1:  [b0] [b1] [b2] [b3] [b4] [b5] [b6] [b7]
                    parents       children (mutated clones)
                    ...repeat...
"""

from __future__ import annotations
import argparse
import copy
import os
import time

import numpy as np

from bittern import Bittern, Brain, Organs, Battery, VOCAB_SIZE
from world import make_corpus, bits_to_str
from probe import teacher_force_ce, per_class_ce, probe_report, UNIFORM


# ── Names ────────────────────────────────────────────────────────────

NAME_POOL = [
    'ada', 'bob', 'cid', 'dot', 'eve', 'fig', 'gus', 'hal',
    'ivy', 'jan', 'kit', 'leo', 'max', 'neo', 'ora', 'pax',
]

GEN_PREFIX = [
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ',
    'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π',
    'ρ', 'σ', 'τ', 'υ',
]


# ── Reproduction ─────────────────────────────────────────────────────

BRAIN_PARAMS = ['embedding', 'W_q', 'W_k', 'W_v', 'W_o', 'W_out', 'b_out']


def spawn(parent, child_name, sigma=0.02, rng=None):
    """Create a child from a parent.

    Brain weights are copied and mutated (Gaussian noise, scale sigma).
    Organs are fresh (they learn via Hebbian during training anyway).
    Battery starts full. Round counter resets.

    This is asexual reproduction with mutation — the simplest thing
    that could produce variation for selection to act on.
    """
    rng = rng or np.random.default_rng()

    child = Bittern(
        name=child_name,
        embed_dim=parent.brain.embed_dim,
        brain_dim=parent.brain.brain_dim,
        context=parent.brain.context,
        seed=int(rng.integers(0, 2**31)),
    )

    # Copy parent brain weights + mutate
    for pname in BRAIN_PARAMS:
        parent_w = getattr(parent.brain, pname)
        noise = rng.normal(0, sigma, parent_w.shape)
        setattr(child.brain, pname, parent_w.copy() + noise)

    return child


# ── Competitive habitat (one generation) ─────────────────────────────

def run_generation(creatures, train_corpus, rounds, lr, block_size):
    """Train a population together in a competitive habitat.

    Same mechanics as habitat.py v3: winner gets 6 train pairs + battery,
    losers get 2 train pairs + bleed. Lonely creatures learn at 2x lr.
    Block-structured world for sustained winning/losing streaks.
    """
    # Build block-structured order
    by_class = {}
    for item in train_corpus:
        by_class.setdefault(item['class'], []).append(item)
    class_names = sorted(by_class.keys())
    class_ptrs = {cls: 0 for cls in class_names}

    block_corpus = []
    while len(block_corpus) < rounds + block_size * len(class_names):
        for cls in class_names:
            pool = by_class[cls]
            for _ in range(block_size):
                idx = class_ptrs[cls] % len(pool)
                block_corpus.append(pool[idx])
                class_ptrs[cls] += 1

    n = len(creatures)
    win_counts = {c.name: 0 for c in creatures}
    lonely_counts = {c.name: 0 for c in creatures}
    t0 = time.time()

    for r in range(rounds):
        # Progress bar
        if r % max(1, rounds // 40) == 0 or r == rounds - 1:
            pct = (r + 1) / rounds
            filled = int(pct * 30)
            bar = '█' * filled + '░' * (30 - filled)
            elapsed = time.time() - t0
            eta = (elapsed / (r + 1)) * (rounds - r - 1) if r > 0 else 0
            leader = max(win_counts, key=win_counts.get) if r > 0 else '...'
            print(f"\r    [{bar}] {pct:5.1%}  "
                  f"{elapsed:5.1f}s elapsed  ETA {eta:4.0f}s  "
                  f"leading: {leader}", end='', flush=True)

        world_bits = block_corpus[r % len(block_corpus)]['bits']

        # Score BEFORE learning (single-position, using predict_fast)
        score_prefix = world_bits[:min(8, len(world_bits) - 1)]
        score_target = int(world_bits[len(score_prefix)])
        scores = []
        for c in creatures:
            probs = c.brain.predict_fast(score_prefix)
            scores.append(-float(np.log(probs[score_target] + 1e-12)))

        winner_idx = int(np.argmin(scores))
        win_counts[creatures[winner_idx].name] += 1

        # Competitive learning + battery
        for i, c in enumerate(creatures):
            is_winner = (i == winner_idx)
            effective_lr = lr * (2.0 if c.battery.is_lonely() else 1.0)
            tp = 6 if is_winner else 2
            c.listen(world_bits, train_pairs=tp, lr=effective_lr)
            c.step()
            if is_winner:
                c.battery.replenish(0.06)
            else:
                c.battery.level = max(0.0, c.battery.level - 0.03)
            if c.battery.is_lonely():
                lonely_counts[c.name] += 1

    print()  # newline after progress bar
    return win_counts, lonely_counts


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_population(creatures, eval_corpus):
    """Evaluate each creature. Returns list of (creature, fitness_dict)."""
    results = []
    for c in creatures:
        tf = teacher_force_ce(c, eval_corpus)
        pc = per_class_ce(c, eval_corpus)
        classes_below_uniform = sum(1 for v in pc.values() if v < UNIFORM)
        results.append({
            'creature': c,
            'tf_ce': tf,
            'per_class': pc,
            'classes_below': classes_below_uniform,
        })
    # Sort by fitness: lower tf_ce = better
    results.sort(key=lambda r: r['tf_ce'])
    return results


# ── Main evolution loop ──────────────────────────────────────────────

PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
]


def evolve(pop_size=8, generations=10, rounds_per_gen=5000,
           seed=0, sigma=0.02, lr=0.02, block_size=20, out='babies',
           names=None, champion_name='champion'):
    """Run the tournament."""
    rng = np.random.default_rng(seed)

    # Build corpora once (shared across all generations)
    corpus_rng = np.random.default_rng(seed + 500)
    train_corpus = make_corpus(per_class=200, length=64, rng=corpus_rng)
    eval_corpus = make_corpus(per_class=20, length=64, rng=corpus_rng)

    # Generation 0: all born naive
    print(f"\n{'='*65}")
    print(f"  EVOLUTION — {pop_size} creatures, {generations} generations")
    print(f"  {rounds_per_gen} rounds/gen, sigma={sigma}, block={block_size}")
    print(f"{'='*65}")

    creatures = []
    name_pool = names if names else NAME_POOL
    for i in range(pop_size):
        name = f"{GEN_PREFIX[0]}_{name_pool[i % len(name_pool)]}"
        c = Bittern(name=name, seed=seed + i)
        creatures.append(c)

    champion_history = []

    for gen in range(generations):
        gen_label = GEN_PREFIX[min(gen, len(GEN_PREFIX)-1)]
        print(f"\n{'─'*65}")
        print(f"  Generation {gen} ({gen_label}): "
              f"{', '.join(c.name for c in creatures)}")
        print(f"{'─'*65}")

        # Train together
        t0 = time.time()
        wins, lonelies = run_generation(
            creatures, train_corpus, rounds_per_gen, lr, block_size)
        elapsed = time.time() - t0

        # Show habitat stats
        print(f"  training: {rounds_per_gen} rounds ({elapsed:.1f}s)")
        for c in creatures:
            w = wins.get(c.name, 0)
            l = lonelies.get(c.name, 0)
            pct_w = 100 * w / rounds_per_gen if rounds_per_gen > 0 else 0
            pct_l = 100 * l / rounds_per_gen if rounds_per_gen > 0 else 0
            print(f"    {c.name:12s}  wins={pct_w:4.0f}%  "
                  f"lonely={pct_l:4.0f}%  "
                  f"battery={c.battery.level:.2f}")

        # Evaluate
        results = evaluate_population(creatures, eval_corpus)
        print(f"\n  fitness ranking (lower CE = better):")
        for rank, r in enumerate(results):
            c = r['creature']
            marker = '♛' if rank == 0 else '·' if rank < pop_size // 2 else '✗'
            pc_str = '  '.join(
                f"{cls}={ce:.3f}" for cls, ce in sorted(r['per_class'].items()))
            print(f"    {marker} {rank+1}. {c.name:12s}  "
                  f"CE={r['tf_ce']:.3f}  ({pc_str})")

        # Champion babble
        champ = results[0]['creature']
        print(f"\n  champion {champ.name} babbles:")
        for p in PROBE_PROMPTS:
            bb = champ.babble(p['bits'], n=24)
            print(f"    {p['class']:10s}  {bits_to_str(p['bits'])} → "
                  f"{bits_to_str(bb)}")

        champion_history.append({
            'gen': gen,
            'name': champ.name,
            'tf_ce': results[0]['tf_ce'],
            'per_class': results[0]['per_class'],
        })

        # Selection + reproduction (skip on last generation)
        if gen < generations - 1:
            survivors = [r['creature'] for r in results[:pop_size // 2]]
            dead = [r['creature'] for r in results[pop_size // 2:]]
            print(f"\n  selection: {', '.join(c.name for c in survivors)} survive")
            print(f"             {', '.join(c.name for c in dead)} die")

            # Reproduce
            next_gen = []
            next_label = GEN_PREFIX[min(gen + 1, len(GEN_PREFIX)-1)]
            for i, parent in enumerate(survivors):
                # Parent survives (keeps its trained brain)
                parent.battery = Battery()  # fresh battery
                parent.round = 0
                next_gen.append(parent)

                # One child per parent
                child_name = (f"{next_label}_"
                              f"{name_pool[(i + pop_size//2) % len(name_pool)]}")
                child = spawn(parent, child_name, sigma=sigma, rng=rng)
                next_gen.append(child)
                print(f"    {parent.name} → {child.name} (σ={sigma})")

            creatures = next_gen

    # Final summary
    print(f"\n{'='*65}")
    print(f"  EVOLUTION COMPLETE — champion trajectory")
    print(f"{'='*65}")
    for h in champion_history:
        pc = h['per_class']
        pc_str = '  '.join(f"{cls}={ce:.3f}" for cls, ce in sorted(pc.items()))
        print(f"  gen {h['gen']:2d}  {h['name']:12s}  CE={h['tf_ce']:.3f}  "
              f"({pc_str})")

    # Final champion full probe
    final_champ = results[0]['creature']
    print(f"\n  final champion: {final_champ.name}")
    probe_report(final_champ, eval_corpus, PROBE_PROMPTS,
                 label=f"CHAMPION {final_champ.name}")

    # Compare to a loner trained for the same total rounds
    total_rounds = rounds_per_gen * generations
    print(f"\n  control: loner trained {total_rounds} rounds (no competition)")
    loner = Bittern(name='loner', seed=seed + 999)
    for r in range(total_rounds):
        item = train_corpus[r % len(train_corpus)]
        loner.listen(item['bits'], train_pairs=4, lr=lr)
        loner.step()
        loner.battery.replenish(0.02)
    probe_report(loner, eval_corpus, PROBE_PROMPTS,
                 label=f'loner @ {loner.round}')

    # Save champion
    os.makedirs(out, exist_ok=True)
    final_champ.save(os.path.join(out, champion_name))
    print(f"\n  saved champion -> {out}/{champion_name}")

    return final_champ, champion_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pop', type=int, default=8,
                   help='population size (must be even)')
    p.add_argument('--generations', type=int, default=10)
    p.add_argument('--rounds', type=int, default=5000,
                   help='training rounds per generation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sigma', type=float, default=0.02,
                   help='mutation magnitude')
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--block-size', type=int, default=20)
    p.add_argument('--out', default='babies')
    p.add_argument('--names', type=str, default=None,
                   help='comma-separated creature names (e.g. "zip,zap,zop,zug")')
    p.add_argument('--champion', type=str, default='champion',
                   help='save name for the winner (default: champion)')
    args = p.parse_args()

    name_list = args.names.split(',') if args.names else None

    evolve(pop_size=args.pop, generations=args.generations,
           rounds_per_gen=args.rounds, seed=args.seed,
           sigma=args.sigma, lr=args.lr, block_size=args.block_size,
           out=args.out, names=name_list, champion_name=args.champion)


if __name__ == '__main__':
    main()
