"""
habitat.py — a shared world where bitterns live together.

Three bitterns, born naive, sharing a world of bit rhythms. Each
round they all hear the same world sequence, babble responses, and
then hear each other's babbles. Battery replenishes from social
agreement — when your neighbor babbles what you expected, you
feel less lonely.

This is the first time bitterns interact. The colony architecture
(colony_bittern.py) was an engineering trick — pre-trained
specialists with a router. This is creatures that live together,
learn from the same world AND from each other, and develop
whatever they develop.

    python habitat.py                       # default 30k rounds
    python habitat.py --rounds 50000        # longer
    python habitat.py --n-creatures 2       # pair instead of trio

What we're watching for:
    - Emergent specialization (different creatures get better at
      different rhythms, without being told to)
    - Social reinforcement (creatures that agree are happier)
    - Collective vs individual (does the group outperform loners?)
    - Babble dynamics (do they develop coordinated patterns?)
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

from bittern import Bittern, VOCAB_SIZE
from world import make_corpus, bits_to_str
from probe import (probe_report, teacher_force_ce, free_running_ce,
                   per_class_ce, UNIFORM)


# How much of a neighbor's babble to learn from, relative to world lr
SOCIAL_LR_SCALE = 0.5

# How many bits each creature babbles per round as social signal
SOCIAL_BABBLE_LEN = 8

# Battery bonus per unit of social agreement (0-1 scale)
SOCIAL_BATTERY_SCALE = 0.03


PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
    {'bits': [0, 1, 1, 0, 0, 1],          'class': 'mixed'},
    {'bits': [1, 0, 0, 0, 1, 1, 1, 0],    'class': 'mixed'},
]


class Habitat:
    """A shared world where bitterns live together.

    Each round:
        1. World emits a sequence (from corpus).
        2. All creatures listen to the world sequence and learn.
        3. Each creature babbles a short response.
        4. Each creature hears every neighbor's babble and learns from it.
        5. Battery dynamics: social agreement replenishes,
           disagreement lets it decay naturally.

    The creatures share nothing — no weights, no organs, no state.
    They only interact through the bit stream: hearing each other's
    babble. Whatever develops, develops from that.
    """

    def __init__(self, names=None, seed=0):
        if names is None:
            names = ['ari', 'bea', 'cal']
        self.names = names
        self.creatures = []
        for i, name in enumerate(names):
            c = Bittern(name=name, seed=seed + i)
            self.creatures.append(c)
        self.round = 0
        self.social_history = []  # track agreement over time

    def live_round(self, world_bits, lr=0.02):
        """One round of shared living. Competition hurts.

        1. Score all creatures on this sequence BEFORE learning
        2. Winner gets extra training (6 pairs) and battery (+0.06)
        3. Losers get reduced training (2 pairs) and BLEED battery (-0.03)
        4. Lonely creatures learn at 2x lr (desperation = plasticity)
        5. Natural battery decay on top of everything

        A creature that loses 10 rounds in a row drops from 0.8 to
        ~0.15 (0.8 - 10*(0.005+0.03) = 0.45, then lonely kicks in).
        It starts learning at double speed, which either saves it or
        accelerates its divergence from the pack.
        """
        n = len(self.creatures)

        # 1. Score all creatures BEFORE learning (single-position, fast)
        score_prefix = world_bits[:min(8, len(world_bits) - 1)]
        score_target = int(world_bits[len(score_prefix)])
        scores = []
        for c in self.creatures:
            probs = c.brain.predict_fast(score_prefix)
            scores.append(-float(np.log(probs[score_target] + 1e-12)))

        # 2. Winner = lowest prediction loss
        winner_idx = int(np.argmin(scores))

        # 3. Competitive learning + battery
        for i, c in enumerate(self.creatures):
            is_winner = (i == winner_idx)

            # Lonely creatures learn faster (desperation)
            effective_lr = lr * (2.0 if c.battery.is_lonely() else 1.0)

            # Winner gets more training
            tp = 6 if is_winner else 2
            c.listen(world_bits, train_pairs=tp, lr=effective_lr)
            c.step()  # natural decay (0.005)

            if is_winner:
                c.battery.replenish(0.06)       # fed
            else:
                c.battery.level = max(0.0,
                    c.battery.level - 0.03)      # bleed

        # 4. Babble for social observation (no learning from babble)
        babbles = []
        for c in self.creatures:
            prompt = world_bits[-min(len(world_bits), c.brain.context):]
            bb = c.babble(prompt, n=SOCIAL_BABBLE_LEN)
            babbles.append(bb)

        # 5. Agreement tracking (diagnostics only)
        agreements = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                match = sum(a == b for a, b in
                            zip(babbles[i], babbles[j])) / len(babbles[i])
                agreements[i, j] = match

        self.round += 1

        metrics = {
            'round': self.round,
            'agreements': agreements,
            'batteries': [c.battery.level for c in self.creatures],
            'lonely': [c.battery.is_lonely() for c in self.creatures],
            'winner': self.names[winner_idx],
            'scores': scores,
        }
        self.social_history.append(metrics)
        return metrics

    def probe_all(self, eval_corpus, label=None):
        """Probe every creature and show comparative results."""
        title = label or f"habitat @ round {self.round}"
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

        # Social state
        if self.social_history:
            recent = self.social_history[-200:]
            # Winner distribution
            winner_counts = {}
            for m in recent:
                w = m.get('winner', None)
                if w:
                    winner_counts[w] = winner_counts.get(w, 0) + 1
            total_w = sum(winner_counts.values())
            if total_w > 0:
                w_parts = [f"{n}={100*c/total_w:.0f}%"
                           for n, c in sorted(winner_counts.items())]
                print(f"  winner distribution (recent {len(recent)}): "
                      f"{', '.join(w_parts)}")

            for i, c in enumerate(self.creatures):
                lonely_pct = np.mean([m['lonely'][i] for m in recent])
                print(f"  {c.name}: battery={c.battery.level:.2f}  "
                      f"lonely={lonely_pct:.0%} of recent")

        # Per-creature probe
        results = {}
        for c in self.creatures:
            r = probe_report(c, eval_corpus, PROBE_PROMPTS,
                             label=f"{c.name} @ {c.round}")
            results[c.name] = r

        # Comparative per-class table
        print(f"\n  Per-class CE comparison:")
        classes = sorted(set(
            cls for r in results.values()
            for cls in r['per_class'].keys()
        ))
        header = f"  {'':8s}" + ''.join(f"  {cls:>10s}" for cls in classes)
        print(header)
        for name in self.names:
            r = results[name]
            row = f"  {name:8s}"
            for cls in classes:
                ce = r['per_class'].get(cls, 0)
                marker = '✓' if ce < UNIFORM * 0.5 else '·' if ce < UNIFORM else '✗'
                row += f"  {ce:8.3f}{marker}"
            print(row)

        return results

    def show_babbles(self, prompts=None, n=24):
        """Show each creature's babble side-by-side."""
        if prompts is None:
            prompts = PROBE_PROMPTS
        print(f"\n  Babbles ({n} bits):")
        for p in prompts:
            cls = p['class']
            bits = p['bits']
            print(f"    prompt={bits_to_str(bits)} ({cls}):")
            for c in self.creatures:
                bb = c.babble(bits, n=n)
                print(f"      {c.name:6s}  {bits_to_str(bb)}")

    def show_social_babble(self, prompt, n=16):
        """Each creature babbles, then hears the others and babbles again.
        Shows the live social dynamic in action."""
        print(f"\n  Social babble (prompt={bits_to_str(prompt)}):")
        # Round 1: independent babble
        babbles = []
        for c in self.creatures:
            bb = c.babble(prompt, n=n)
            babbles.append(bb)
            print(f"    round 1  {c.name:6s}  {bits_to_str(bb)}")

        # Round 2: each creature hears neighbors, then babbles again
        print(f"    --- hearing each other ---")
        for i, c in enumerate(self.creatures):
            # Context = prompt + what neighbors just said
            combined = list(prompt)
            for j, bb in enumerate(babbles):
                if i != j:
                    combined.extend(bb)
            # Babble from this richer context
            bb2 = c.babble(combined[-c.brain.context:], n=n)
            print(f"    round 2  {c.name:6s}  {bits_to_str(bb2)}")


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------

def run_habitat(names=None, seed=0, rounds=30000, out='babies',
                per_class=200, length=64, lr=0.02, block_size=20):
    """Birth creatures and let them live together.

    block_size: how many consecutive rounds use the same rhythm class.
    Larger blocks create sustained winning/losing streaks, which is
    what makes batteries actually move. Default 20 means ~60 rounds
    per full cycle through all three classes.
    """
    if names is None:
        names = ['ari', 'bea', 'cal']

    rng = np.random.default_rng(seed)

    print(f"\n=== building world (seed={seed}) ===")
    train_corpus = make_corpus(per_class=per_class, length=length, rng=rng)
    eval_corpus = make_corpus(per_class=20, length=length, rng=rng)

    # Build block-structured presentation order:
    # group by class, then cycle through blocks
    by_class = {}
    for item in train_corpus:
        by_class.setdefault(item['class'], []).append(item)
    class_names = sorted(by_class.keys())
    class_ptrs = {cls: 0 for cls in class_names}

    def next_block():
        """Yield block_size items from each class in rotation."""
        items = []
        for cls in class_names:
            pool = by_class[cls]
            for _ in range(block_size):
                idx = class_ptrs[cls] % len(pool)
                items.append(pool[idx])
                class_ptrs[cls] += 1
        return items

    block_corpus = []
    while len(block_corpus) < rounds + block_size * len(class_names):
        block_corpus.extend(next_block())

    print(f"  train: {len(train_corpus)} sequences, block_size={block_size}")
    print(f"  world order: {block_size} rounds of each class in rotation")

    print(f"\n=== birthing {len(names)} creatures: {', '.join(names)} ===")
    habitat = Habitat(names=names, seed=seed)
    for c in habitat.creatures:
        print(f"  {c.name}: brain_dim={c.brain.brain_dim}  "
              f"embed_dim={c.brain.embed_dim}  "
              f"context={c.brain.context}  seed={c.seed}")

    # Baseline probe
    habitat.probe_all(eval_corpus, label='untrained')
    habitat.show_babbles()

    # Living
    print(f"\n=== living for {rounds} rounds ===")
    checkpoints = sorted(set([
        max(1, rounds // 4),
        max(1, rounds // 2),
        max(1, 3 * rounds // 4),
        rounds,
    ]))
    chk = 0
    t0 = time.time()

    for r in range(rounds):
        item = block_corpus[r % len(block_corpus)]
        habitat.live_round(item['bits'], lr=lr)

        if habitat.round == checkpoints[chk]:
            elapsed = time.time() - t0
            print(f"\n  --- checkpoint @ round {habitat.round} "
                  f"({elapsed:.1f}s) ---")
            habitat.probe_all(eval_corpus,
                              label=f'habitat @ {habitat.round}')
            habitat.show_babbles()
            habitat.show_social_babble([0, 1, 0, 1, 0, 1])
            chk = min(chk + 1, len(checkpoints) - 1)

    # Compare to a loner (solo bittern trained same rounds, no social)
    print(f"\n=== control: solo bittern, same rounds, no social ===")
    loner = Bittern(name='loner', seed=seed + 100)
    for r in range(rounds):
        item = train_corpus[r % len(train_corpus)]
        loner.listen(item['bits'], train_pairs=4, lr=lr)
        loner.step()
        loner.battery.replenish(0.02)
    probe_report(loner, eval_corpus, PROBE_PROMPTS,
                 label=f'loner @ {loner.round}')

    # Save
    os.makedirs(out, exist_ok=True)
    for c in habitat.creatures:
        path = os.path.join(out, c.name)
        c.save(path)
        print(f"  saved {c.name} -> {path}")
    loner.save(os.path.join(out, 'loner'))
    print(f"  saved loner -> {out}/loner")

    return habitat, loner


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=30000)
    p.add_argument('--n-creatures', type=int, default=3)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--block-size', type=int, default=20,
                   help='rounds of same class before switching')
    p.add_argument('--out', default='babies')
    args = p.parse_args()

    # Name the creatures
    name_pool = ['ari', 'bea', 'cal', 'dia', 'eve']
    names = name_pool[:args.n_creatures]

    run_habitat(names=names, seed=args.seed, rounds=args.rounds,
                lr=args.lr, out=args.out, block_size=args.block_size)


if __name__ == '__main__':
    main()
