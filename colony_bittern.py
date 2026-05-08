"""
colony_bittern.py — a colony of specialist bitterns.

Each bittern is trained on a single rhythm class (steady_2, swelling,
staccato). At inference, all three hear the same input. The one with
the lowest recent prediction loss gets to speak.

This is the pipit colony architecture (competitive selection by recent
loss) but with bittern's attention brain as each specialist — combining
bittern's much stronger single-class performance with pipit's proven
routing mechanism.

    python colony_bittern.py                    # train + evaluate
    python colony_bittern.py --rounds 65000     # full 65k training
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

from bittern import Bittern, VOCAB_SIZE
from world import make_corpus, bits_to_str
from probe import probe_report, teacher_force_ce, per_class_ce, UNIFORM


# ----------------------------------------------------------------------
#  Colony
# ----------------------------------------------------------------------

class BitternColony:
    """Three specialist bitterns with competitive selection.

    Each specialist is trained on one rhythm class. At inference,
    all hear the same input. The specialist with the lowest recent
    prediction loss speaks. This is pipit's colony architecture
    with bittern's stronger brain.
    """

    def __init__(self, specialists: dict[str, Bittern],
                 selector_alpha: float = 0.9):
        self.specialists = specialists
        self.class_names = list(specialists.keys())
        self.selector_alpha = selector_alpha
        self.running_loss = {name: float(np.log(2))
                             for name in self.class_names}

    def predict_all(self, window):
        """Get predictions from all specialists for a context window."""
        predictions = {}
        losses = {}
        for name, bittern in self.specialists.items():
            probs = bittern.brain.predict_probs(window)
            predictions[name] = probs
        return predictions

    def tick(self, bit, context):
        """All specialists predict, winner selected by running loss.

        Returns (winner_name, predictions_dict).
        """
        bit = int(bit)
        predictions = {}
        losses = {}

        for name, bittern in self.specialists.items():
            window = context[-bittern.brain.context:]
            probs = bittern.brain.predict_probs(window)
            predictions[name] = probs
            # Loss on this actual bit
            losses[name] = -np.log(probs[bit] + 1e-12)

        # Winner = lowest running loss BEFORE this tick
        winner = min(self.class_names, key=lambda n: self.running_loss[n])

        # Update running losses
        for name in self.class_names:
            a = self.selector_alpha
            self.running_loss[name] = (a * self.running_loss[name]
                                       + (1 - a) * losses[name])

        return winner, predictions, losses

    def reset(self):
        """Reset running losses to uniform."""
        self.running_loss = {name: float(np.log(2))
                             for name in self.class_names}

    def babble(self, prompt, n=32):
        """Process prompt bits, then generate n bits using competitive
        selection. Each generated bit uses the winner's prediction."""
        context = list(prompt)
        self.reset()

        # Feed prompt
        for bit in prompt:
            self.tick(bit, context[:context.index(bit)+1] if context else [bit])

        # Actually feed all prompt bits properly
        self.reset()
        feed_ctx = []
        for bit in prompt:
            feed_ctx.append(int(bit))
            if len(feed_ctx) > 1:
                self.tick(bit, feed_ctx[:-1])

        # Generate
        out = []
        winners = []
        full_ctx = list(prompt)
        for _ in range(n):
            # Get winner's prediction
            winner = min(self.class_names,
                         key=lambda n: self.running_loss[n])
            bittern = self.specialists[winner]
            window = full_ctx[-bittern.brain.context:]
            probs = bittern.brain.predict_probs(window)

            # Sample from winner's distribution
            choice = int(bittern.rng.choice(VOCAB_SIZE, p=probs))
            out.append(choice)
            winners.append(winner)

            # All specialists observe the generated bit
            full_ctx.append(choice)
            self.tick(choice, full_ctx[:-1])

        return out, winners


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------

PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
    {'bits': [0, 1, 1, 0, 0, 1],          'class': 'mixed'},
    {'bits': [1, 0, 0, 0, 1, 1, 1, 0],    'class': 'mixed'},
]


def train_specialist(name, rhythm_class, seed=0, rounds=65000,
                     per_class=200, length=64, lr=0.02):
    """Train a single bittern on one rhythm class."""
    rng = np.random.default_rng(seed)
    train_corpus = make_corpus(per_class=per_class, length=length,
                               classes=[rhythm_class], rng=rng)
    eval_corpus = make_corpus(per_class=20, length=length,
                              classes=[rhythm_class], rng=rng)

    bittern = Bittern(name=name, seed=seed)

    t0 = time.time()
    for r in range(rounds):
        item = train_corpus[r % len(train_corpus)]
        bittern.listen(item['bits'], train_pairs=4, lr=lr)
        bittern.step()
        bittern.battery.replenish(0.02)

    elapsed = time.time() - t0

    # Single-class eval
    tf_ce = teacher_force_ce(bittern, eval_corpus)
    print(f"  {rhythm_class}: CE={tf_ce:.3f} (ln2={UNIFORM:.3f})  "
          f"{'below uniform' if tf_ce < UNIFORM else 'ABOVE uniform'}  "
          f"({elapsed:.1f}s, {rounds} rounds)")

    # Babble test
    if rhythm_class == 'steady_2':
        prompt = [0, 1, 0, 1, 0, 1]
    elif rhythm_class == 'swelling':
        prompt = [0, 0, 0, 0, 0, 0, 1, 1]
    else:
        prompt = [1, 1, 0, 0, 0, 0, 0, 0]
    bb = bittern.babble(prompt, n=32)
    print(f"  babble: {bits_to_str(prompt)} -> {bits_to_str(bb)}")

    return bittern


def colony_probe(colony, eval_corpus, label=None):
    """Evaluate colony on a mixed corpus.

    For each sequence, reset the colony, feed it tick-by-tick, and
    measure CE using the WINNER's prediction at each step. Reports
    overall colony CE, per-class CE, and winner selection breakdown.
    """
    title = label or "colony"
    print(f"\n  --- colony probe: {title} ---")
    print(f"    uniform-random reference   ln(2) = {UNIFORM:.3f} nats")

    by_class = {}
    winner_counts = {}
    all_specialist_losses = {}

    for item in eval_corpus:
        cls = item['class']
        bits = item['bits']
        colony.reset()

        context = []
        for i, bit in enumerate(bits):
            bit = int(bit)
            if len(context) > 0:
                winner, predictions, losses = colony.tick(bit, context)
                # Colony loss = winner's prediction loss on this bit
                colony_loss = losses[winner]
                by_class.setdefault(cls, []).append(colony_loss)
                winner_counts.setdefault(cls, {}).setdefault(winner, 0)
                winner_counts[cls][winner] += 1

                # Track all specialist losses
                for sname, sloss in losses.items():
                    all_specialist_losses.setdefault(
                        cls, {}).setdefault(sname, []).append(sloss)
            context.append(bit)

    # Overall colony CE
    all_losses = []
    for v in by_class.values():
        all_losses.extend(v)
    colony_ce = float(np.mean(all_losses))
    print(f"    colony_ce (winner-selected)  {colony_ce:.3f}")

    # Per-class
    print(f"    per-class colony CE:")
    for cls in sorted(by_class.keys()):
        ce = float(np.mean(by_class[cls]))
        bar_w = 24
        rel = min(1.0, ce / UNIFORM)
        filled = int(rel * bar_w)
        bar = '█' * filled + '░' * (bar_w - filled)
        marker = '✓' if ce < UNIFORM * 0.5 else '·' if ce < UNIFORM else '✗'
        print(f"      {cls:10s} {marker} {ce:.3f}  [{bar}]")

    # Winner breakdown
    print(f"    winner selection:")
    for cls in sorted(winner_counts.keys()):
        counts = winner_counts[cls]
        total = sum(counts.values())
        parts = []
        for name in sorted(counts.keys()):
            c = counts[name]
            pct = 100 * c / total if total > 0 else 0
            parts.append(f"{name}={pct:.0f}%")
        print(f"      {cls:10s}  {', '.join(parts)}")

    # Individual specialist CE on each class
    print(f"    individual specialist CE:")
    for cls in sorted(all_specialist_losses.keys()):
        parts = []
        for name in sorted(all_specialist_losses[cls].keys()):
            ce = float(np.mean(all_specialist_losses[cls][name]))
            marker = '✓' if ce < UNIFORM else '✗'
            parts.append(f"{name}={ce:.3f}{marker}")
        print(f"      {cls:10s}  {', '.join(parts)}")

    return {'colony_ce': colony_ce,
            'per_class': {cls: float(np.mean(v))
                          for cls, v in by_class.items()}}


def show_colony_babbles(colony, prompts, n=32):
    """Show colony babble with winner annotations."""
    print(f"\n    colony babbles ({n} bits):")
    for prompt_info in prompts:
        cls = prompt_info['class']
        bits = prompt_info['bits']
        bb, winners = colony.babble(bits, n=n)

        # Compress winner tags
        w_str = ''
        for w in winners:
            if w == 'steady_2':
                w_str += '2'
            elif w == 'swelling':
                w_str += 'W'
            else:
                w_str += 'T'

        print(f"      {cls:10s}  prompt={bits_to_str(bits):10s}  "
              f"babble={bits_to_str(bb)}")
        print(f"      {'':10s}  {'':10s}           "
              f"winner={w_str}")


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=65000,
                   help='training rounds PER specialist')
    p.add_argument('--selector-alpha', type=float, default=0.9)
    p.add_argument('--out', default='babies')
    args = p.parse_args()

    classes = ['steady_2', 'swelling', 'staccato']
    specialists = {}

    for i, cls in enumerate(classes):
        print(f"\n--- training specialist: {cls} ---")
        bittern = train_specialist(
            name=f'b_{cls}', rhythm_class=cls,
            seed=args.seed + i, rounds=args.rounds,
        )
        specialists[cls] = bittern

    colony = BitternColony(specialists,
                           selector_alpha=args.selector_alpha)
    print(f"\n=== colony assembled: {len(specialists)} specialists ===")

    # Evaluate on mixed corpus
    rng_eval = np.random.default_rng(args.seed + 999)
    eval_corpus = make_corpus(per_class=20, length=64, rng=rng_eval)

    colony_probe(colony, eval_corpus, label='colony @ 65k per specialist')
    show_colony_babbles(colony, PROBE_PROMPTS)

    # Save specialists
    os.makedirs(args.out, exist_ok=True)
    for cls, bittern in specialists.items():
        path = os.path.join(args.out, f'b_{cls}')
        bittern.save(path)
        print(f"  saved {cls} -> {path}")

    # Also compare to b1 for reference
    try:
        b1 = Bittern.load(f'{args.out}/b1')
        print(f"\n--- b1 reference (round {b1.round}) ---")
        b1_per = per_class_ce(b1, eval_corpus)
        for cls, ce in sorted(b1_per.items()):
            marker = '✓' if ce < UNIFORM * 0.5 else '·' if ce < UNIFORM else '✗'
            print(f"  {cls:10s} {marker} {ce:.3f}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
