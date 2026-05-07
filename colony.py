"""
colony.py — a colony of specialist pipits.

Each pipit is trained on a single rhythm class. When the colony hears
input, ALL pipits process it simultaneously. The one with the lowest
recent prediction loss gets to speak — its output IS the colony's
output.

This is competitive entrainment: the pipit that's currently resonating
with the input rhythm wins. No neural network needed, no learned
gating, no attention. The physics of oscillator entrainment does the
class selection automatically.

    python colony.py                    # train + evaluate
    python colony.py --rounds 50000    # longer training

Architecture:

    Colony
      ├── pipit_steady_2   (trained on 010101...)
      ├── pipit_swelling   (trained on long-pause/long-burst)
      └── pipit_staccato   (trained on short-burst/long-gap)

    Each tick:
        1. All pipits hear the same input bit
        2. All pipits compute their prediction loss
        3. The pipit with lowest recent loss is "loudest"
        4. Colony output = loudest pipit's emission

    The selector is just a running average of recent loss per pipit.
    Low loss = good entrainment = I know this rhythm = trust me.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

from pipit import Pipit, VOCAB_SIZE
from world import make_corpus, bits_to_str, CLASSES
from probe_pipit import UNIFORM


class Colony:
    """A colony of specialist pipits with competitive selection.

    Each pipit is a specialist trained on one rhythm class. The colony
    selects the best predictor at each tick based on recent loss.
    """

    def __init__(self, specialists: dict[str, Pipit],
                 selector_alpha: float = 0.9):
        """
        specialists: dict mapping class name -> trained Pipit
        selector_alpha: EMA decay for the loss tracker (higher = longer
                        memory for selection, lower = faster switching)
        """
        self.specialists = specialists
        self.class_names = list(specialists.keys())
        self.selector_alpha = selector_alpha

        # Running loss per specialist (lower = better = wins)
        self.running_loss = {name: float(np.log(2))
                             for name in self.class_names}

        self.round = 0

    def tick(self, input_bit, emit=True):
        """All specialists hear the input. Best predictor speaks.

        Winner is selected based on prior performance (running_loss
        BEFORE this tick). Then running_loss is updated with this
        tick's result. No look-ahead.

        Returns (output_bit, winner_name, predictions_dict, losses_dict).
        """
        input_bit = int(input_bit)
        predictions = {}
        losses = {}

        # All specialists process the input (no learning — they're frozen)
        for name, pipit in self.specialists.items():
            _, probs, loss = pipit.tick(input_bit, learn=False, emit=False)
            predictions[name] = probs
            losses[name] = loss

        # Winner = lowest running loss BEFORE this tick (no look-ahead)
        winner = min(self.class_names, key=lambda n: self.running_loss[n])

        # NOW update running losses with this tick's results
        for name in self.class_names:
            a = self.selector_alpha
            self.running_loss[name] = (a * self.running_loss[name]
                                       + (1 - a) * losses[name])

        # Emit from winner's prediction
        output_bit = None
        if emit:
            probs = predictions[winner]
            temperature = self.specialists[winner].temperature
            if temperature < 1e-6:
                output_bit = int(np.argmax(probs))
            else:
                logits = np.log(probs + 1e-12) / temperature
                logits = logits - logits.max()
                e = np.exp(logits)
                sp = e / e.sum()
                rng = self.specialists[winner].rng
                output_bit = int(rng.choice(VOCAB_SIZE, p=sp))

        self.round += 1
        return output_bit, winner, predictions, losses

    def babble(self, prompt, n=32):
        """Process prompt, then generate n bits using competitive selection."""
        for bit in prompt:
            self.tick(int(bit), emit=False)
        out = []
        winners = []
        last_bit = int(prompt[-1]) if prompt else 0
        for _ in range(n):
            output, winner, _, _ = self.tick(last_bit, emit=True)
            out.append(output)
            winners.append(winner)
            last_bit = output
        return out, winners

    def babble_snapshot(self, prompt, n=32):
        """Babble without modifying any specialist's state."""
        # Snapshot all specialists
        snaps = {}
        rng_states = {}
        for name, pipit in self.specialists.items():
            snaps[name] = pipit.brain._snapshot()
            rng_states[name] = pipit.rng.bit_generator.state
        loss_snap = dict(self.running_loss)
        round_snap = self.round

        out, winners = self.babble(prompt, n=n)

        # Restore
        for name, pipit in self.specialists.items():
            pipit.brain._restore(snaps[name])
            pipit.rng.bit_generator.state = rng_states[name]
        self.running_loss = loss_snap
        self.round = round_snap
        return out, winners

    def reset_all_phases(self, rng=None):
        """Reset all specialists' phases and running losses."""
        for pipit in self.specialists.values():
            pipit.brain.reset_phases(rng)
        self.running_loss = {name: float(np.log(2))
                             for name in self.class_names}


# ----------------------------------------------------------------------
#  Training: birth specialists
# ----------------------------------------------------------------------

def train_specialist(name, rhythm_class, n_osc=32, seed=0,
                     rounds=10000, per_class=200, length=64):
    """Train a single pipit on one rhythm class."""
    rng = np.random.default_rng(seed)
    corpus = make_corpus(per_class=per_class, length=length,
                         classes=[rhythm_class], rng=rng)

    pipit = Pipit(name=name, n_osc=n_osc, memory_mode='ema',
                  mem_alpha=0.97, seed=seed)

    for r in range(rounds):
        item = corpus[r % len(corpus)]
        pipit.brain.reset_phases(pipit.rng)
        for bit in item['bits']:
            pipit.tick(int(bit), learn=True, emit=False)
        pipit.battery.replenish(0.02)

    return pipit


def train_colony(n_osc=32, seed=0, rounds=10000, per_class=200,
                 length=64, selector_alpha=0.9,
                 classes=None, out='babies'):
    """Train one specialist per rhythm class, assemble a colony."""
    if classes is None:
        classes = ['steady_2', 'swelling', 'staccato']

    specialists = {}
    for i, cls in enumerate(classes):
        print(f"\n--- training specialist: {cls} ---")
        t0 = time.time()
        pipit = train_specialist(
            name=f'spec_{cls}', rhythm_class=cls,
            n_osc=n_osc, seed=seed + i,
            rounds=rounds, per_class=per_class, length=length,
        )
        elapsed = time.time() - t0

        # Quick single-class eval
        rng_eval = np.random.default_rng(seed + 100 + i)
        eval_corpus = make_corpus(per_class=20, length=length,
                                  classes=[cls], rng=rng_eval)
        losses = []
        for item in eval_corpus:
            pipit.brain.reset_phases(pipit.rng)
            for bit in item['bits']:
                _, loss = pipit.brain.tick(bit, learn=False)
                losses.append(loss)
        ce = float(np.mean(losses))
        print(f"  {cls}: CE={ce:.3f} (ln2={UNIFORM:.3f})  "
              f"{'✓ below uniform' if ce < UNIFORM else '✗ above uniform'}  "
              f"({elapsed:.1f}s)")

        # Babble test
        if cls == 'steady_2':
            prompt = [0, 1, 0, 1, 0, 1]
        elif cls == 'swelling':
            prompt = [0, 0, 0, 0, 0, 0, 1, 1]
        else:
            prompt = [1, 1, 0, 0, 0, 0, 0, 0]
        bb = pipit.babble_snapshot(prompt, n=32)
        print(f"  babble: {bits_to_str(prompt)} → {bits_to_str(bb)}")

        specialists[cls] = pipit

    colony = Colony(specialists, selector_alpha=selector_alpha)
    print(f"\n=== colony assembled: {len(specialists)} specialists ===")
    return colony


# ----------------------------------------------------------------------
#  Colony evaluation
# ----------------------------------------------------------------------

def colony_probe(colony, eval_corpus, label=None):
    """Evaluate the colony on a mixed corpus.

    For each sequence:
        - Reset all specialists' phases
        - Feed the sequence tick by tick
        - Track which specialist wins at each tick
        - Measure loss using the WINNER's prediction

    Reports:
        - Colony CE (using winner's prediction)
        - Per-class colony CE
        - Winner breakdown (which specialist wins for each class)
    """
    title = label or f"colony @ tick {colony.round}"
    print(f"\n  --- colony probe: {title} ---")
    print(f"    uniform-random reference   ln(2) = {UNIFORM:.3f} nats")

    by_class = {}       # class -> list of losses
    by_class_raw = {}   # class -> {specialist_name -> list of losses}
    winner_counts = {}  # class -> {specialist_name -> count}

    for item in eval_corpus:
        cls = item['class']
        bits = item['bits']

        colony.reset_all_phases()

        for bit in bits:
            _, winner, predictions, losses = colony.tick(int(bit), emit=False)

            # Colony loss = winner's loss
            colony_loss = losses[winner]
            by_class.setdefault(cls, []).append(colony_loss)

            # Track per-specialist losses (for comparison)
            for name, loss in losses.items():
                by_class_raw.setdefault(cls, {}).setdefault(name, []).append(loss)

            # Track winner
            winner_counts.setdefault(cls, {}).setdefault(winner, [0])[0] += 1

    # Colony CE
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
    print(f"    winner selection (who speaks for each class):")
    for cls in sorted(winner_counts.keys()):
        counts = winner_counts[cls]
        total = sum(v[0] for v in counts.values())
        parts = []
        for name in sorted(counts.keys()):
            c = counts[name][0]
            pct = 100 * c / total if total > 0 else 0
            parts.append(f"{name}={pct:.0f}%")
        print(f"      {cls:10s}  {', '.join(parts)}")

    # Per-specialist CE (what each specialist alone would score)
    print(f"    individual specialist CE (for reference):")
    for cls in sorted(by_class_raw.keys()):
        parts = []
        for name in sorted(by_class_raw[cls].keys()):
            ce = float(np.mean(by_class_raw[cls][name]))
            marker = '✓' if ce < UNIFORM else '✗'
            parts.append(f"{name}={ce:.3f}{marker}")
        print(f"      {cls:10s}  {', '.join(parts)}")

    return {'colony_ce': colony_ce, 'per_class': {
        cls: float(np.mean(v)) for cls, v in by_class.items()}}


# ----------------------------------------------------------------------
#  Colony babble display
# ----------------------------------------------------------------------

def show_colony_babbles(colony, prompts, n=32):
    print(f"    colony babbles ({n} bits, winner in brackets):")
    for prompt_info in prompts:
        cls = prompt_info['class']
        bits = prompt_info['bits']
        colony.reset_all_phases()
        bb, winners = colony.babble_snapshot(bits, n=n)

        # Compress winner sequence
        w_str = ''
        prev = None
        for w in winners:
            tag = w[0].upper()  # first letter: S/s for steady/swelling/staccato
            if w == 'steady_2':
                tag = '2'
            elif w == 'swelling':
                tag = 'W'
            else:
                tag = 'T'
            w_str += tag

        print(f"      {cls:10s}  prompt={bits_to_str(bits):10s}  "
              f"babble={bits_to_str(bb)}")
        print(f"      {'':10s}  {'':10s}           "
              f"winner={w_str}")


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
    {'bits': [0, 1, 1, 0, 0, 1],          'class': 'mixed'},
    {'bits': [1, 0, 0, 0, 1, 1, 1, 0],    'class': 'mixed'},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-osc', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=10000,
                   help='training rounds PER specialist')
    p.add_argument('--selector-alpha', type=float, default=0.9,
                   help='EMA decay for winner selection (0.9 = ~10 tick memory)')
    p.add_argument('--out', default='babies')
    args = p.parse_args()

    # Train specialists
    colony = train_colony(
        n_osc=args.n_osc, seed=args.seed, rounds=args.rounds,
        selector_alpha=args.selector_alpha, out=args.out,
    )

    # Evaluate on mixed corpus
    rng_eval = np.random.default_rng(args.seed + 999)
    eval_corpus = make_corpus(per_class=20, length=64, rng=rng_eval)
    colony_probe(colony, eval_corpus)
    show_colony_babbles(colony, PROBE_PROMPTS)

    # Save specialists
    os.makedirs(args.out, exist_ok=True)
    for name, pipit in colony.specialists.items():
        path = os.path.join(args.out, f'colony_{name}')
        pipit.save(path)
        print(f"  saved {name} -> {path}")


if __name__ == '__main__':
    main()
