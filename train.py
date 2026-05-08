"""
train.py — train one bittern on bit rhythms.

    python train.py                              # birth 'b1', 10000 rounds
    python train.py --rounds 20000               # birth 'b1', 20000 rounds
    python train.py --name nightly --seed 7      # birth 'nightly', seed 7
    python train.py --name b1 --continue         # load 'b1', train MORE
    python train.py --name b1 --continue --rounds 5000   # 5k more rounds

Without --continue: births a fresh bittern (overwrites existing save).
With --continue: loads the saved bittern and trains for `--rounds`
more rounds. The bittern's existing round count is preserved (an
8000-round bittern continued for 5000 more becomes a 13000-round
bittern). Continued training appends to the same save file.

The bittern listens to a rotating corpus of mixed rhythm classes.
Each listen() does Hebbian on lateral + a single backprop step on
the brain (random prefix split, predict next bit). Battery decays
each round and refills slightly from listening — same arrangement
as glaud-i.

We probe four times (untrained baseline, then three checkpoints) so
the trajectory is visible even on a single training run, and we
print babble samples at each checkpoint so you can read what the
bittern is producing without loading and chatting separately.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np

from bittern import Bittern
from world import make_corpus, bits_to_str
from probe import probe_report


PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
    # A few stress-test prompts that aren't in any class:
    {'bits': [0, 1, 1, 0, 0, 1],          'class': 'mixed'},
    {'bits': [1, 0, 0, 0, 1, 1, 1, 0],    'class': 'mixed'},
]


def show_babbles(bittern, prompts, n=24):
    print(f"    babbles ({n} bits each):")
    for prompt in prompts:
        bb = bittern.babble(prompt['bits'], n=n)
        print(f"      {prompt['class']:10s}  "
              f"prompt={bits_to_str(prompt['bits']):10s}  "
              f"babble={bits_to_str(bb)}")


def train(name='b1', seed=0, rounds=10000, out='babies',
          per_class=200, length=64, eval_per_class=20,
          train_pairs=4, lr=0.02, continue_=False,
          balanced=False, self_hear_p=0.0):
    rng = np.random.default_rng(seed)

    print(f"\n=== building corpora (seed={seed}) ===")
    train_corpus = make_corpus(per_class=per_class, length=length, rng=rng)
    eval_corpus = make_corpus(per_class=eval_per_class, length=length, rng=rng)
    print(f"  train: {len(train_corpus)} sequences, {length} bits each")
    print(f"  eval:  {len(eval_corpus)} sequences, {length} bits each")

    save_path = os.path.join(out, name)

    if continue_:
        print(f"\n=== loading {name} for continued training ===")
        bittern = Bittern.load(save_path)
        print(f"  brain_dim={bittern.brain.brain_dim}  "
              f"embed_dim={bittern.brain.embed_dim}  "
              f"context={bittern.brain.context}")
        print(f"  current round: {bittern.round}")
        start_round = bittern.round
        end_round = start_round + rounds
        # Probe current state — this is t=0 for THIS session but t=N globally
        probe_report(bittern, eval_corpus, PROBE_PROMPTS,
                     label=f'{name} resume @ {start_round}')
        show_babbles(bittern, PROBE_PROMPTS)
    else:
        print(f"\n=== birthing {name} ===")
        bittern = Bittern(name=name, seed=seed)
        print(f"  brain_dim={bittern.brain.brain_dim}  "
              f"embed_dim={bittern.brain.embed_dim}  "
              f"context={bittern.brain.context}")
        start_round = 0
        end_round = rounds
        probe_report(bittern, eval_corpus, PROBE_PROMPTS, label='untrained')
        show_babbles(bittern, PROBE_PROMPTS)

    # Training. Checkpoints are computed within THIS session's range.
    print(f"\n=== listening for {rounds} rounds "
          f"(round {start_round} -> {end_round}) ===")
    if balanced:
        print(f"  balanced target sampling: ON")
    if self_hear_p > 0:
        print(f"  self-hearing: p={self_hear_p}")
    checkpoints = sorted(set([
        start_round + max(1, rounds // 4),
        start_round + max(1, rounds // 2),
        start_round + max(1, 3 * rounds // 4),
        end_round,
    ]))
    chk = 0
    t0 = time.time()
    for r in range(rounds):
        item = train_corpus[r % len(train_corpus)]
        bittern.listen(item['bits'], train_pairs=train_pairs, lr=lr,
                       balanced=balanced, self_hear_p=self_hear_p)
        bittern.step()
        bittern.battery.replenish(0.02)  # listening warms the battery

        if bittern.round == checkpoints[chk]:
            elapsed = time.time() - t0
            recent = bittern.recent_brain_losses[-200:]
            avg_loss = float(np.mean(recent)) if recent else 0.0
            print(f"\n  --- checkpoint @ round {bittern.round}  "
                  f"(elapsed {elapsed:.1f}s, "
                  f"recent brain loss {avg_loss:.3f}) ---")
            probe_report(bittern, eval_corpus, PROBE_PROMPTS,
                         label=f'{name} @ {bittern.round}')
            show_babbles(bittern, PROBE_PROMPTS)
            chk = min(chk + 1, len(checkpoints) - 1)

    # Save
    os.makedirs(out, exist_ok=True)
    bittern.save(save_path)
    print(f"\n=== saved {name} -> {save_path}.{{npz,json}} "
          f"(round {bittern.round}) ===")
    return bittern


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='b1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=10000)
    p.add_argument('--out', default='babies')
    p.add_argument('--per-class', type=int, default=200)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--eval-per-class', type=int, default=20)
    p.add_argument('--train-pairs', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--continue', dest='continue_', action='store_true',
                   help='load existing bittern and train for MORE rounds')
    p.add_argument('--balanced', action='store_true',
                   help='balance target bits during training (fix M2)')
    p.add_argument('--self-hear', type=float, default=0.0,
                   help='scheduled sampling probability (0-1)')
    args = p.parse_args()
    train(args.name, args.seed, args.rounds, args.out,
          per_class=args.per_class, length=args.length,
          eval_per_class=args.eval_per_class,
          train_pairs=args.train_pairs, lr=args.lr,
          continue_=args.continue_, balanced=args.balanced,
          self_hear_p=args.self_hear)


if __name__ == '__main__':
    main()
