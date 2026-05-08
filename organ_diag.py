"""
organ_diag.py — diagnostic for organ-brain alignment.

Tests whether the organs are actually transmitting brain signal during
babble. In v1 the answer was no — organs were ANTI-correlated with the
brain (3.5% agreement, where 50% would be random). After fixing the
organ dynamics and adding supervised sensitivity learning to listen(),
agreement is 100%.

Run on any saved bittern:

    python organ_diag.py b2

Reports three babbles per prompt:
    organs       - normal path
    brain sample - bypass organs, sample from brain.predict_probs
    brain argmax - bypass organs, take brain's most likely bit

Plus a single agreement number: how often organ choice matches brain
argmax over 200 ticks. Pre-fix: ~3-5% (anti-correlated). Post-fix: 100%.

If this number drifts away from 100%, the organs have stopped tracking
the brain and there's a regression.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np

from bittern import Bittern


def s(bits):
    return ''.join(str(b) for b in bits)


def diag(name, path='babies'):
    bittern = Bittern.load(f'{path}/{name}')
    print(f"loaded '{name}' (round {bittern.round})")
    print(f"  brain_dim={bittern.brain.brain_dim} "
          f"embed_dim={bittern.brain.embed_dim} "
          f"context={bittern.brain.context}")
    print()

    prompts = [
        ('010101',   'steady_2'),
        ('00000011', 'swelling'),
        ('11000000', 'staccato'),
        ('011001',   'mixed'),
        ('10001110', 'mixed'),
    ]
    print("Three babble modes side-by-side:")
    for prompt_str, label in prompts:
        prompt = [int(c) for c in prompt_str]
        # Reset rng each time so outputs are comparable across modes
        bittern.rng = np.random.default_rng(123)
        bb_org = bittern.babble(prompt, n=32, mode='organs')
        bittern.rng = np.random.default_rng(123)
        bb_smp = bittern.babble(prompt, n=32, mode='brain_sample')
        bittern.rng = np.random.default_rng(123)
        bb_arg = bittern.babble(prompt, n=32, mode='brain_argmax')
        print(f"\n  prompt {prompt_str}  ({label}):")
        print(f"    organs       : {s(bb_org)}")
        print(f"    brain sample : {s(bb_smp)}")
        print(f"    brain argmax : {s(bb_arg)}")

    # Agreement statistic
    matches = 0
    total = 200
    context = [0, 1, 0, 1]
    for _ in range(total):
        window = context[-bittern.brain.context:]
        bs, _ = bittern.brain.encode(window)
        probs = bittern.brain.predict_probs(window)
        brain_argmax = int(np.argmax(probs))
        organ_choice, _ = bittern.organs.fire(
            bs, context[-1], np.zeros(2), 0.0,
            np.random.default_rng(0))
        if organ_choice == brain_argmax:
            matches += 1
        context.append(organ_choice)
    agreement = matches / total
    print(f"\nOrgan-vs-brain-argmax agreement: "
          f"{matches}/{total} = {agreement:.1%}")
    print(f"  (random chance = 50%, post-fix should be ~100%)")
    if agreement < 0.7:
        print(f"  ⚠ LOW AGREEMENT — organs may have decoupled from brain")
    return agreement


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--path', default='babies')
    args = p.parse_args()
    diag(args.name, args.path)


if __name__ == '__main__':
    main()
