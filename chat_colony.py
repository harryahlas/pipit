"""
chat_colony.py — interactive chat with a colony of specialists.

    python chat_colony.py

Loads the three specialists from babies/colony_*, assembles a colony,
and lets you type bit prompts. Shows which specialist wins at each
tick so you can see the colony switching between experts.
"""

from __future__ import annotations
import argparse
import os

from pipit import Pipit
from world import bits_to_str, str_to_bits
from colony import Colony


def load_colony(path='babies', selector_alpha=0.9):
    """Load colony specialists from saved files."""
    classes = ['steady_2', 'swelling', 'staccato']
    specialists = {}
    for cls in classes:
        fpath = os.path.join(path, f'colony_{cls}')
        if os.path.exists(fpath + '.json'):
            specialists[cls] = Pipit.load(fpath)
            print(f"  loaded {cls} (tick {specialists[cls].round})")
        else:
            print(f"  ⚠ {fpath} not found — skipping {cls}")
    if not specialists:
        print("No specialists found. Run `python colony.py` first.")
        return None
    return Colony(specialists, selector_alpha=selector_alpha)


def chat(path='babies', n=32, selector_alpha=0.9):
    print(f"loading colony from {path}/colony_*")
    colony = load_colony(path, selector_alpha)
    if colony is None:
        return

    print(f"\n  specialists: {', '.join(colony.class_names)}")
    print(f"  selector_alpha: {selector_alpha}")
    print()
    print("enter bit prompts (e.g., 0101). empty line to quit.")
    print("output shows: babble bits + which specialist won each tick")
    print("  2=steady_2  W=swelling  T=staccato")

    while True:
        try:
            line = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        prompt = str_to_bits(line)
        if not prompt:
            print("  (no bits found)")
            continue

        colony.reset_all_phases()
        bb, winners = colony.babble_snapshot(prompt, n=n)

        w_str = ''
        for w in winners:
            if w == 'steady_2':
                w_str += '2'
            elif w == 'swelling':
                w_str += 'W'
            else:
                w_str += 'T'

        print(f"    bits:   {bits_to_str(bb)}")
        print(f"    winner: {w_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', default='babies')
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--selector-alpha', type=float, default=0.9)
    args = p.parse_args()
    chat(args.path, args.n, args.selector_alpha)


if __name__ == '__main__':
    main()
