"""
chat_pipit.py — interactive bit-stream chat with a pipit.

    python chat_pipit.py p1

Type a bit prompt (e.g., 0101 or 11000000) and the pipit will
respond with three babbles. Empty line to quit.
"""

from __future__ import annotations
import argparse

from pipit import Pipit
from world import bits_to_str, str_to_bits


def chat(name, path='babies', n=32, replies=3):
    pipit = Pipit.load(f'{path}/{name}')
    print(f"loaded '{name}' (tick {pipit.round})")
    print(f"  n_osc={pipit.brain.n_osc}")
    R = pipit.brain.entrainment()
    top = R.sum(axis=1).argsort()[::-1][:3]
    for i in top:
        T = abs(2 * 3.14159 / pipit.brain.omega[i])
        print(f"  osc[{i}] T≈{T:.1f} R=[{R[i,0]:.2f},{R[i,1]:.2f}]")
    print(f"  battery: {pipit.battery.level:.2f} "
          f"(lonely: {pipit.battery.is_lonely()})")
    print()
    print("enter bit prompts (e.g., 0101). empty line to quit.")
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
        for _ in range(replies):
            bb = pipit.babble_snapshot(prompt, n=n)
            print(f"    {bits_to_str(bb)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--path', default='babies')
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--replies', type=int, default=3)
    args = p.parse_args()
    chat(args.name, args.path, args.n, args.replies)


if __name__ == '__main__':
    main()
