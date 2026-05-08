"""
chat.py — interactive bit-stream chat with a bittern.

    python chat.py b1
    python chat.py b1 --path babies

Type a bit prompt (e.g., 0101 or 11000000) and the bittern will
babble three responses to it. Empty line to quit.
"""

from __future__ import annotations
import argparse

from bittern import Bittern
from world import bits_to_str, str_to_bits


def chat(name, path='babies', n=32, replies=3):
    bittern = Bittern.load(f'{path}/{name}')
    print(f"loaded '{name}' (round {bittern.round})")
    print(f"  brain_dim={bittern.brain.brain_dim} "
          f"embed_dim={bittern.brain.embed_dim} "
          f"context={bittern.brain.context}")
    print(f"  battery: {bittern.battery.level:.2f} "
          f"(lonely: {bittern.battery.is_lonely()})")
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
            print("  (no bits found — enter only 0s and 1s)")
            continue
        for _ in range(replies):
            bb = bittern.babble(prompt, n=n)
            print(f"    {bits_to_str(bb)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--path', default='babies')
    p.add_argument('--n', type=int, default=32,
                   help='bits to babble per response')
    p.add_argument('--replies', type=int, default=3,
                   help='number of independent samples per prompt')
    args = p.parse_args()
    chat(args.name, args.path, args.n, args.replies)


if __name__ == '__main__':
    main()
