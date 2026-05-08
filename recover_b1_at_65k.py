"""
recover_b1_at_65k.py — restore the 65k checkpoint.

The b1 bittern was trained past its sweet spot (resumed past 65k,
collapsed to constant-output attractors by 100k+). This script
overwrites babies/b1 with a fresh 65k bittern using the same defaults
as the original run, so the good version is back on disk.

    python recover_b1_at_65k.py

Same seed, same corpus params, same training loop as `python train.py
--name b1 --rounds 65000`. Should take ~3 minutes.
"""
import sys
sys.path.insert(0, '.')

from train import train

if __name__ == '__main__':
    print("Recovering b1 at round 65000 (overwrites babies/b1)...")
    train(name='b1', seed=0, rounds=65000)
    print("\nDone. Verify with:")
    print("    python organ_diag.py b1     # should report ~100% agreement")
    print("    python chat.py b1           # 0101 prompts should alternate")
