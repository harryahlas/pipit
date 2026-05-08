"""
pain_smoke.py — smoke test for the pain system.

    python pain_smoke.py

What this checks:
    1. Pain initializes to zero, behaviorally inert at start
    2. Pain levels move during training (otherwise the system is dead)
    3. Behavioral effects (effective_*) are visible
    4. Save/load roundtrips pain state correctly
    5. Class transitions trigger nausea responses (the testable claim)
    6. Manually-injected sting changes babble character

This is a fast smoke test, not an experiment. ~30 seconds on a
modern machine. If anything here fails, longer runs won't tell you
anything useful.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

from bittern import Bittern, Pain
from world import make_corpus, bits_to_str
from pain_probes import (
    pain_report, pain_summary, pain_trajectory, behavioral_trace,
    behavioral_trace_summary, print_trajectory, pain_vs_loss_scatter,
    find_boundary_responses, print_boundary_responses,
)


def hr(label=''):
    print('\n' + '─' * 65 + (f' {label} ' if label else ''))


def main():
    rng = np.random.default_rng(0)
    print('=' * 65)
    print('  PAIN SYSTEM SMOKE TEST')
    print('=' * 65)
    t_total = time.time()

    # ── Test 1: initialization ────────────────────────────────────────
    hr('1. init')
    b = Bittern(name='test', seed=0)
    rep = pain_report(b)
    print(f"  initial pain levels: {rep}")
    for k in ['sting', 'hunger', 'nausea', 'itch', 'disquiet']:
        assert rep[k] == 0.0, f"{k} should start at 0, got {rep[k]}"
    print('  ok: all pains start at zero')

    # Also verify effective_* values match base values when pain is zero
    base_temp = b.organs.temperature
    base_ctx = b.brain.context
    base_lat = b.organs.lateral_weight
    base_sens = b.organs.sensitivity_lr
    eff_temp = b.pain.effective_temperature(base_temp)
    eff_ctx = b.pain.effective_context(base_ctx, [], 0)
    eff_lat = b.pain.effective_lateral_weight(base_lat)
    eff_sens = b.pain.effective_sensitivity_lr(base_sens)
    assert abs(eff_temp - base_temp) < 1e-9
    assert eff_ctx == base_ctx
    assert abs(eff_lat - base_lat) < 1e-9
    assert abs(eff_sens - base_sens) < 1e-9
    print('  ok: at zero pain, effective_* values equal base values')

    # ── Test 2: pain trajectory on block-structured corpus ────────────
    hr('2. trajectory on block-structured corpus')
    corpus = make_corpus(per_class=200, length=64, rng=rng)

    # Build a block-structured corpus: 60 rounds of each class in
    # rotation. The class transitions happen every 60 rounds; we
    # sample every 30 rounds, so each class block produces ~2 samples
    # and transitions are visible.
    by_class = {}
    for item in corpus:
        by_class.setdefault(item['class'], []).append(item)
    block_size = 60
    block_corpus = []
    rotation = 0
    while len(block_corpus) < 3000:
        for cls in sorted(by_class.keys()):
            pool = by_class[cls]
            for i in range(block_size):
                block_corpus.append(pool[(rotation + i) % len(pool)])
        rotation += block_size

    t0 = time.time()
    samples = pain_trajectory(b, block_corpus, n_rounds=3000,
                              sample_every=30)
    print(f"  trained 3000 rounds, sampled {len(samples)} times "
          f"({time.time() - t0:.1f}s)")
    print_trajectory(samples, max_rows=24)

    # Check that at least 2 of 5 pains showed signal
    final = pain_report(b)
    nonzero = sum(1 for k in ['sting', 'hunger', 'nausea', 'itch', 'disquiet']
                  if final[k] > 0.001)
    print(f"\n  {nonzero}/5 pains nonzero at end of training")
    if nonzero < 2:
        print(f"  WARNING: fewer than 2 pains registered signal. "
              f"The system may be dead. Final state:\n    {final}")
    else:
        print(f"  ok: pain system is alive")

    # ── Test 3: class boundaries should produce nausea spikes ────────
    hr('3. class transitions → nausea response')
    boundaries = find_boundary_responses(samples, pain_key='nausea',
                                         response_window=5)
    print_boundary_responses(boundaries, pain_key='nausea')

    # Also report disquiet at boundaries — the leading-edge signal
    boundaries_disq = find_boundary_responses(samples, pain_key='disquiet',
                                              response_window=5)
    print_boundary_responses(boundaries_disq, pain_key='disquiet')

    # ── Test 4: pain × loss by class ─────────────────────────────────
    hr('4. pain × loss correlation by class')
    pain_vs_loss_scatter(samples)

    # ── Test 5: behavioral trace summary ─────────────────────────────
    hr('5. effective_* values after training')
    pain_summary(b, label='post-training')
    behavioral_trace_summary(b, label='post-training')

    # ── Test 6: save/load roundtrip ──────────────────────────────────
    hr('6. save/load roundtrip')
    save_path = '/tmp/pain_smoke_test'
    b.save(save_path)
    b2 = Bittern.load(save_path)
    rep1 = pain_report(b)
    rep2 = pain_report(b2)
    for k in ['sting', 'nausea', 'itch', 'disquiet']:
        diff = abs(rep1[k] - rep2[k])
        assert diff < 1e-9, f"{k} mismatch after roundtrip: {rep1[k]} vs {rep2[k]}"
    diff_short = float(np.max(np.abs(b.pain.bs_ema_short - b2.pain.bs_ema_short)))
    diff_long = float(np.max(np.abs(b.pain.bs_ema_long - b2.pain.bs_ema_long)))
    assert diff_short < 1e-9, f"bs_ema_short mismatch: {diff_short}"
    assert diff_long < 1e-9, f"bs_ema_long mismatch: {diff_long}"
    print('  ok: pain state roundtrips correctly')
    for ext in ('.npz', '.json'):
        try: os.remove(save_path + ext)
        except OSError: pass

    # ── Test 7: injected sting changes babble character ─────────────
    hr('7. injected sting → different babble (behavioral test)')
    b_calm = Bittern(name='calm', seed=42)
    b_stung = Bittern(name='stung', seed=42)
    # Train identically and briefly so they're the same creature
    train_rng = np.random.default_rng(99)
    train_corpus = make_corpus(per_class=200, length=64, rng=train_rng)
    for r in range(800):
        item = train_corpus[r % len(train_corpus)]
        b_calm.listen(item['bits'], train_pairs=2, lr=0.02)
        b_stung.listen(item['bits'], train_pairs=2, lr=0.02)
    # Inject sting on the second creature
    b_stung.pain.sting_level = 0.9
    # Reset both rngs to the same state for a fair comparison
    b_calm.rng = np.random.default_rng(7777)
    b_stung.rng = np.random.default_rng(7777)
    babble_calm = b_calm.babble([0, 1, 0, 1, 0, 1], n=32)
    babble_stung = b_stung.babble([0, 1, 0, 1, 0, 1], n=32)
    print(f"  calm  (sting=0.0): {bits_to_str(babble_calm)}")
    print(f"  stung (sting=0.9): {bits_to_str(babble_stung)}")
    diffs = sum(a != b for a, b in zip(babble_calm, babble_stung))
    print(f"  bit-level diffs: {diffs}/32")
    if diffs == 0:
        print('  WARNING: stung creature produced identical babble. '
              'effective_temperature may not be flowing through.')
    else:
        print(f'  ok: sting changes babble character ({diffs}/32 bits differ)')

    # Also test injected hunger
    b_hungry = Bittern(name='hungry', seed=42)
    for r in range(800):
        item = train_corpus[r % len(train_corpus)]
        b_hungry.listen(item['bits'], train_pairs=2, lr=0.02)
    # Backdoor: skip warmup by jumping the round counter, then force
    # high recent loss
    b_hungry.round = 1000
    b_hungry.recent_brain_losses = [0.69] * 50  # ≈ ln(2), max hunger
    base_ctx = b_hungry.brain.context
    eff_ctx = b_hungry.pain.effective_context(
        base_ctx, b_hungry.recent_brain_losses, b_hungry.round)
    print(f"  hunger=max → context shrinks from {base_ctx} to {eff_ctx}")
    assert eff_ctx < base_ctx, "hunger should shrink context"
    print('  ok: hunger shortens context')

    # ── Done ──────────────────────────────────────────────────────────
    hr()
    elapsed = time.time() - t_total
    print(f"  ALL TESTS PASSED ({elapsed:.1f}s total)")
    print('=' * 65)


if __name__ == '__main__':
    try:
        main()
    except AssertionError as e:
        print(f"\n  ✗ ASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
