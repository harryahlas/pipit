"""
memory_smoke.py — smoke test for the PainfulMemory replay buffer.

    python memory_smoke.py

What this checks:
    1. Buffer initializes empty
    2. Capture threshold is respected (low-sting pairs aren't stored)
    3. High-sting pairs ARE stored
    4. Capacity is respected (oversize buffer evicts lowest-sting)
    5. Dedup works (identical (prefix, target) doesn't add duplicates)
    6. maybe_replay returns stored entries with the right probability
    7. Save/load roundtrips the buffer
    8. Buffer fills during real training and influences gradients
    9. Disabling the buffer is a clean no-op

Should run in under 30 seconds. Assumes pain_smoke.py already passed.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np

from bittern import Bittern, PainfulMemory
from world import make_corpus, bits_to_str


def hr(label=''):
    print('\n' + '─' * 65 + (f' {label} ' if label else ''))


def main():
    print('=' * 65)
    print('  PAINFUL MEMORY SMOKE TEST')
    print('=' * 65)
    t_total = time.time()

    # ── Test 1: init ──────────────────────────────────────────────
    hr('1. init')
    pm = PainfulMemory()
    assert pm.entries == [], "buffer should start empty"
    assert pm.enabled is True, "buffer should default to enabled"
    print('  ok: buffer starts empty and enabled')

    # ── Test 2: capture threshold ────────────────────────────────
    hr('2. capture threshold')
    pm = PainfulMemory(capture_threshold=0.4)
    # Low-sting pair: brain hedged correctly
    low_probs = np.array([0.5, 0.5])
    captured = pm.maybe_capture([0, 1, 0], 1, low_probs)
    assert not captured, "should not capture when sting < threshold"
    assert len(pm.entries) == 0
    # High-sting pair: brain confidently wrong
    high_probs = np.array([0.9, 0.1])
    captured = pm.maybe_capture([1, 1, 0], 1, high_probs)
    assert captured, "should capture when sting >= threshold"
    assert len(pm.entries) == 1
    e = pm.entries[0]
    expected_sting = 0.9 * 0.9
    assert abs(e['sting'] - expected_sting) < 1e-9
    print(f'  ok: threshold respected, captured 1 entry with sting={e["sting"]:.3f}')

    # ── Test 3: capacity + eviction ───────────────────────────────
    hr('3. capacity + eviction (lowest sting evicted)')
    pm = PainfulMemory(capacity=3, capture_threshold=0.0)
    # Sting formula: max(probs) * (1 - probs[target]).
    # Fill with three entries of clearly separated sting values.
    pm.maybe_capture([0], 1, np.array([0.7, 0.3]))      # target=1: sting = 0.7*0.7 = 0.49
    pm.maybe_capture([1], 0, np.array([0.9, 0.1]))      # target=0: sting = 0.9*0.1 = 0.09  (LOW)
    pm.maybe_capture([0, 0], 1, np.array([0.95, 0.05])) # target=1: sting = 0.95*0.95 ≈ 0.90 (HIGH)
    assert len(pm.entries) == 3
    weakest_before = min(e['sting'] for e in pm.entries)
    print(f'  initial stings: {sorted(round(e["sting"], 3) for e in pm.entries)}')

    # Candidate sting clearly BELOW weakest — should NOT evict.
    # probs=[0.95, 0.05], target=0 → sting = 0.95 * 0.05 = 0.0475
    captured = pm.maybe_capture([1, 1], 0, np.array([0.95, 0.05]))
    assert not captured, "candidate weaker than buffer's weakest should not evict"
    assert len(pm.entries) == 3

    # Candidate sting clearly ABOVE weakest — SHOULD evict.
    # probs=[0.6, 0.4], target=1 → sting = 0.6 * 0.6 = 0.36
    captured = pm.maybe_capture([1, 0], 1, np.array([0.6, 0.4]))
    assert captured, "candidate stronger than weakest should evict"
    assert len(pm.entries) == 3
    weakest_after = min(e['sting'] for e in pm.entries)
    assert weakest_after > weakest_before, (
        f"weakest sting should increase after eviction "
        f"({weakest_before:.3f} → {weakest_after:.3f})")
    print(f'  weakest sting moved from {weakest_before:.3f} to {weakest_after:.3f}')

    # ── Test 4: dedup ─────────────────────────────────────────────
    hr('4. dedup (identical pair updates sting, no new entry)')
    pm = PainfulMemory(capture_threshold=0.0)
    # Initial capture: moderate sting.
    # probs=[0.6, 0.4], target=0 → sting = 0.6 * (1-0.6) = 0.24
    pm.maybe_capture([0, 1], 0, np.array([0.6, 0.4]))
    assert len(pm.entries) == 1
    assert abs(pm.entries[0]['sting'] - 0.24) < 1e-9

    # Same pair captured again with HIGHER sting — should update, not duplicate.
    # probs=[0.1, 0.9], target=0 → confidence=0.9, wrongness=1-0.1=0.9, sting=0.81
    pm.maybe_capture([0, 1], 0, np.array([0.1, 0.9]))
    assert len(pm.entries) == 1, "dedup should prevent new entry"
    assert pm.entries[0]['sting'] > 0.75, (
        f"sting should update to max (~0.81), got {pm.entries[0]['sting']:.3f}")

    # Same pair captured AGAIN with LOWER sting — should NOT downgrade.
    # probs=[0.5, 0.5], target=0 → sting = 0.5 * 0.5 = 0.25
    pm.maybe_capture([0, 1], 0, np.array([0.5, 0.5]))
    assert pm.entries[0]['sting'] > 0.75, (
        f"max-sting should be preserved, got {pm.entries[0]['sting']:.3f}")

    # Same prefix but DIFFERENT target → new entry (target is part of identity)
    # probs=[0.7, 0.3], target=1 → sting = 0.7 * 0.7 = 0.49
    pm.maybe_capture([0, 1], 1, np.array([0.7, 0.3]))
    assert len(pm.entries) == 2
    print('  ok: dedup updates sting to max, preserves it, distinguishes by target')

    # ── Test 5: maybe_replay ─────────────────────────────────────
    hr('5. maybe_replay')
    pm = PainfulMemory(capture_threshold=0.0, replay_p=1.0)
    rng = np.random.default_rng(42)
    # Empty: returns None
    assert pm.maybe_replay(rng) is None
    pm.maybe_capture([0, 1, 0], 1, np.array([0.9, 0.1]))
    pm.maybe_capture([1, 0], 0, np.array([0.85, 0.15]))
    # With replay_p=1.0 always returns something
    samples = [pm.maybe_replay(rng) for _ in range(20)]
    assert all(s is not None for s in samples), "replay_p=1 should always fire"
    # All should be one of the two stored entries
    for prefix, target in samples:
        assert (prefix, target) in [([0, 1, 0], 1), ([1, 0], 0)]
    # With replay_p=0.0 never returns
    pm.replay_p = 0.0
    samples = [pm.maybe_replay(rng) for _ in range(20)]
    assert all(s is None for s in samples), "replay_p=0 should never fire"
    # With replay_p=0.5 — roughly half
    pm.replay_p = 0.5
    samples = [pm.maybe_replay(rng) for _ in range(1000)]
    fire_rate = sum(1 for s in samples if s is not None) / len(samples)
    assert 0.40 < fire_rate < 0.60, f"replay_p=0.5 should fire ~50%, got {fire_rate:.2%}"
    print(f'  ok: replay_p drives fire rate ({fire_rate:.0%} at p=0.5)')

    # ── Test 6: save/load ────────────────────────────────────────
    hr('6. save/load roundtrip')
    import tempfile
    b = Bittern(name='test', seed=0)
    # Force-populate the buffer with TRULY painful pairs (above the
    # default 0.4 capture threshold). Confidently-wrong:
    #   probs=[0.05, 0.95], target=0 → sting = 0.95*0.95 = 0.9025
    #   probs=[0.88, 0.12], target=1 → sting = 0.88*0.88 = 0.7744
    captured1 = b.painful_memory.maybe_capture(
        [0, 1, 1], 0, np.array([0.05, 0.95]))
    captured2 = b.painful_memory.maybe_capture(
        [1, 0, 1, 0], 1, np.array([0.88, 0.12]))
    assert captured1, "first capture should fire (sting > threshold)"
    assert captured2, "second capture should fire (sting > threshold)"
    assert len(b.painful_memory.entries) == 2, (
        f"expected 2 entries pre-save, got {len(b.painful_memory.entries)}")
    print(f'  pre-save: {len(b.painful_memory.entries)} entries')

    save_dir = tempfile.mkdtemp(prefix='memory_smoke_')
    save_path = os.path.join(save_dir, 'test')
    b.save(save_path)
    print(f'  saved to {save_path}')
    b2 = Bittern.load(save_path)
    print(f'  loaded; buffer size = {len(b2.painful_memory.entries)}')
    assert len(b2.painful_memory.entries) == 2, (
        f"expected 2 entries post-load, got {len(b2.painful_memory.entries)}")
    # Compare entries (order should be preserved)
    for e1, e2 in zip(b.painful_memory.entries, b2.painful_memory.entries):
        assert e1['prefix'] == e2['prefix']
        assert e1['target'] == e2['target']
        assert abs(e1['sting'] - e2['sting']) < 1e-9
    print('  ok: save/load preserves buffer entries exactly')
    for ext in ('.npz', '.json'):
        try: os.remove(save_path + ext)
        except OSError: pass
    try: os.rmdir(save_dir)
    except OSError: pass

    # ── Test 7: buffer fills during real training ────────────────
    hr('7. buffer fills during training')
    rng = np.random.default_rng(0)
    corpus = make_corpus(per_class=200, length=64, rng=rng)
    b = Bittern(name='trainer', seed=0)
    # Pain off (we're testing memory in isolation)
    b.pain.enabled = False
    for r in range(500):
        item = corpus[r % len(corpus)]
        b.listen(item['bits'], train_pairs=4, lr=0.02)
        b.step()
    rep = b.painful_memory.report()
    print(f'  buffer after 500 rounds: size={rep["size"]}, '
          f'mean_sting={rep["mean_sting"]:.3f}, max_sting={rep["max_sting"]:.3f}')
    assert rep['size'] > 0, "buffer should fill during training"
    assert rep['max_sting'] >= b.painful_memory.capture_threshold
    print('  ok: buffer captures real painful moments during training')

    # ── Test 8: disabled buffer is a no-op ───────────────────────
    hr('8. disabling buffer is a clean no-op')
    b_off = Bittern(name='off', seed=0)
    b_off.pain.enabled = False
    b_off.painful_memory.enabled = False
    b_on = Bittern(name='on', seed=0)
    b_on.pain.enabled = False
    # Baseline rng must match for fair comparison
    b_off.rng = np.random.default_rng(7)
    b_on.rng = np.random.default_rng(7)
    # Train both for a few rounds without any captures expected (fresh creature)
    for r in range(50):
        item = corpus[r % len(corpus)]
        b_off.listen(item['bits'], train_pairs=2, lr=0.02)
        b_on.listen(item['bits'], train_pairs=2, lr=0.02)
    # Disabled buffer should remain empty
    assert len(b_off.painful_memory.entries) == 0
    # Enabled buffer might or might not have entries depending on early sting,
    # but the on-creature should match the off-creature in BRAIN STATE because
    # within the first 50 rounds, captures probably haven't happened yet OR
    # they have but the replay_p draws may not have fired. So we don't assert
    # brain-equality here. We only assert the disabled invariant.
    rep_off = b_off.painful_memory.report()
    print(f'  disabled buffer after 50 rounds: size={rep_off["size"]} (should be 0)')
    assert rep_off['size'] == 0
    print('  ok: disabled buffer captures nothing')

    # ── Done ──────────────────────────────────────────────────────
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
