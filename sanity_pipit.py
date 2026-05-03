"""
sanity_pipit.py — verifies that the pipit creature is working correctly.

No gradient check needed (pipit has no hand-derived backprop — all
learning is Hebbian). Instead we verify:

  1. Single-class learning: a pipit trained on pure steady_2 should
     predict alternation accurately (CE well below ln(2)).
  2. Phase memory entrainment: the period-2 oscillator should show
     high R (concentration) after hearing alternating bits.
  3. No constant collapse: babbled output should have reasonable
     bit balance (not all-0 or all-1).
  4. Save/load round-trip: a saved and reloaded pipit should
     produce the same predictions.
"""
import sys
import numpy as np
from pipit import Pipit

def check():
    print("Pipit sanity check:\n")
    all_ok = True

    # 1. Single-class learning
    print("  1. Single-class learning (steady_2):")
    p = Pipit(name='test', n_osc=16, seed=0)
    seq = [0,1] * 32  # 64 bits of pure alternation
    for _ in range(200):
        p.brain.reset_phases(np.random.default_rng(42))  # same start
        for bit in seq:
            p.brain.tick(bit, learn=True)
    # Evaluate from a clean phase start with warmup
    # The creature needs ~16 ticks to entrain its oscillators
    snap = p.brain._snapshot()
    p.brain.reset_phases(np.random.default_rng(42))
    for bit in [0,1] * 8:  # 16-tick warmup
        p.brain.tick(bit, learn=False)
    losses = []
    for bit in [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]:
        _, loss = p.brain.tick(bit, learn=False)
        losses.append(loss)
    p.brain._restore(snap)
    ce = float(np.mean(losses))
    ok = ce < 0.693  # below uniform = learned something
    all_ok &= ok
    print(f"     CE on alternation: {ce:.3f}  "
          f"(need < 0.693 = ln2)  {'✓' if ok else '✗'}")
    # Also check babble accuracy (the real test)
    snap2 = p.brain._snapshot()
    p.brain.reset_phases(np.random.default_rng(42))
    for b in [0,1,0,1]:
        p.brain.tick(b, learn=False)
    out = []
    last = 1
    for _ in range(16):
        probs, _ = p.brain.tick(last, learn=False)
        last = int(np.argmax(probs))
        out.append(last)
    p.brain._restore(snap2)
    # Check if output alternates (regardless of starting phase)
    alternates = sum(out[i] != out[i+1] for i in range(len(out)-1))
    ok2 = alternates >= 13  # at least 13/15 transitions alternate
    all_ok &= ok2
    ostr = ''.join(map(str, out))
    print(f"     babble: {ostr}  alternations: {alternates}/15  "
          f"(need ≥13)  {'✓' if ok2 else '✗'}")

    # 2. Entrainment
    print("  2. Period-2 entrainment:")
    R = p.brain.entrainment()
    r0 = R[0].sum()  # period-2 oscillator total entrainment
    ok = r0 > 0.3
    all_ok &= ok
    print(f"     osc[0] (T≈2) total R: {r0:.2f}  "
          f"(need > 0.30)  {'✓' if ok else '✗'}")

    # 3. No constant collapse
    print("  3. Babble bit balance:")
    bb = p.babble_snapshot([0,1,0,1], n=100)
    frac_1 = sum(bb) / len(bb)
    ok = 0.2 < frac_1 < 0.8
    all_ok &= ok
    print(f"     fraction of 1s in babble: {frac_1:.2f}  "
          f"(need 0.20-0.80)  {'✓' if ok else '✗'}")

    # 4. Save/load round-trip
    print("  4. Save/load round-trip:")
    p.save('/tmp/test_pipit')
    p2 = Pipit.load('/tmp/test_pipit')
    probs1 = p.brain.predict_probs()
    probs2 = p2.brain.predict_probs()
    diff = float(np.max(np.abs(probs1 - probs2)))
    ok = diff < 1e-10
    all_ok &= ok
    print(f"     max prob difference: {diff:.2e}  "
          f"(need < 1e-10)  {'✓' if ok else '✗'}")

    print(f"\nResult: {'ALL OK' if all_ok else 'FAIL'}")
    return all_ok

if __name__ == '__main__':
    ok = check()
    sys.exit(0 if ok else 1)
