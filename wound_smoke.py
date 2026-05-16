"""
wound_smoke.py — acceptance tests for the pain v5 (wound) mechanism.

Five test groups, mapping to the design's central claims:

    1. Unit-level Wound behavior. Threshold gating, tick/inflict
       semantics, immune case at tenderness=0, life-cycle reset
       semantics. Pure-class tests, no Bittern integration.

    2. Costless when unused — the central v5 acceptance criterion.
       Two Bitterns, same seed, same listen schedule:
         A: wound.enabled = False
         B: wound.enabled = True but threshold = 1.1 (above the
            max possible sting = 1.0, so no wound ever fires)
       All brain params (including pain_embedding) must be
       bit-identical at the end. This is the cell v4 failed.

    3. Wound actually freezes the brain. A wounded creature's
       brain weights do not change during a train_step. Verified
       by comparing the brain weights before and after a forced
       wound iteration.

    4. Heritability. spawn() copies tenderness and mutates it by
       wound_sigma, clipping to [0, 1]. Repeated cloning explores
       the unit interval.

    5. Persistence. save/load round-trip preserves tenderness,
       threshold, counters, and the toggle. reset_emotional_state()
       preserves tenderness but clears runtime state.

Exit code 0 on full pass, 1 on any failure.
"""

from __future__ import annotations
import os
import sys
import tempfile

import numpy as np

from bittern import Brain, Bittern, Wound


# ── test runner ──────────────────────────────────────────────────────

class _Reporter:
    def __init__(self):
        self.passes = 0
        self.fails = 0

    def section(self, label):
        print(f"\n── {label} " + "─" * (60 - len(label)))

    def check(self, label, ok, detail=''):
        mark = '✓' if ok else '✗'
        if ok:
            self.passes += 1
        else:
            self.fails += 1
        d = f"   ({detail})" if detail else ''
        print(f"  {mark} {label}{d}")

    def summary(self):
        total = self.passes + self.fails
        print(f"\n{'─' * 65}")
        print(f"  {self.passes}/{total} checks passed"
              f"{'  — ALL GREEN' if self.fails == 0 else f'  — {self.fails} FAILED'}")
        return self.fails == 0


REPORT = _Reporter()


# ── 1. Unit-level Wound behavior ─────────────────────────────────────

def test_wound_unit():
    REPORT.section("Wound class — threshold / tick / inflict / immune")

    # tenderness is clipped at construction
    w = Wound(tenderness=1.5)
    REPORT.check("tenderness clipped to [0,1] at construction (upper)",
                 w.tenderness == 1.0,
                 f"got {w.tenderness}")
    w = Wound(tenderness=-0.5)
    REPORT.check("tenderness clipped to [0,1] at construction (lower)",
                 w.tenderness == 0.0,
                 f"got {w.tenderness}")

    # is_wounded honors enabled flag
    w = Wound(tenderness=0.5)
    w.wound_remaining = 5
    REPORT.check("is_wounded True when enabled and counter > 0",
                 w.is_wounded() is True)
    w.enabled = False
    REPORT.check("is_wounded False when disabled, regardless of counter",
                 w.is_wounded() is False)
    w.enabled = True
    w.wound_remaining = 0
    REPORT.check("is_wounded False when counter == 0",
                 w.is_wounded() is False)

    # maybe_inflict respects threshold
    w = Wound(tenderness=0.5, threshold=0.6, max_len=8)
    # sting = confidence * wrongness. probs=[0.9, 0.1], target=1
    #   confidence=0.9, wrongness=0.9, sting=0.81 → crosses 0.6
    fired = w.maybe_inflict(np.array([0.9, 0.1]), 1)
    REPORT.check("maybe_inflict fires when sting > threshold",
                 fired and w.wound_remaining > 0,
                 f"remaining={w.wound_remaining}")

    w = Wound(tenderness=0.5, threshold=0.6, max_len=8)
    # probs=[0.7, 0.3], target=1 → confidence=0.7, wrongness=0.7,
    #   sting=0.49 → below 0.6
    fired = w.maybe_inflict(np.array([0.7, 0.3]), 1)
    REPORT.check("maybe_inflict no-op when sting < threshold",
                 (not fired) and w.wound_remaining == 0,
                 f"remaining={w.wound_remaining}")

    # maybe_inflict no-op when disabled
    w = Wound(tenderness=0.5, threshold=0.6)
    w.enabled = False
    fired = w.maybe_inflict(np.array([0.99, 0.01]), 1)
    REPORT.check("maybe_inflict no-op when disabled",
                 (not fired) and w.wound_remaining == 0)

    # tenderness=0 is the immune case
    w = Wound(tenderness=0.0, threshold=0.1, max_len=8)
    fired = w.maybe_inflict(np.array([0.99, 0.01]), 1)
    REPORT.check("tenderness=0 makes the creature immune (no wound inflicted)",
                 (not fired) and w.wound_remaining == 0)

    # Duration scales with tenderness: ceil(tend * max_len)
    w = Wound(tenderness=0.3, threshold=0.1, max_len=8)
    w.maybe_inflict(np.array([0.99, 0.01]), 1)
    REPORT.check("duration = ceil(tenderness * max_len)",
                 w.wound_remaining == int(np.ceil(0.3 * 8)),
                 f"got {w.wound_remaining}, expected {int(np.ceil(0.3 * 8))}")

    w = Wound(tenderness=1.0, threshold=0.1, max_len=8)
    w.maybe_inflict(np.array([0.99, 0.01]), 1)
    REPORT.check("tenderness=1 gives full max_len wound",
                 w.wound_remaining == 8)

    # tick decrements to 0 and stays there
    w = Wound(tenderness=0.5, threshold=0.1, max_len=8)
    w.maybe_inflict(np.array([0.99, 0.01]), 1)
    start = w.wound_remaining
    for _ in range(start):
        w.tick()
    REPORT.check("tick decrements wound_remaining to 0 over `start` calls",
                 w.wound_remaining == 0,
                 f"after {start} ticks: {w.wound_remaining}")
    REPORT.check("tick at 0 is no-op (floor)",
                 (w.tick() or True) and w.wound_remaining == 0)

    # steps_silenced counter tracks the ticks
    w = Wound(tenderness=0.5, threshold=0.1, max_len=8)
    w.maybe_inflict(np.array([0.99, 0.01]), 1)
    n_inflicted = w.wounds_inflicted
    start = w.wound_remaining
    for _ in range(start):
        w.tick()
    REPORT.check("steps_silenced counter accumulates",
                 w.steps_silenced == start,
                 f"silenced={w.steps_silenced}, expected {start}")
    REPORT.check("wounds_inflicted counter accumulates",
                 w.wounds_inflicted == 1)

    # reset_for_life: tenderness persists, runtime clears
    w = Wound(tenderness=0.7, threshold=0.1, max_len=8)
    w.maybe_inflict(np.array([0.99, 0.01]), 1)
    w.tick()
    w.tick()
    w.reset_for_life()
    REPORT.check("reset_for_life preserves tenderness",
                 w.tenderness == 0.7)
    REPORT.check("reset_for_life clears wound_remaining",
                 w.wound_remaining == 0)
    REPORT.check("reset_for_life clears wounds_inflicted",
                 w.wounds_inflicted == 0)
    REPORT.check("reset_for_life clears steps_silenced",
                 w.steps_silenced == 0)


# ── 2. Costless when unused — the central acceptance criterion ──────

def test_costless_when_unused():
    REPORT.section("costless when unused — A (disabled) vs B (threshold=1.1)")

    # The v4 failure: with proprioception enabled but no records,
    # the brain still paid a representational cost. We test the
    # analogous claim for v5: a wound that NEVER fires costs zero.
    #
    # Strategy: build two identical Bitterns, same seed. A has
    # wound.enabled = False. B has wound.enabled = True but a
    # threshold above the maximum possible sting (which is 1.0,
    # since sting = confidence * wrongness, both in [0,1]). B's
    # maybe_inflict will always return False. The listen() loop
    # passes is_wounded()=False on every iteration, so
    # eff_brain_lr is never zeroed. Brain weight trajectories
    # must be bit-identical.

    rng_corpus = np.random.default_rng(7)
    corpus = []
    for _ in range(20):
        n = int(rng_corpus.integers(16, 32))
        corpus.append({'bits': rng_corpus.integers(0, 2, size=n).tolist()})

    a = Bittern(name='A', seed=123, embed_dim=8, brain_dim=8, context=8)
    a.wound.enabled = False
    # Disable all other pain mechanisms too — we are testing v5 in
    # isolation against baseline.
    a.pain.enabled = False
    a.painful_memory.enabled = False
    a.scars.enabled = False
    a.proprioception.enabled = False

    b = Bittern(name='B', seed=123, embed_dim=8, brain_dim=8, context=8)
    b.wound.enabled = True
    b.wound.threshold = 1.1   # nothing crosses; no wound ever fires
    b.pain.enabled = False
    b.painful_memory.enabled = False
    b.scars.enabled = False
    b.proprioception.enabled = False

    for r in range(50):
        item = corpus[r % len(corpus)]
        a.listen(item['bits'], train_pairs=3, lr=0.02)
        a.step()
        b.listen(item['bits'], train_pairs=3, lr=0.02)
        b.step()

    # Verify B never recorded a wound (it shouldn't have)
    REPORT.check("B's threshold prevents any wound from firing",
                 b.wound.wounds_inflicted == 0,
                 f"B wounds_inflicted = {b.wound.wounds_inflicted}")

    # All brain params bit-identical
    params = ['embedding', 'W_q', 'W_k', 'W_v',
              'W_o', 'W_out', 'b_out', 'pain_embedding']
    max_diff_ab = 0.0
    worst_ab = ''
    for pname in params:
        wa = getattr(a.brain, pname)
        wb = getattr(b.brain, pname)
        d = float(np.max(np.abs(wa - wb)))
        if d > max_diff_ab:
            max_diff_ab = d
            worst_ab = pname
    REPORT.check("brain weights bit-identical when wound never fires",
                 max_diff_ab == 0.0,
                 f"max abs diff = {max_diff_ab:.2e} on {worst_ab}")


def test_wound_diverges_when_active():
    REPORT.section("wound actually diverges when it fires — A vs C")

    # Same setup as the costless test, but C uses a low threshold
    # so wounds DO fire. C should diverge from A.

    rng_corpus = np.random.default_rng(7)
    corpus = []
    for _ in range(20):
        n = int(rng_corpus.integers(16, 32))
        corpus.append({'bits': rng_corpus.integers(0, 2, size=n).tolist()})

    a = Bittern(name='A', seed=123, embed_dim=8, brain_dim=8, context=8)
    a.wound.enabled = False
    a.pain.enabled = False
    a.painful_memory.enabled = False
    a.scars.enabled = False
    a.proprioception.enabled = False

    c = Bittern(name='C', seed=123, embed_dim=8, brain_dim=8, context=8)
    c.wound.enabled = True
    c.wound.threshold = 0.1   # low threshold — wounds fire freely
    c.wound.tenderness = 0.5
    c.pain.enabled = False
    c.painful_memory.enabled = False
    c.scars.enabled = False
    c.proprioception.enabled = False

    for r in range(50):
        item = corpus[r % len(corpus)]
        a.listen(item['bits'], train_pairs=3, lr=0.02)
        a.step()
        c.listen(item['bits'], train_pairs=3, lr=0.02)
        c.step()

    REPORT.check("C accumulated wounds with low threshold",
                 c.wound.wounds_inflicted > 0,
                 f"C wounds_inflicted = {c.wound.wounds_inflicted}, "
                 f"steps_silenced = {c.wound.steps_silenced}")

    params = ['embedding', 'W_q', 'W_k', 'W_v',
              'W_o', 'W_out', 'b_out']
    max_diff_ac = 0.0
    for pname in params:
        wa = getattr(a.brain, pname)
        wc = getattr(c.brain, pname)
        d = float(np.max(np.abs(wa - wc)))
        if d > max_diff_ac:
            max_diff_ac = d
    REPORT.check("wound-active creature diverges from disabled twin",
                 max_diff_ac > 0.0,
                 f"max param diff = {max_diff_ac:.4e}")


# ── 3. Wound freezes the brain mid-listen ────────────────────────────

def test_brain_frozen_during_wound():
    REPORT.section("brain weights frozen while wound_remaining > 0")

    # Force a wound, then take a snapshot, then call listen with
    # one train_pair, and verify weights didn't move. We disable
    # all other pain mechanisms so the only thing that could
    # change a weight is train_step.

    b = Bittern(name='W', seed=999, embed_dim=8, brain_dim=8, context=8)
    b.wound.enabled = True
    b.wound.tenderness = 1.0   # max wound duration
    b.wound.threshold = 0.6
    b.pain.enabled = False
    b.painful_memory.enabled = False
    b.scars.enabled = False
    b.proprioception.enabled = False

    # Force the wound directly (we don't need to wait for it to
    # fire naturally — we are testing the GATE, not the trigger).
    b.wound.wound_remaining = 5

    snapshot = {}
    params = ['embedding', 'W_q', 'W_k', 'W_v', 'W_o', 'W_out', 'b_out']
    for pname in params:
        snapshot[pname] = getattr(b.brain, pname).copy()

    # Listen with several train_pairs. While wounded, none should
    # cause a weight update.
    rng = np.random.default_rng(42)
    for _ in range(3):
        bits = rng.integers(0, 2, size=20).tolist()
        b.listen(bits, train_pairs=1, lr=0.05)

    REPORT.check("wound_remaining ticked down to 2 (5 - 3 listens × 1 pair)",
                 b.wound.wound_remaining == 2,
                 f"got {b.wound.wound_remaining}")

    max_diff = 0.0
    worst = ''
    for pname in params:
        d = float(np.max(np.abs(getattr(b.brain, pname) - snapshot[pname])))
        if d > max_diff:
            max_diff = d
            worst = pname
    REPORT.check("brain weights identical to snapshot while wounded",
                 max_diff == 0.0,
                 f"max abs diff = {max_diff:.2e} on {worst}")

    # Now let the wound expire and check that learning resumes.
    # 2 more pairs to fully clear, then a 3rd that learns.
    rng2 = np.random.default_rng(43)
    # Drain remaining wound
    for _ in range(2):
        bits = rng2.integers(0, 2, size=20).tolist()
        b.listen(bits, train_pairs=1, lr=0.05)
    REPORT.check("wound fully drained after enough listens",
                 b.wound.wound_remaining == 0)

    # Disable wound entirely now (so a new wound from this pair
    # doesn't immediately re-freeze the brain), and verify weights
    # move on the next listen.
    b.wound.enabled = False
    snap2 = {p: getattr(b.brain, p).copy() for p in params}
    bits = rng2.integers(0, 2, size=20).tolist()
    b.listen(bits, train_pairs=2, lr=0.05)
    max_diff_post = 0.0
    for pname in params:
        d = float(np.max(np.abs(getattr(b.brain, pname) - snap2[pname])))
        if d > max_diff_post:
            max_diff_post = d
    REPORT.check("brain weights resume changing once wound is clear",
                 max_diff_post > 0.0,
                 f"max diff post-thaw = {max_diff_post:.4e}")


# ── 4. Heritability ──────────────────────────────────────────────────

def test_heritability():
    REPORT.section("clone_for_child — tenderness inheritance + mutation")

    # Single-step inheritance: child differs from parent by some
    # Gaussian noise (with sigma=0.1 by default), clipped to [0,1].
    p = Wound(tenderness=0.5, mutation_sigma=0.1)
    rng = np.random.default_rng(2026)
    diffs = []
    for _ in range(200):
        c = p.clone_for_child(rng=rng)
        diffs.append(c.tenderness - p.tenderness)
        # Each child must be in [0,1]
        if not (0.0 <= c.tenderness <= 1.0):
            REPORT.check("child tenderness in [0,1]", False,
                         f"got {c.tenderness}")
            break
    else:
        REPORT.check("all 200 children have tenderness in [0,1]", True)

    diffs = np.asarray(diffs)
    # Mean should be near 0 (no bias), std near 0.1
    REPORT.check("mutation has near-zero mean (no bias)",
                 abs(float(np.mean(diffs))) < 0.02,
                 f"mean = {float(np.mean(diffs)):.4f}")
    REPORT.check("mutation has std near sigma (0.1)",
                 0.07 < float(np.std(diffs)) < 0.13,
                 f"std = {float(np.std(diffs)):.4f}")

    # Clipping at boundary: parent at tenderness=0 should mostly
    # see children at 0 (negative noise clipped) and some positive.
    p_low = Wound(tenderness=0.0, mutation_sigma=0.1)
    children_low = [p_low.clone_for_child(rng=rng).tenderness
                    for _ in range(200)]
    REPORT.check("tenderness=0 parent has children with tenderness >= 0",
                 all(t >= 0.0 for t in children_low))
    n_clipped = sum(1 for t in children_low if t == 0.0)
    REPORT.check("about half clipped at 0 (Gaussian centered on boundary)",
                 80 <= n_clipped <= 120,
                 f"clipped at 0: {n_clipped}/200")

    # Clone preserves config (threshold, max_len, mutation_sigma)
    p = Wound(tenderness=0.4, threshold=0.55, max_len=10,
              mutation_sigma=0.07)
    p.enabled = False
    c = p.clone_for_child(rng=rng)
    REPORT.check("clone preserves threshold",
                 c.threshold == 0.55)
    REPORT.check("clone preserves max_len",
                 c.max_len == 10)
    REPORT.check("clone preserves mutation_sigma",
                 c.mutation_sigma == 0.07)
    REPORT.check("clone preserves enabled flag",
                 c.enabled is False)
    # Runtime state always starts fresh in the child
    REPORT.check("clone starts with wound_remaining=0",
                 c.wound_remaining == 0)
    REPORT.check("clone starts with wounds_inflicted=0",
                 c.wounds_inflicted == 0)


# ── 5. Persistence ───────────────────────────────────────────────────

def test_persistence():
    REPORT.section("save/load and reset_emotional_state")

    # Create a Bittern, give it a non-default wound state, save,
    # load, and verify everything survived.
    b = Bittern(name='P', seed=77, embed_dim=8, brain_dim=8, context=8)
    b.wound.tenderness = 0.42
    b.wound.threshold = 0.55
    b.wound.max_len = 12
    b.wound.mutation_sigma = 0.07
    b.wound.wound_remaining = 3
    b.wound.wounds_inflicted = 17
    b.wound.steps_silenced = 49
    b.wound.enabled = False

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'p')
        b.save(path)
        b2 = Bittern.load(path)

    REPORT.check("loaded tenderness matches saved",
                 b2.wound.tenderness == 0.42)
    REPORT.check("loaded threshold matches saved",
                 b2.wound.threshold == 0.55)
    REPORT.check("loaded max_len matches saved",
                 b2.wound.max_len == 12)
    REPORT.check("loaded mutation_sigma matches saved",
                 b2.wound.mutation_sigma == 0.07)
    REPORT.check("loaded wound_remaining matches saved",
                 b2.wound.wound_remaining == 3)
    REPORT.check("loaded wounds_inflicted matches saved",
                 b2.wound.wounds_inflicted == 17)
    REPORT.check("loaded steps_silenced matches saved",
                 b2.wound.steps_silenced == 49)
    REPORT.check("loaded enabled toggle matches saved",
                 b2.wound.enabled is False)

    # reset_emotional_state preserves tenderness, clears runtime
    b3 = Bittern(name='R', seed=88)
    b3.wound.tenderness = 0.66
    b3.wound.wound_remaining = 4
    b3.wound.wounds_inflicted = 12
    b3.wound.steps_silenced = 25
    b3.reset_emotional_state()
    REPORT.check("reset_emotional_state preserves tenderness",
                 b3.wound.tenderness == 0.66)
    REPORT.check("reset_emotional_state clears wound_remaining",
                 b3.wound.wound_remaining == 0)
    REPORT.check("reset_emotional_state clears wounds_inflicted",
                 b3.wound.wounds_inflicted == 0)
    REPORT.check("reset_emotional_state clears steps_silenced",
                 b3.wound.steps_silenced == 0)


# ── main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_wound_unit()
    test_costless_when_unused()
    test_wound_diverges_when_active()
    test_brain_frozen_during_wound()
    test_heritability()
    test_persistence()
    ok = REPORT.summary()
    sys.exit(0 if ok else 1)
