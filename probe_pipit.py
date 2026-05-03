"""
probe_pipit.py — diagnostics for a pipit.

Same metrics as bittern's probe, adapted for tick-by-tick processing:

    teacher_force_ce  : CE when the pipit hears real bits (its loss
                        during experience). This is just the average
                        loss from tick() on real sequences.

    free_running_ce   : CE when the pipit hears its own outputs.
                        The creature babbles, and we measure how
                        surprised it is by what it produces.
                        In a tick-by-tick creature, this is just the
                        loss during babble — no separate mode needed.

    per_class_ce      : teacher-force CE broken out by rhythm class.

    convergence_ce    : NEW metric. Feed the same sequence twice in a
                        row. CE on the second pass should be lower than
                        the first — the creature should get better at a
                        rhythm while experiencing it. Measures whether
                        the online learning is actually working.

All read against ln(2) ≈ 0.693 (the uniform-random reference for
vocab=2).
"""

from __future__ import annotations
import numpy as np

UNIFORM = float(np.log(2))   # 0.693


def teacher_force_ce(pipit, sequences):
    """CE when the pipit hears real bits from the world.
    Snapshot/restore so probing doesn't change the creature."""
    snap = pipit.brain._snapshot()
    losses = []
    for seq in sequences:
        bits = seq['bits'] if isinstance(seq, dict) else seq
        pipit.brain.reset_phases(pipit.rng)
        for bit in bits:
            _, loss = pipit.brain.tick(int(bit), learn=False)
            losses.append(loss)
    pipit.brain._restore(snap)
    return float(np.mean(losses)) if losses else 0.0


def free_running_ce(pipit, prompts, n_per_prompt=32):
    """CE on the pipit's own babbled bits. The creature processes a
    prompt, then feeds its own output back as input. We measure how
    surprised it is by what it generates.

    In a tick-by-tick creature this is just the loss during the
    babble phase — no mode switch needed."""
    snap = pipit.brain._snapshot()
    rng_state = pipit.rng.bit_generator.state
    losses = []
    for prompt in prompts:
        bits = prompt['bits'] if isinstance(prompt, dict) else prompt
        pipit.brain.reset_phases(pipit.rng)
        # Process prompt (no learning)
        for bit in bits:
            pipit.brain.tick(int(bit), learn=False)
        # Babble and measure surprise
        last_bit = int(bits[-1])
        for _ in range(n_per_prompt):
            probs, loss = pipit.brain.tick(last_bit, learn=False)
            losses.append(loss)
            # Emit from prediction
            logits = np.log(probs + 1e-12) / max(pipit.temperature, 1e-6)
            logits = logits - logits.max()
            e = np.exp(logits)
            sp = e / e.sum()
            last_bit = int(pipit.rng.choice(2, p=sp))
    pipit.brain._restore(snap)
    pipit.rng.bit_generator.state = rng_state
    return float(np.mean(losses)) if losses else 0.0


def per_class_ce(pipit, corpus):
    """Teacher-force CE broken out by rhythm class."""
    snap = pipit.brain._snapshot()
    by_class = {}
    for seq in corpus:
        cls = seq['class']
        bits = seq['bits']
        pipit.brain.reset_phases(pipit.rng)
        for bit in bits:
            _, loss = pipit.brain.tick(int(bit), learn=False)
            by_class.setdefault(cls, []).append(loss)
    pipit.brain._restore(snap)
    return {cls: float(np.mean(v)) for cls, v in by_class.items()}


def convergence_ce(pipit, sequences, passes=2):
    """Feed the same sequence multiple times. CE should drop on
    later passes if online learning is working. Returns CE per pass."""
    snap = pipit.brain._snapshot()
    losses_by_pass = [[] for _ in range(passes)]
    for seq in sequences:
        bits = seq['bits'] if isinstance(seq, dict) else seq
        pipit.brain.reset_phases(pipit.rng)
        for p in range(passes):
            for bit in bits:
                _, loss = pipit.brain.tick(int(bit), learn=False)
                losses_by_pass[p].append(loss)
    pipit.brain._restore(snap)
    return [float(np.mean(v)) for v in losses_by_pass]


def probe_report(pipit, eval_corpus, prompts, label=None):
    tf = teacher_force_ce(pipit, eval_corpus)
    fr = free_running_ce(pipit, prompts, n_per_prompt=32)
    by = per_class_ce(pipit, eval_corpus)

    title = label or f"{pipit.name} @ tick {pipit.round}"
    print(f"\n  --- probe: {title} ---")
    print(f"    uniform-random reference   ln(2) = {UNIFORM:.3f} nats")
    print(f"    teacher_force_ce           {tf:.3f}")
    print(f"    free_running_ce            {fr:.3f}   "
          f"(gap = {fr - tf:+.3f})")
    print(f"    per-class teacher-force:")
    for cls, ce in sorted(by.items()):
        bar_w = 24
        rel = min(1.0, ce / UNIFORM)
        filled = int(rel * bar_w)
        bar = '█' * filled + '░' * (bar_w - filled)
        marker = '✓' if ce < UNIFORM * 0.5 else '·' if ce < UNIFORM else '✗'
        print(f"      {cls:10s} {marker} {ce:.3f}  [{bar}]")
    return {'tf_ce': tf, 'fr_ce': fr, 'per_class': by}
