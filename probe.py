"""
probe.py — diagnostics for a bittern.

The free-running probe is the cleanest finding from the clade work:
training loss can drop while the brain becomes increasingly surprised
by its own outputs. With vocab=2 the numbers are interpretable —
ln(2) ≈ 0.693 is the uniform-random reference, so values near 0 mean
"confident and right", values near 0.693 mean "no signal", and values
ABOVE 0.693 mean "confidently wrong about itself."

Four metrics:
    teacher_force_ce  : CE on (real prefix, real next bit) pairs
    free_running_ce   : CE on the bittern's OWN babbled bits
    noise_recovery_ce : CE on real targets after perturbing the prefix
    per-class CE      : teacher-force CE broken out by rhythm class
"""

from __future__ import annotations
import numpy as np


UNIFORM = float(np.log(2))   # 0.693, the vocab=2 random reference


def _take_prefix(bits, end_idx, context):
    start = max(0, end_idx - context)
    return bits[start:end_idx]


def teacher_force_ce(bittern, sequences):
    """CE over (real prefix, real next bit) pairs. Training-loss proxy."""
    losses = []
    ctx = bittern.brain.context
    for seq in sequences:
        bits = seq['bits'] if isinstance(seq, dict) else seq
        for i in range(1, len(bits)):
            prefix = _take_prefix(bits, i, ctx)
            target = int(bits[i])
            probs = bittern.brain.predict_probs(prefix)
            losses.append(-np.log(probs[target] + 1e-12))
    return float(np.mean(losses)) if losses else 0.0


def free_running_ce(bittern, prompts, n_per_prompt=32):
    """For each prompt, babble n bits, then score the brain's surprise
    on its own outputs. fr_ce above UNIFORM means the brain is more
    surprised by itself than by random — confident wrongness."""
    losses = []
    ctx = bittern.brain.context
    for prompt in prompts:
        bits = prompt['bits'] if isinstance(prompt, dict) else prompt
        bits = list(bits)
        babble = bittern.babble(bits, n=n_per_prompt)
        full = bits + babble
        # Score the babbled portion only, conditioning on growing context
        for i in range(len(bits), len(full)):
            prefix = _take_prefix(full, i, ctx)
            target = int(full[i])
            probs = bittern.brain.predict_probs(prefix)
            losses.append(-np.log(probs[target] + 1e-12))
    return float(np.mean(losses)) if losses else 0.0


def noise_recovery_ce(bittern, sequences, n_perturb=1, rng=None):
    """Real targets, but n_perturb bits of the prefix are flipped before
    scoring. Tests robustness to off-distribution context."""
    rng = rng if rng is not None else np.random.default_rng(7)
    losses = []
    ctx = bittern.brain.context
    for seq in sequences:
        bits = list(seq['bits'] if isinstance(seq, dict) else seq)
        for i in range(2, len(bits)):
            prefix = list(_take_prefix(bits, i, ctx))
            target = int(bits[i])
            for _ in range(min(n_perturb, len(prefix))):
                idx = int(rng.integers(0, len(prefix)))
                prefix[idx] = 1 - prefix[idx]
            probs = bittern.brain.predict_probs(prefix)
            losses.append(-np.log(probs[target] + 1e-12))
    return float(np.mean(losses)) if losses else 0.0


def per_class_ce(bittern, corpus):
    """Teacher-force CE broken out by rhythm class."""
    by_class = {}
    ctx = bittern.brain.context
    for seq in corpus:
        cls = seq['class']
        bits = seq['bits']
        for i in range(1, len(bits)):
            prefix = _take_prefix(bits, i, ctx)
            target = int(bits[i])
            probs = bittern.brain.predict_probs(prefix)
            loss = -np.log(probs[target] + 1e-12)
            by_class.setdefault(cls, []).append(loss)
    return {cls: float(np.mean(v)) for cls, v in by_class.items()}


# ----------------------------------------------------------------------
#  Pretty-print
# ----------------------------------------------------------------------

def probe_report(bittern, eval_corpus, prompts, label=None):
    tf = teacher_force_ce(bittern, eval_corpus)
    fr = free_running_ce(bittern, prompts, n_per_prompt=32)
    nr = noise_recovery_ce(bittern, eval_corpus, n_perturb=1)
    by = per_class_ce(bittern, eval_corpus)

    title = label or f"{bittern.name} @ round {bittern.round}"
    print(f"\n  --- probe: {title} ---")
    print(f"    uniform-random reference   ln(2) = {UNIFORM:.3f} nats")
    print(f"    teacher_force_ce           {tf:.3f}")
    print(f"    free_running_ce            {fr:.3f}   "
          f"(gap = {fr - tf:+.3f})")
    print(f"    noise_recovery_ce          {nr:.3f}   "
          f"(gap = {nr - tf:+.3f})")
    print(f"    per-class teacher-force:")
    for cls, ce in sorted(by.items()):
        bar_w = 24
        rel = min(1.0, ce / UNIFORM)
        filled = int(rel * bar_w)
        bar = '█' * filled + '░' * (bar_w - filled)
        marker = '✓' if ce < UNIFORM * 0.5 else '·' if ce < UNIFORM else '✗'
        print(f"      {cls:10s} {marker} {ce:.3f}  [{bar}]")
    return {'tf_ce': tf, 'fr_ce': fr, 'nr_ce': nr, 'per_class': by}
