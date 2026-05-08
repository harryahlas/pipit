"""
pain_probes.py — diagnostics for the pain system.

Five probes:
    pain_report(bittern)       — dict snapshot of all five pains + battery
    pain_summary(bittern)      — pretty-printed snapshot
    pain_trajectory(bittern,   — train and sample pain state over time;
                    corpus, n)   shows how pain moves with class transitions
    behavioral_trace(bittern,  — record effective_* values; shows what
                     sequence)   the creature ACTUALLY does differently
    pain_vs_loss_scatter(...)  — per-class pain levels paired with losses

These complement probe.py's prediction-quality probes. Together they
let you ask: not just "how well does this creature predict" but
"what is this creature experiencing, and how is that changing what it does."
"""

from __future__ import annotations
import numpy as np

from bittern import VOCAB_SIZE


UNIFORM = float(np.log(2))


# ── Single-snapshot probes ────────────────────────────────────────────

def pain_report(bittern):
    """Dict snapshot of all five pains plus battery."""
    rep = bittern.pain.report(bittern.recent_brain_losses, bittern.round)
    rep['battery'] = float(bittern.battery.level)
    rep['lonely'] = bool(bittern.battery.is_lonely())
    return rep


def pain_summary(bittern, label=None):
    """Pretty-print pain state with horizontal bars."""
    title = label or f"{bittern.name} @ round {bittern.round}"
    rep = pain_report(bittern)
    print(f"\n  --- pain: {title} ---")
    bar_w = 20
    for k in ['sting', 'hunger', 'nausea', 'itch', 'disquiet']:
        v = rep[k]
        filled = int(min(1.0, v) * bar_w)
        bar = '█' * filled + '░' * (bar_w - filled)
        print(f"    {k:9s}  {v:.3f}  [{bar}]")
    bat_filled = int(rep['battery'] * bar_w)
    bat_bar = '█' * bat_filled + '░' * (bar_w - bat_filled)
    lonely_tag = '  ← LONELY' if rep['lonely'] else ''
    print(f"    {'battery':9s}  {rep['battery']:.3f}  [{bat_bar}]{lonely_tag}")


# ── Behavioral trace ─────────────────────────────────────────────────

def behavioral_trace(bittern, sequence):
    """Record what effective_* values the creature would use along
    a sequence. Pain doesn't update during babble, so values are
    constant within a single trace — but comparing traces from
    creatures in different pain states (or one creature at different
    points in time) shows what pain actually changes.

    Returns a list of dicts, one per position."""
    pain = bittern.pain
    rec = bittern.recent_brain_losses
    rd = bittern.round
    base_ctx = bittern.brain.context
    base_temp = bittern.organs.temperature
    base_lat = bittern.organs.lateral_weight
    base_sens_lr = bittern.organs.sensitivity_lr

    return [{
        'pos': i,
        'bit': int(b),
        'effective_context': pain.effective_context(base_ctx, rec, rd),
        'effective_temperature': pain.effective_temperature(base_temp),
        'effective_lateral_weight': pain.effective_lateral_weight(base_lat),
        'effective_sensitivity_lr': pain.effective_sensitivity_lr(base_sens_lr),
    } for i, b in enumerate(sequence)]


def behavioral_trace_summary(bittern, label=None):
    """Single-line summary of effective_* values, useful for comparing
    creatures or checkpoints. The values are constant within a babble,
    so we just report them once."""
    pain = bittern.pain
    rec = bittern.recent_brain_losses
    rd = bittern.round

    eff_ctx = pain.effective_context(bittern.brain.context, rec, rd)
    eff_temp = pain.effective_temperature(bittern.organs.temperature)
    eff_lat = pain.effective_lateral_weight(bittern.organs.lateral_weight)
    eff_sens = pain.effective_sensitivity_lr(bittern.organs.sensitivity_lr)
    eff_brain_lr_mult = pain.effective_brain_lr(1.0)  # multiplier, base lr=1

    title = label or f"{bittern.name} @ round {bittern.round}"
    print(f"  behavior: {title}")
    print(f"    context           {bittern.brain.context:>3} → "
          f"{eff_ctx:>3}   (hunger)")
    print(f"    temperature      {bittern.organs.temperature:.3f} → "
          f"{eff_temp:.3f}   (sting)")
    print(f"    lateral_weight   {bittern.organs.lateral_weight:.3f} → "
          f"{eff_lat:.3f}   (nausea)")
    print(f"    sensitivity_lr   {bittern.organs.sensitivity_lr:.4f} → "
          f"{eff_sens:.4f}  (itch)")
    print(f"    brain_lr_mult     1.000 → {eff_brain_lr_mult:.3f}   (disquiet)")


# ── Trajectory ───────────────────────────────────────────────────────

def pain_trajectory(bittern, corpus, n_rounds, sample_every=50,
                    train_pairs=4, lr=0.02, replenish=0.02):
    """Train for n_rounds, sampling pain state every sample_every rounds.
    Returns a list of dicts. Iterates the corpus round-robin (so if you
    pass a block-structured corpus, you'll see how pain responds to
    block boundaries).

    This DOES train the bittern — it's the "live" version of pain
    measurement. Use behavioral_trace if you want to inspect a frozen
    creature without modifying it."""
    samples = []
    for r in range(n_rounds):
        item = corpus[r % len(corpus)]
        bittern.listen(item['bits'], train_pairs=train_pairs, lr=lr)
        bittern.step()
        bittern.battery.replenish(replenish)
        if r % sample_every == 0 or r == n_rounds - 1:
            avg = (float(np.mean(bittern.recent_brain_losses[-50:]))
                   if bittern.recent_brain_losses else 0.0)
            samples.append({
                'round': bittern.round,
                **bittern.pain.report(bittern.recent_brain_losses,
                                      bittern.round),
                'avg_loss': avg,
                'class': item.get('class', '?'),
                'battery': float(bittern.battery.level),
            })
    return samples


def print_trajectory(samples, max_rows=None):
    """Pretty-print a pain_trajectory output. Set max_rows to truncate
    long trajectories — head and tail are shown in that case."""
    print(f"\n  {'round':>6}  {'class':<10s}  {'sting':>5s}  {'hunger':>6s}  "
          f"{'nausea':>6s}  {'itch':>5s}  {'disq':>5s}  {'loss':>5s}  "
          f"{'bat':>4s}")
    rows = list(samples)
    if max_rows is not None and len(rows) > max_rows:
        head = max_rows // 2
        tail = max_rows - head
        for s in rows[:head]:
            _print_traj_row(s)
        print(f"  {'...':>6}  {'...':<10s}  {'...':>5s}")
        for s in rows[-tail:]:
            _print_traj_row(s)
    else:
        for s in rows:
            _print_traj_row(s)


def _print_traj_row(s):
    print(f"  {s['round']:>6}  {s['class']:<10s}  "
          f"{s['sting']:>5.3f}  {s['hunger']:>6.3f}  "
          f"{s['nausea']:>6.3f}  {s['itch']:>5.3f}  "
          f"{s['disquiet']:>5.3f}  {s['avg_loss']:>5.3f}  "
          f"{s['battery']:>4.2f}")


# ── Pain × loss scatter ──────────────────────────────────────────────

def pain_vs_loss_scatter(samples):
    """Bin samples by class and report mean pain level vs mean loss.
    Useful for testing the hypothesis that sting anti-correlates with
    future loss (calibrated creatures), hunger positively correlates
    with loss (it measures loss), etc."""
    by_class = {}
    for s in samples:
        by_class.setdefault(s['class'], []).append(s)

    print(f"\n  Pain × loss by class:")
    print(f"    {'class':<10s}  {'n':>3s}  {'loss':>5s}  "
          f"{'sting':>5s}  {'hunger':>6s}  {'nausea':>6s}  "
          f"{'itch':>5s}  {'disq':>5s}")
    for cls in sorted(by_class.keys()):
        rows = by_class[cls]
        n = len(rows)
        loss = float(np.mean([r['avg_loss'] for r in rows]))
        sting = float(np.mean([r['sting'] for r in rows]))
        hunger = float(np.mean([r['hunger'] for r in rows]))
        nausea = float(np.mean([r['nausea'] for r in rows]))
        itch = float(np.mean([r['itch'] for r in rows]))
        disq = float(np.mean([r['disquiet'] for r in rows]))
        print(f"    {cls:<10s}  {n:>3d}  {loss:>5.3f}  "
              f"{sting:>5.3f}  {hunger:>6.3f}  {nausea:>6.3f}  "
              f"{itch:>5.3f}  {disq:>5.3f}")


# ── Class-boundary detection ─────────────────────────────────────────

def find_boundary_responses(samples, pain_key='nausea',
                            response_window=5):
    """For each class transition in the trajectory, report the change
    in a given pain level over the next `response_window` samples.
    Tests the claim that nausea (or any pain) spikes at boundaries."""
    boundaries = []
    for i in range(1, len(samples)):
        if samples[i]['class'] != samples[i-1]['class']:
            before = samples[i-1][pain_key]
            after_window = samples[i:i+response_window]
            if not after_window:
                continue
            after_max = max(s[pain_key] for s in after_window)
            boundaries.append({
                'round': samples[i]['round'],
                'from': samples[i-1]['class'],
                'to': samples[i]['class'],
                'before': before,
                'after_max': after_max,
                'rise': after_max - before,
            })
    return boundaries


def print_boundary_responses(boundaries, pain_key='nausea'):
    if not boundaries:
        print(f"\n  No class boundaries found in trajectory.")
        return
    print(f"\n  {pain_key} response at class transitions:")
    print(f"    {'round':>6}  {'from':<10s} → {'to':<10s}  "
          f"{'before':>6s}  {'after':>6s}  {'rise':>6s}")
    for b in boundaries:
        marker = '↑' if b['rise'] > 0.05 else '·'
        print(f"    {b['round']:>6}  {b['from']:<10s} → {b['to']:<10s}  "
              f"{b['before']:>6.3f}  {b['after_max']:>6.3f}  "
              f"{b['rise']:>+6.3f} {marker}")
    risers = sum(1 for b in boundaries if b['rise'] > 0.05)
    print(f"    -> {risers}/{len(boundaries)} boundaries produced a rise > 0.05")
