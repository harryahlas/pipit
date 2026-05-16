"""
evolve.py — natural selection for bitterns.

Two levels of pressure:
    1. WITHIN a generation: competitive habitat (winner eats, losers bleed)
    2. BETWEEN generations: the weakest die, the strongest reproduce

Each generation:
    - Population of N creatures lives together in a competitive habitat
    - After training, evaluate each creature on a mixed eval corpus
    - Bottom half dies
    - Top half reproduces: clone brain weights + Gaussian mutation
    - Offspring get fresh organs, fresh battery, mutated brain

The mutation is the only source of variation after generation 0.
Selection is the only thing shaping the population. No intelligent
design, no hyperparameter tuning mid-run. Whatever wins, wins.

    python evolve.py                              # 8 creatures, 10 generations
    python evolve.py --pop 12 --generations 15    # bigger tournament
    python evolve.py --rounds 8000 --sigma 0.03   # tune pressure
    python evolve.py --no-pain                    # ablation: pain disabled
    python evolve.py --no-proprioception          # ablation: pain v4 disabled

Architecture:

    Generation 0:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
                    |    |    |    |    |    |    |    |
                    ---- compete in habitat for N rounds ----
                    |    |    |    |    |    |    |    |
                    evaluate fitness (teacher-force CE)
                    |    |    |    |    |    |    |    |
                    rank: keep top 4, kill bottom 4
                    |    |    |    |
                    reproduce with mutation
                    |  ↘  |  ↘  |  ↘  |  ↘
    Generation 1:  [b0] [b1] [b2] [b3] [b4] [b5] [b6] [b7]
                    parents       children (mutated clones)
                    ...repeat...

Two changes from the original:

  (1) The loneliness lr-multiplier was removed. The sigma=0
      diagnostic showed that with `effective_lr = lr * (2.0 if
      lonely else 1.0)`, champions DEGRADED across generations even
      with no mutation — same brain weights, same world, identical
      opponents, but each successive generation re-trained the brain
      destructively when batteries crashed. With the multiplier
      removed, champions accumulate progress generation over
      generation, and the loner control no longer beats them.

  (2) Surviving parents now call `reset_emotional_state()` rather
      than just `Battery()` — so parents enter the next generation
      with the same pain warmup as their children, equalizing the
      starting conditions and making selection operate on brain
      weights rather than on stale emotional state.

  (3) A `--no-pain` CLI flag is wired through to disable the pain
      system on every creature (gen 0, every child, the loner). Use
      it to compare with-pain vs without-pain champions.
"""

from __future__ import annotations
import argparse
import copy
import os
import time

import numpy as np

from bittern import Bittern, Brain, Organs, Battery, VOCAB_SIZE
from world import make_corpus, bits_to_str
from probe import teacher_force_ce, per_class_ce, probe_report, UNIFORM


# ── Names ────────────────────────────────────────────────────────────

NAME_POOL = [
    'ada', 'bob', 'cid', 'dot', 'eve', 'fig', 'gus', 'hal',
    'ivy', 'jan', 'kit', 'leo', 'max', 'neo', 'ora', 'pax',
]

GEN_PREFIX = [
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ',
    'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π',
    'ρ', 'σ', 'τ', 'υ',
]


# ── Reproduction ─────────────────────────────────────────────────────

# Brain parameters that get cloned + mutated when a parent spawns a
# child. `pain_embedding` (pain v4 / proprioception) is included here
# so that the brain's learned translation of pain → representations
# is heritable, the same way W_q is. With proprioception disabled
# across an entire run, this parameter is mutated but never used in
# computation, which is harmless.
BRAIN_PARAMS = ['embedding', 'W_q', 'W_k', 'W_v', 'W_o', 'W_out', 'b_out',
                'pain_embedding']


def spawn(parent, child_name, sigma=0.02, rng=None,
          wound_sigma=None):
    """Create a child from a parent.

    Brain weights are copied and mutated (Gaussian noise, scale sigma).
    Organs are fresh (they learn via Hebbian during training anyway).
    Battery starts full. Round counter resets. Pain is fresh.

    Scars are STRUCTURAL and HERITABLE. The child receives a deep
    copy of the parent's scar buffer — same scar directions, same
    enabled flag. Scar capacity loss is part of the inheritance:
    a child of a 4-scar parent starts life with 4 dimensions of
    brain_state already carved out. This is the substrate
    selection is supposed to operate on.

    Wound's TENDERNESS is HERITABLE. The child receives a copy of
    the parent's tenderness with one Gaussian-mutation step, clipped
    to [0, 1]. The mutation magnitude is wound_sigma (default: the
    Wound class's own default, currently 0.1). This is larger than
    the brain-weight sigma because tenderness is a single scalar on
    a unit interval and needs a meaningful per-spawn delta to allow
    selection to explore the range.
    """
    rng = rng or np.random.default_rng()

    child = Bittern(
        name=child_name,
        embed_dim=parent.brain.embed_dim,
        brain_dim=parent.brain.brain_dim,
        context=parent.brain.context,
        seed=int(rng.integers(0, 2**31)),
    )

    # Copy parent brain weights + mutate
    for pname in BRAIN_PARAMS:
        parent_w = getattr(parent.brain, pname)
        noise = rng.normal(0, sigma, parent_w.shape)
        setattr(child.brain, pname, parent_w.copy() + noise)

    # Inherit scars structurally. Deep-copy via clone_for_child.
    child.scars = parent.scars.clone_for_child()

    # Inherit wound tenderness. clone_for_child handles the Gaussian
    # mutation. When wound_sigma is None, the Wound class's own
    # default is used. The RNG draw happens unconditionally (even
    # when wound is disabled) — mirrors the pain_embedding pattern
    # and keeps the RNG state independent of toggle configuration.
    child.wound = parent.wound.clone_for_child(sigma=wound_sigma, rng=rng)

    return child


def _set_pain(c, enabled):
    """Helper: flip the pain master toggle on a creature."""
    c.pain.enabled = bool(enabled)


def _set_replay(c, enabled):
    """Helper: flip the painful-memory master toggle on a creature."""
    c.painful_memory.enabled = bool(enabled)


def _set_scars(c, enabled):
    """Helper: flip the scar master toggle on a creature.

    Note: this only affects whether scars are CONSULTED (projection
    + capture). The buffer's contents are preserved either way, so
    you can ablate a creature mid-experiment without destroying its
    accumulated scars."""
    c.scars.enabled = bool(enabled)


def _set_proprioception(c, enabled):
    """Helper: flip the proprioception (pain v4) master toggle.

    With enabled=False, Proprioception.get_pain_for_prefix returns
    None — Bittern.listen passes pain=None to brain methods, no
    pain term enters encode, no gradient flows to pain_embedding.
    The Brain.pain_embedding parameter is still present and still
    mutated by spawn() (harmlessly, since it's unused) so a single
    seed produces an identical creature regardless of toggle."""
    c.proprioception.enabled = bool(enabled)


def _set_wound(c, enabled):
    """Helper: flip the wound (pain v5) master toggle on a creature.

    With enabled=False, Wound.is_wounded() always returns False, so
    the listen() loop never zeros out the brain lr — behavior is
    bit-identical to the pre-v5 path. The tenderness parameter is
    still mutated by spawn() (one scalar per spawn, harmless when
    unused), so a single seed produces an identical creature
    regardless of toggle."""
    c.wound.enabled = bool(enabled)


# ── Competitive habitat (one generation) ─────────────────────────────

def run_generation(creatures, train_corpus, rounds, lr, block_size):
    """Train a population together in a competitive habitat.

    Winner gets 6 train pairs + battery, losers get 2 train pairs +
    bleed. Block-structured world for sustained winning/losing
    streaks.

    The loneliness 2x lr-multiplier was removed in this version
    (see module docstring for the diagnostic). Lonely creatures
    train at the same rate as everyone else; battery dynamics
    still affect emotion bias and the (now-disabled-able) pain
    system, but they no longer reach into raw lr.
    """
    # Build block-structured order
    by_class = {}
    for item in train_corpus:
        by_class.setdefault(item['class'], []).append(item)
    class_names = sorted(by_class.keys())
    class_ptrs = {cls: 0 for cls in class_names}

    block_corpus = []
    while len(block_corpus) < rounds + block_size * len(class_names):
        for cls in class_names:
            pool = by_class[cls]
            for _ in range(block_size):
                idx = class_ptrs[cls] % len(pool)
                block_corpus.append(pool[idx])
                class_ptrs[cls] += 1

    n = len(creatures)
    win_counts = {c.name: 0 for c in creatures}
    lonely_counts = {c.name: 0 for c in creatures}
    t0 = time.time()

    for r in range(rounds):
        # Progress bar
        if r % max(1, rounds // 40) == 0 or r == rounds - 1:
            pct = (r + 1) / rounds
            filled = int(pct * 30)
            bar = '█' * filled + '░' * (30 - filled)
            elapsed = time.time() - t0
            eta = (elapsed / (r + 1)) * (rounds - r - 1) if r > 0 else 0
            leader = max(win_counts, key=win_counts.get) if r > 0 else '...'
            print(f"\r    [{bar}] {pct:5.1%}  "
                  f"{elapsed:5.1f}s elapsed  ETA {eta:4.0f}s  "
                  f"leading: {leader}", end='', flush=True)

        world_bits = block_corpus[r % len(block_corpus)]['bits']

        # Score BEFORE learning (single-position, using predict_fast)
        score_prefix = world_bits[:min(8, len(world_bits) - 1)]
        score_target = int(world_bits[len(score_prefix)])
        scores = []
        for c in creatures:
            probs = c.brain.predict_fast(score_prefix)
            scores.append(-float(np.log(probs[score_target] + 1e-12)))

        winner_idx = int(np.argmin(scores))
        win_counts[creatures[winner_idx].name] += 1

        # Competitive learning + battery.
        # Loneliness lr-multiplier removed — see module docstring.
        for i, c in enumerate(creatures):
            is_winner = (i == winner_idx)
            tp = 6 if is_winner else 2
            c.listen(world_bits, train_pairs=tp, lr=lr)
            c.step()
            if is_winner:
                c.battery.replenish(0.06)
            else:
                c.battery.level = max(0.0, c.battery.level - 0.03)
            if c.battery.is_lonely():
                lonely_counts[c.name] += 1

    print()  # newline after progress bar
    return win_counts, lonely_counts


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_population(creatures, eval_corpus):
    """Evaluate each creature. Returns list of (creature, fitness_dict)."""
    results = []
    for c in creatures:
        tf = teacher_force_ce(c, eval_corpus)
        pc = per_class_ce(c, eval_corpus)
        classes_below_uniform = sum(1 for v in pc.values() if v < UNIFORM)
        results.append({
            'creature': c,
            'tf_ce': tf,
            'per_class': pc,
            'classes_below': classes_below_uniform,
        })
    # Sort by fitness: lower tf_ce = better
    results.sort(key=lambda r: r['tf_ce'])
    return results


# ── Main evolution loop ──────────────────────────────────────────────

PROBE_PROMPTS = [
    {'bits': [0, 1, 0, 1, 0, 1],          'class': 'steady_2'},
    {'bits': [0, 0, 0, 0, 0, 0, 1, 1],    'class': 'swelling'},
    {'bits': [1, 1, 0, 0, 0, 0, 0, 0],    'class': 'staccato'},
]


def evolve(pop_size=8, generations=10, rounds_per_gen=5000,
           seed=0, sigma=0.02, lr=0.02, block_size=20, out='babies',
           names=None, champion_name='champion',
           pain_enabled=True, replay_enabled=True,
           scars_enabled=True, scar_warmup=None,
           propio_enabled=True, wound_enabled=True,
           wound_sigma=None):
    """Run the tournament.

    scar_warmup: if not None, override every creature's
        scars.warmup with this value. None leaves the Scars
        default (1000) in place. Set to 0 to reproduce the
        original v3 (no warmup).

    propio_enabled: pain v4 (proprioception) master toggle. When
        False, the brain receives pain=None everywhere — no pain
        term enters encode, no gradient flows to pain_embedding.
        The pain_embedding parameter is still mutated by spawn()
        regardless, but it's unused when the toggle is off.

    wound_enabled: pain v5 (wound) master toggle. When False, the
        brain never silences during listen() — bit-identical to
        pre-v5 path. Tenderness is still mutated by spawn() (one
        scalar, harmless when unused).

    wound_sigma: per-spawn mutation magnitude for tenderness. If
        None, the Wound class's own default is used (currently
        0.1). Tenderness mutates more aggressively than brain
        weights because it's a single scalar on a unit interval.
    """
    rng = np.random.default_rng(seed)

    # Build corpora once (shared across all generations)
    corpus_rng = np.random.default_rng(seed + 500)
    train_corpus = make_corpus(per_class=200, length=64, rng=corpus_rng)
    eval_corpus = make_corpus(per_class=20, length=64, rng=corpus_rng)

    def _apply_warmup(c):
        if scar_warmup is not None:
            c.scars.warmup = int(scar_warmup)

    # Generation 0: all born naive
    print(f"\n{'='*65}")
    print(f"  EVOLUTION — {pop_size} creatures, {generations} generations")
    print(f"  {rounds_per_gen} rounds/gen, sigma={sigma}, block={block_size}")
    print(f"  pain:           {'ON' if pain_enabled else 'OFF (ablation)'}")
    print(f"  replay:         {'ON' if replay_enabled else 'OFF (ablation)'}")
    if scars_enabled:
        wm = scar_warmup if scar_warmup is not None else 'default(1000)'
        print(f"  scars:          ON  (warmup={wm})")
    else:
        print(f"  scars:          OFF (ablation)")
    print(f"  proprioception: {'ON' if propio_enabled else 'OFF (ablation)'}")
    if wound_enabled:
        ws = wound_sigma if wound_sigma is not None else 'default(0.1)'
        print(f"  wound (v5):     ON  (tenderness σ={ws})")
    else:
        print(f"  wound (v5):     OFF (ablation)")
    print(f"{'='*65}")

    creatures = []
    name_pool = names if names else NAME_POOL
    for i in range(pop_size):
        name = f"{GEN_PREFIX[0]}_{name_pool[i % len(name_pool)]}"
        c = Bittern(name=name, seed=seed + i)
        _set_pain(c, pain_enabled)
        _set_replay(c, replay_enabled)
        _set_scars(c, scars_enabled)
        _set_proprioception(c, propio_enabled)
        _set_wound(c, wound_enabled)
        _apply_warmup(c)
        creatures.append(c)

    champion_history = []

    for gen in range(generations):
        gen_label = GEN_PREFIX[min(gen, len(GEN_PREFIX)-1)]
        print(f"\n{'─'*65}")
        print(f"  Generation {gen} ({gen_label}): "
              f"{', '.join(c.name for c in creatures)}")
        print(f"{'─'*65}")

        # Train together
        t0 = time.time()
        wins, lonelies = run_generation(
            creatures, train_corpus, rounds_per_gen, lr, block_size)
        elapsed = time.time() - t0

        # Show habitat stats
        print(f"  training: {rounds_per_gen} rounds ({elapsed:.1f}s)")
        for c in creatures:
            w = wins.get(c.name, 0)
            l = lonelies.get(c.name, 0)
            pct_w = 100 * w / rounds_per_gen if rounds_per_gen > 0 else 0
            pct_l = 100 * l / rounds_per_gen if rounds_per_gen > 0 else 0
            print(f"    {c.name:12s}  wins={pct_w:4.0f}%  "
                  f"lonely={pct_l:4.0f}%  "
                  f"battery={c.battery.level:.2f}")

        # Evaluate
        results = evaluate_population(creatures, eval_corpus)
        print(f"\n  fitness ranking (lower CE = better):")
        for rank, r in enumerate(results):
            c = r['creature']
            marker = '♛' if rank == 0 else '·' if rank < pop_size // 2 else '✗'
            pc_str = '  '.join(
                f"{cls}={ce:.3f}" for cls, ce in sorted(r['per_class'].items()))
            sr = c.scars.report()
            sc_str = (f"scars={sr['size']}/{sr['capacity']}"
                      + (f" max_cos={sr['max_pairwise_cos']:+.2f}"
                         if sr['size'] >= 2 else ""))
            wr = c.wound.report()
            # Show tenderness always (the heritable substrate); show
            # wound activity only when the toggle is on.
            if wound_enabled:
                w_str = (f"tend={wr['tenderness']:.2f} "
                         f"wounds={wr['wounds_inflicted']} "
                         f"silenced={wr['steps_silenced']}")
            else:
                w_str = f"tend={wr['tenderness']:.2f}"
            print(f"    {marker} {rank+1}. {c.name:12s}  "
                  f"CE={r['tf_ce']:.3f}  {sc_str}  {w_str}  ({pc_str})")

        # Champion babble
        champ = results[0]['creature']
        print(f"\n  champion {champ.name} babbles:")
        for p in PROBE_PROMPTS:
            bb = champ.babble(p['bits'], n=24)
            print(f"    {p['class']:10s}  {bits_to_str(p['bits'])} → "
                  f"{bits_to_str(bb)}")

        # Champion scar trajectory: snapshot scar directions per gen.
        # Stored as a (k, brain_dim) array so we can later compare
        # whether independent lineages converge on similar scars.
        champ_dirs = champ.scars.directions()
        # Wound trajectory: tenderness is the heritable substrate
        # selection acts on. Tracking it across generations is the
        # primary readout of the v5 experiment.
        wr = champ.wound.report()
        champion_history.append({
            'gen': gen,
            'name': champ.name,
            'tf_ce': results[0]['tf_ce'],
            'per_class': results[0]['per_class'],
            'scars_size': champ.scars.report()['size'],
            'scars_directions': (champ_dirs.tolist()
                                  if champ_dirs is not None else []),
            'tenderness': wr['tenderness'],
            'wounds_inflicted': wr['wounds_inflicted'],
            'steps_silenced': wr['steps_silenced'],
        })

        # Selection + reproduction (skip on last generation)
        if gen < generations - 1:
            survivors = [r['creature'] for r in results[:pop_size // 2]]
            dead = [r['creature'] for r in results[pop_size // 2:]]
            print(f"\n  selection: {', '.join(c.name for c in survivors)} survive")
            print(f"             {', '.join(c.name for c in dead)} die")

            # Reproduce
            next_gen = []
            next_label = GEN_PREFIX[min(gen + 1, len(GEN_PREFIX)-1)]
            for i, parent in enumerate(survivors):
                # Parent survives. Reset emotional state (battery,
                # pain, painful_memory, round, recent_brain_losses)
                # so the parent enters the next generation as a peer
                # of its own children rather than a senior with stale
                # moods. The trained brain weights persist; everything
                # else is fresh.
                #
                # SCARS specifically are NOT touched by reset_emotional_state
                # — they are structural. A surviving parent enters
                # the next generation with the same scar buffer it
                # ended this generation with.
                parent_scars_before = len(parent.scars.vectors)
                parent_tenderness_before = parent.wound.tenderness
                parent.reset_emotional_state()
                assert len(parent.scars.vectors) == parent_scars_before, (
                    "reset_emotional_state should not touch scars")
                assert parent.wound.tenderness == parent_tenderness_before, (
                    "reset_emotional_state should not touch wound.tenderness")
                _set_pain(parent, pain_enabled)
                _set_replay(parent, replay_enabled)
                _set_scars(parent, scars_enabled)
                _set_proprioception(parent, propio_enabled)
                _set_wound(parent, wound_enabled)
                _apply_warmup(parent)
                next_gen.append(parent)

                # One child per parent. spawn() copies the parent's
                # scars deep into the child via clone_for_child, and
                # copies + mutates the parent's tenderness via
                # Wound.clone_for_child.
                child_name = (f"{next_label}_"
                              f"{name_pool[(i + pop_size//2) % len(name_pool)]}")
                child = spawn(parent, child_name, sigma=sigma, rng=rng,
                              wound_sigma=wound_sigma)
                _set_pain(child, pain_enabled)
                _set_replay(child, replay_enabled)
                _set_scars(child, scars_enabled)
                _set_proprioception(child, propio_enabled)
                _set_wound(child, wound_enabled)
                _apply_warmup(child)
                next_gen.append(child)
                inh = len(child.scars.vectors)
                tnd_p = parent.wound.tenderness
                tnd_c = child.wound.tenderness
                print(f"    {parent.name} → {child.name} "
                      f"(σ={sigma}, scars inherited={inh}, "
                      f"tend {tnd_p:.2f}→{tnd_c:.2f})")

            creatures = next_gen

    # Final summary
    print(f"\n{'='*65}")
    print(f"  EVOLUTION COMPLETE — champion trajectory")
    print(f"  pain:           {'ON' if pain_enabled else 'OFF (ablation)'}")
    print(f"  replay:         {'ON' if replay_enabled else 'OFF (ablation)'}")
    print(f"  scars:          {'ON' if scars_enabled else 'OFF (ablation)'}")
    print(f"  proprioception: {'ON' if propio_enabled else 'OFF (ablation)'}")
    print(f"  wound (v5):     {'ON' if wound_enabled else 'OFF (ablation)'}")
    print(f"{'='*65}")
    for h in champion_history:
        pc = h['per_class']
        pc_str = '  '.join(f"{cls}={ce:.3f}" for cls, ce in sorted(pc.items()))
        w_str = (f"tend={h['tenderness']:.2f} "
                 f"wounds={h['wounds_inflicted']}")
        print(f"  gen {h['gen']:2d}  {h['name']:12s}  CE={h['tf_ce']:.3f}  "
              f"scars={h['scars_size']}  {w_str}  ({pc_str})")

    # Final champion full probe
    final_champ = results[0]['creature']
    print(f"\n  final champion: {final_champ.name}")
    probe_report(final_champ, eval_corpus, PROBE_PROMPTS,
                 label=f"CHAMPION {final_champ.name}")
    # Final scar report
    print(f"\n  final champion scars: {final_champ.scars.report()}")
    # Final wound report — tenderness is the heritable trait that
    # selection has been acting on. wounds_inflicted / steps_silenced
    # describe the FINAL generation only (per-life counters reset at
    # generation boundary). Tenderness is the cross-generation story.
    print(f"  final champion wound: {final_champ.wound.report()}")

    # Pain v4 probe: compare calm vs sustained-pain babble on the
    # champion. If proprioception is off, this just produces two
    # identical babbles — diagnostic in itself.
    if propio_enabled:
        print(f"\n  pain v4 probe — calm vs sustained-pain babble:")
        for p in PROBE_PROMPTS:
            calm = final_champ.babble(p['bits'], n=24, induced_pain=None)
            pained = final_champ.babble(p['bits'], n=24, induced_pain=0.7)
            print(f"    {p['class']:10s}  {bits_to_str(p['bits'])}")
            print(f"      calm    → {bits_to_str(calm)}")
            print(f"      pained  → {bits_to_str(pained)}")
        # Norm of pain_embedding tells us whether the brain put any
        # learning into the pain channel at all.
        pe_norm = float(np.linalg.norm(final_champ.brain.pain_embedding))
        print(f"  pain_embedding norm: {pe_norm:.4f} "
              f"(init ≈ {0.1 * np.sqrt(final_champ.brain.embed_dim):.4f})")

    # Compare to a loner trained for the same total rounds
    total_rounds = rounds_per_gen * generations
    print(f"\n  control: loner trained {total_rounds} rounds (no competition)")
    loner = Bittern(name='loner', seed=seed + 999)
    _set_pain(loner, pain_enabled)
    _set_replay(loner, replay_enabled)
    _set_scars(loner, scars_enabled)
    _set_proprioception(loner, propio_enabled)
    _set_wound(loner, wound_enabled)
    _apply_warmup(loner)
    for r in range(total_rounds):
        item = train_corpus[r % len(train_corpus)]
        loner.listen(item['bits'], train_pairs=4, lr=lr)
        loner.step()
        loner.battery.replenish(0.02)
    probe_report(loner, eval_corpus, PROBE_PROMPTS,
                 label=f'loner @ {loner.round}')

    # Save champion
    os.makedirs(out, exist_ok=True)
    final_champ.save(os.path.join(out, champion_name))
    print(f"\n  saved champion -> {out}/{champion_name}")

    return final_champ, champion_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pop', type=int, default=8,
                   help='population size (must be even)')
    p.add_argument('--generations', type=int, default=10)
    p.add_argument('--rounds', type=int, default=5000,
                   help='training rounds per generation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sigma', type=float, default=0.02,
                   help='mutation magnitude')
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--block-size', type=int, default=20)
    p.add_argument('--out', default='babies')
    p.add_argument('--names', type=str, default=None,
                   help='comma-separated creature names (e.g. "zip,zap,zop,zug")')
    p.add_argument('--champion', type=str, default='champion',
                   help='save name for the winner (default: champion)')
    p.add_argument('--no-pain', action='store_true',
                   help='disable the pain system entirely (ablation control)')
    p.add_argument('--no-replay', action='store_true',
                   help='disable the painful-memory replay buffer (ablation control)')
    p.add_argument('--no-scars', action='store_true',
                   help='disable the scar system entirely (ablation control)')
    p.add_argument('--scar-warmup', type=int, default=None,
                   help='rounds before scar capture is enabled '
                        '(default: 1000; set to 0 to reproduce '
                        'original v3)')
    p.add_argument('--no-proprioception', action='store_true',
                   help='disable the pain v4 (proprioception) channel '
                        '(ablation control)')
    p.add_argument('--no-wound', action='store_true',
                   help='disable the pain v5 (wound) silencing '
                        '(ablation control)')
    p.add_argument('--wound-sigma', type=float, default=None,
                   help='per-spawn mutation magnitude for tenderness '
                        '(default: Wound class default, 0.1)')
    args = p.parse_args()

    name_list = args.names.split(',') if args.names else None

    evolve(pop_size=args.pop, generations=args.generations,
           rounds_per_gen=args.rounds, seed=args.seed,
           sigma=args.sigma, lr=args.lr, block_size=args.block_size,
           out=args.out, names=name_list, champion_name=args.champion,
           pain_enabled=not args.no_pain,
           replay_enabled=not args.no_replay,
           scars_enabled=not args.no_scars,
           scar_warmup=args.scar_warmup,
           propio_enabled=not args.no_proprioception,
           wound_enabled=not args.no_wound,
           wound_sigma=args.wound_sigma)


if __name__ == '__main__':
    main()
