# bittern — handoff for next session (post-colony)

## What this project is — the spirit

This is a lineage of **minimum-complexity creatures** that learn from streams of symbols. They are not language models. They are not being trained to perform a task. They are organisms that live in a world of symbols, learn from experience, and develop behavior.

The founding project is **glaud-i** — a single-organism transformer-based creature built in pure numpy with hand-derived backprop. The architectural philosophy, which every successor must preserve:

1. **Multiple learning signals in tension.** The brain learns by gradient descent on next-symbol prediction. The organs learn by Hebbian update on lateral and reward-shaped sensitivity. These are different training paths into the same creature.

2. **An inside.** The battery decays each round. Below threshold the creature is "lonely," which shifts emotion bias, which shifts organ activation. The instinct emerges from representation, not rules.

3. **Hand-derived backprop in numpy.** No frameworks. The point is that the math is yours.

4. **Probes you can read.** Every creature has diagnostics that tell you what's actually happening inside.

5. **Learning on its own, not being taught.** The creature listens to a world and develops internal structure from that experience. It is not supervised in the ML sense — it is an organism encountering stimuli. The listen/learn loop is meant to feel like living, not like training.

Reference posts:
- glaud-i Milestone 1: https://harry.ahlas.com/post/2026-04-24-glaud-i-milestone-01/
- clade Milestone 1: https://harry.ahlas.com/post/2026-04-29-clade-milestone-01/
- glaud-i repo (archived): https://github.com/harryahlas/glaud-i

## The lineage so far

### glaud-i
Single-organism transformer creature, vocab=27 (a-z + space). Pure numpy. Hand-derived backprop on a real transformer. Archived after reaching its milestone.

### clade
Two same-architecture babies, vocab=27. Key finding (M3): **training loss drops while the brain becomes confidently wrong about its own outputs.** Teacher-forced cross-entropy improves, but free-running CE (where the creature hears its own babble) rises past uniform — the creature memorizes corpus patterns without learning to generalize to its own behavior. Two attempted fixes (4-D world grounding, whole-sequence reward) failed. The free-running probe is clade's main contribution.

### bittern
Vocab=2 cousin of clade. Single bit per tick. Born to test whether clade's exposure-bias pattern appears on a binary substrate where ln(2) ≈ 0.693 is the reference for every CE measurement. ~700 lines of numpy including hand-derived backprop on a single-head causal-attention brain.

**What bittern built:**
- Brain (single-head causal attention with hand-derived backprop, gradcheck to ~1e-11)
- Organs (two organs that compete to fire, fixed after organ_diag found 3.5% → 100% agreement)
- Battery (decay/replenish drive)
- World (bit-rhythm generators: steady_2, swelling, staccato, plus others)
- Probes (teacher-force CE, free-running CE, noise-recovery CE, per-class breakdown)
- The full tool suite (train.py, chat.py, gradcheck.py, organ_diag.py)

**What bittern found:**
1. The clade M3 finding reproduces on binary substrate, faster and sharper.
2. Organs were anti-correlated with the brain (3.5% agreement). Fixed by dropping dead momentum, scaling fatigue, adding supervised sensitivity learning. Agreement → 100%.
3. Continued training past ~65k rounds destroys the bittern (constant-output attractor collapse). The 65k checkpoint is the good one.

**The named open problem (M2):** Brain class-marginal collapse. Even at the 65k peak, swelling and staccato per-class CE never dropped below uniform — the brain learned to predict the class marginal (always 0) rather than actual rhythm structure.

### pipit (a sibling, not a successor)
Replaced the entire neural network brain with **coupled phase oscillators and Hebbian phase memory**. No layers, no weight matrices, no backpropagation. Prediction by von-Mises similarity (resonance detection), learning by Hebbian phase association. Pure numpy.

**What pipit proved:**
1. **Exposure bias is structural, not a training bug.** The listen/babble split is the cause. Any creature with separate training and generation modes will have fr_ce > tf_ce eventually. Pipit has no such split (one loop: tick), and fr_ce stayed below tf_ce across all training — a structural guarantee.
2. **Single-class entrainment works without any neural network.** A pipit trained on pure steady_2 gets CE = 0.353 and babbles perfect alternation.
3. **Multi-class prediction requires conditional computation.** Six readout variants were tried. The oscillators entrained correctly but the readout couldn't determine which oscillator to trust for the current input.
4. **Colony of specialists with competitive selection works architecturally.** Three single-class pipits, each listening to the same input, winner selected by lowest recent loss. The routing worked perfectly. But individual specialists were too weak.
5. **Two pipits hearing each other converge toward shared sparse output.** Emergent co-silence from mutual phase-memory accumulation.

## What was done this session

Three interventions were tested independently, as prescribed by the handoff. One at a time, no stacking.

### Intervention 1: Balanced target sampling — PARTIAL WIN

**Problem addressed:** M2 class-marginal collapse. Staccato corpus is 74.9% zeros, swelling is 52.8% zeros. The brain learns to always predict 0 for these classes because the corpus gradient pulls toward the marginal.

**Fix:** Added `balanced=True` parameter to `listen()` in `bittern.py`. When enabled, split points are sampled so that target=0 and target=1 are equally represented — half from the 0-pool, half from the 1-pool. This directly removes the gradient bias without changing the rhythm generators. Added `--balanced` flag to `train.py`.

**Result (b_balanced, 50k rounds, seed=0):**

| Class    | b1 @ 65k | b_balanced @ 50k |
|----------|----------|------------------|
| staccato | 0.738 ✗  | 0.689 ·          |
| steady_2 | 0.215 ✓  | 0.346 ✓          |
| swelling | 0.651 ·  | 0.594 ·          |

- Staccato CE crossed from above uniform to below — the ✗ became ·
- Staccato babble changed from constant-0 (`000000000000`) to structured bursts (`000010100001000000100010`)
- Swelling CE improved but babble still constant-1
- steady_2 CE regressed but stayed ✓
- The 25k checkpoint had better per-class numbers (staccato 0.623, swelling 0.482), suggesting the balanced sweet spot may be earlier

**Assessment:** Balanced target sampling partially breaks the class-marginal collapse. It's strongest for staccato (which had the worst imbalance at 75/25). Swelling (53/47) gets moderate help. The fix is in the right direction but insufficient on its own.

### Intervention 2: Self-hearing (scheduled sampling) — NO CLEAR WIN

**Problem addressed:** Exposure bias. The brain trains on real prefixes but generates from its own distribution.

**Fix:** Added `self_hear_p` parameter to `listen()` in `bittern.py`. With probability self_hear_p, each prefix bit is replaced by the brain's own sampled prediction from the preceding context. Added `--self-hear` flag to `train.py`.

**Result (b_selfhear, 50k rounds, seed=0, p=0.1):**

| Class    | b1 @ 65k | b_selfhear @ 50k |
|----------|----------|-------------------|
| staccato | 0.738 ✗  | 0.620 ·           |
| steady_2 | 0.215 ✓  | 0.467 ·           |
| swelling | 0.651 ·  | 0.552 ·           |

- fr_ce/tf_ce gap narrowed slightly (from -0.239 to -0.148 at ~50k)
- But babble quality regressed: steady_2 lost its perfect alternation (became all-1s)
- Per-class staccato improved but steady_2 badly regressed

**Assessment:** The handoff's concern about fr_ce rising above tf_ce doesn't manifest in bittern. fr_ce is always BELOW tf_ce because the constant-output attractor is self-consistent (predict 0, babble 0, not surprised). Pipit's structural guarantee came from having no listen/babble split at all — bolting scheduled sampling onto bittern's split is insufficient. Self-hearing at p=0.1 adds noise to training without addressing the real bottleneck (class-marginal collapse or multi-class interference).

**Possible follow-up:** Higher self-hearing rates (p=0.3, 0.5) or a completely different implementation where the creature alternates between listen and babble episodes during training (closer to pipit's architecture). Not attempted this session.

### Intervention 3: Colony of bittern specialists — CLEAR WIN

**Problem addressed:** Multi-class prediction from a single brain.

**Fix:** `colony_bittern.py` — trains three bitterns, each on a single rhythm class, for 65k rounds each. At inference, all three hear the input. The specialist with the lowest EMA prediction loss speaks. This is pipit's colony architecture with bittern's much stronger attention brain.

**Result (colony, 65k rounds per specialist):**

| Class    | b1 @ 65k | Colony  | Individual specialist |
|----------|----------|---------|----------------------|
| staccato | 0.738 ✗  | 0.625 · | 0.426                |
| steady_2 | 0.215 ✓  | 0.043 ✓ | 0.023                |
| swelling | 0.651 ·  | 0.387 · | 0.370                |
| overall  | 0.535    | 0.352   |                      |

Winner selection accuracy: steady_2 98%, swelling 98%, staccato 90%.

- **Every class improved dramatically.** Colony CE 0.352 vs b1 CE 0.535.
- **Routing works perfectly.** The right specialist wins for the right class ≥90% of the time.
- **Swelling specialist learned actual transitions.** Babble: `11111111100011111111111110000011` — long-run phase transitions instead of constant-1. This is the session's biggest qualitative finding.
- **steady_2 specialist nearly perfect.** CE = 0.023, perfect alternation babble.
- **Staccato specialist still suffers marginal collapse in babble** (teacher-forced CE is good at 0.426, but babble is still mostly-0s). This is the exposure bias manifesting within the specialist — it performs well on real staccato context but collapses when hearing its own output.

## Named open problems

### M2 (updated): Staccato specialist babble collapse
The class-marginal collapse is now isolated to a single specialist. The staccato specialist achieves CE = 0.426 (well below uniform) in teacher-forced mode but babbles mostly-0s. This is exposure bias within a single-class specialist: the brain learns staccato structure from real prefixes but can't maintain it when hearing its own output. Balanced target sampling (Intervention 1) partially addresses this — combining balanced training with the colony is the natural next step.

### M3: Colony is not one creature
The colony solves the multi-class prediction problem but the three specialists don't share state. Each has independent weights, independent organs, independent batteries. A true bittern successor would need a single brain that can do what the colony does. The colony establishes the target performance — the next architecture needs to match it without specialist separation.

### M4: Swelling specialist phase inversion
The swelling specialist babbles long-run transitions (`111111111100001111111111100000`) which shows it learned the right *shape* (long-on, long-off) but the transitions don't always match the correct phase of the input swelling pattern. The specialist has learned the statistical structure but not the phase alignment.

## Current state of code

### Files
- `bittern.py` — the creature. Updated with `balanced` and `self_hear_p` parameters to `listen()`.
- `world.py` — bit-rhythm generators + corpus builder. Unchanged.
- `probe.py` — diagnostic report. Unchanged.
- `train.py` — training loop. Updated with `--balanced` and `--self-hear` flags.
- `chat.py` — interactive bit-stream chat. Unchanged.
- `gradcheck.py` — gradient verification. Passes to ~1e-11.
- `organ_diag.py` — organ-brain alignment diagnostic. Unchanged.
- `recover_b1_at_65k.py` — restores the b1 checkpoint. Unchanged.
- `colony_bittern.py` — **NEW.** Colony of bittern specialists with competitive routing. Train + evaluate in one script.
- `pipit.py` — oscillator creature. Reference only.
- `colony.py` — pipit colony. Reference only.

### Saved bitterns in babies/
- `b1` — the original bittern, 65k rounds, the reference point
- `b_balanced` — balanced target sampling, 50k rounds
- `b_selfhear` — self-hearing p=0.1, 50k rounds
- `b_steady_2` — colony specialist, 65k rounds on steady_2 only
- `b_swelling` — colony specialist, 65k rounds on swelling only
- `b_staccato` — colony specialist, 65k rounds on staccato only

### Verification commands
```bash
python gradcheck.py                          # should report ALL OK
python organ_diag.py b1                      # should report 100% agreement
python organ_diag.py b_balanced              # should report 100% agreement
python colony_bittern.py --rounds 65000      # reproduces colony results (~3 min)
```

## What to do next session

### 1. Colony + balanced staccato specialist (natural combination)
The staccato specialist's babble collapse is the same class-marginal problem that balanced sampling partially fixed. Train a staccato specialist with `balanced=True` and see if its babble improves while maintaining the colony's routing accuracy.

```python
# In colony_bittern.py, modify train_specialist to accept balanced=True
# Train only the staccato specialist with balanced=True
# Replace it in the colony, re-evaluate
```

This is not "stacking fixes" — it's applying Intervention 1's isolated finding to the specific specialist that needs it.

### 2. Habitat architecture (one brain, conditional routing)
The colony proves the target performance. The next architectural question: can one bittern brain learn to match the colony's performance with some form of conditional computation? Ideas:

- **Context-dependent readout heads.** One brain, three W_out heads. A lightweight classifier (trained on organ signals or lateral activity) selects which head to use.
- **Rhythmic mode embedding.** Augment the brain's input with a mode vector that encodes which rhythm class the creature thinks it's hearing. The mode vector is updated by a running estimate, not a ground-truth label.

Both of these add complexity. The colony establishes what "good enough" looks like, so any unified architecture should be measured against colony performance.

### 3. Two-creature social dynamics (longer-term)
Two bittern colonies hearing each other. Pipit found that two pipits converge toward shared sparse output through mutual phase-memory accumulation. Does the same happen with attention-brain creatures? The colony architecture makes this tractable — each creature is a colony of specialists, and mutual hearing could create interesting competitive/cooperative dynamics between colonies.

## What NOT to do
- **Don't try to make self-hearing work by increasing p.** The experiment showed that exposure bias in bittern manifests as constant-output attractor collapse, not the fr_ce > tf_ce pattern from clade. The structural problem is that the brain's own output is self-consistent with its marginal prediction, so hearing more of its own voice just reinforces the marginal.
- **Don't modify the staccato/swelling generators to be balanced.** The bit imbalance is a PROPERTY of these rhythms. Staccato IS mostly zeros. The fix belongs in the learning algorithm, not the world.
- **Don't train past the sweet spot.** The colony specialists are at 65k. The b_balanced sweet spot was ~25k. Probe at checkpoints and save when per-class CE starts rising.
- **Don't intervene on organs.** Organ agreement is 100% across all experiments. The organs are doing their job.

## Pattern from the lineage (reinforced)

Prior sessions went 1-for-3 on speculative experimental directions. This session went 1-for-3 as well (colony won, balanced partial, self-hearing no). The wins came from addressing structural issues with structural solutions (colony = separate the classes that the single brain can't handle). The partial came from addressing a real problem (gradient bias) with a direct fix. The miss came from importing a finding from a different architecture (pipit's no-split guarantee) that doesn't transfer to bittern's split architecture.

**Updated pattern: structural solutions to structural problems win. Algorithmic tweaks to fix structural problems don't.**
