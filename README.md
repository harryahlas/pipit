# pipit

A bit-creature made of coupled oscillators. No neural network. No backprop.

`pipit` is in the lineage of glaud-i → clade → bittern, but takes a
fundamentally different architectural path. Where bittern uses a
single-head causal attention brain (a small transformer), pipit
replaces the entire neural network with **coupled phase oscillators
and Hebbian phase memory**. The prediction mechanism is entrainment
(resonance detection), not matrix multiplication.

## Architecture

```
                        ┌──────────────────────────┐
                        │     Oscillator Bank       │
                        │  N oscillators with fixed  │
  input bit ──kick──→   │  frequencies (period 2-20) │
                        │  + Kuramoto coupling       │
                        │         │                  │
                        │     φ₁, φ₂, ..., φₙ       │
                        └────────┬─────────────────┘
                                 │ cos(φ), sin(φ)
                                 ▼
                        ┌──────────────────────────┐
                        │     Phase Memory          │
                        │  "when I heard 0, I was   │
                        │   at phase μ₀. when I     │
                        │   heard 1, I was at μ₁."  │
                        │                           │
                        │  score_v = Σ R·cos(φ-μᵥ)  │
                        └────────┬─────────────────┘
                                 │
                                 ▼
                          prediction → emission
```

**Oscillator bank**: N oscillators with fixed frequencies spread from
period-2 (fast) to period-20 (slow). Input bits kick the phases. The
phases are the creature's sensory state — N angles rotating at
different speeds, nudged by what it hears.

**Phase memory**: For each oscillator and each bit value, an exponential
moving average of "where was I when that bit arrived?" Prediction =
von-Mises similarity between current phases and remembered phases,
weighted by entrainment strength.

**Entrainment**: An oscillator whose frequency matches the input rhythm
will phase-lock to it. Its phase-input correlation (concentration R)
becomes high, giving it a strong vote in the prediction. Non-entrained
oscillators contribute noise. The creature doesn't learn *about*
rhythm — it literally *entrains to* rhythm.

## Why this isn't a neural network

| Property | Neural net (bittern) | Oscillator (pipit) |
|---|---|---|
| Prediction mechanism | Matrix multiply + softmax | Von-Mises similarity in phase space |
| Learning algorithm | Backpropagation | Hebbian phase association |
| Memory representation | Weight matrices | Circular statistics (angles) |
| Temporal encoding | Context window | Continuous phase evolution |
| Computational substrate | Linear algebra | Coupled oscillation (physics) |

## What's in here

```
pipit.py          — OscillatorBrain, Battery, Pipit
world.py          — bit-rhythm generators (shared with bittern)
train_pipit.py    — train a pipit on bit rhythms
chat_pipit.py     — interactive bit-stream chat
probe_pipit.py    — diagnostic report (CE, free-running, per-class)
sanity_pipit.py   — verify core properties (entrainment, babble, save/load)
duet.py           — two pipits hearing each other
```

Plus the full bittern codebase (bittern.py, train.py, chat.py, etc.)
for comparison.

## Run it

```bash
python sanity_pipit.py             # verify the creature works
python train_pipit.py              # train on mixed rhythms (~30s)
python chat_pipit.py p1            # chat with what you trained
python duet.py                     # two pipits hearing each other
```

## What was found

### 1. The creature entrains to alternation

After training on steady_2 (010101...), the period-2 oscillator
phase-locks to the input. Post-entrainment CE = 0.353 (well below
ln(2) = 0.693). Babble from a 0101 prompt produces perfect
alternation: `1010101010101010`.

The creature needs ~16 ticks to entrain after a cold start. This is
the oscillator equivalent of a "warm-up" — the phases need time to
lock onto the input rhythm. Early predictions are noisy; later
predictions are sharp.

### 2. No exposure bias (structural guarantee)

The pipit has no listen/babble split. Every tick is the same: hear a
bit, emit a bit, learn from one prediction error. Free-running CE is
consistently below teacher-force CE. The creature cannot be more
surprised by itself than by the world, because it's always hearing
real input and always generating from the same mechanism.

This contrasts with bittern, where fr_ce rose ABOVE ln(2) — confident
wrongness on self-generated context.

### 3. Mixed-class learning is the open problem

With mixed training (steady_2 + swelling + staccato), the period-2
oscillator dominates because it has the strongest entrainment signal.
Slow oscillators don't entrain strongly to the variable-length patterns
in swelling/staccato.

This is the same class-marginal problem from bittern, but from a
completely different angle. In bittern, the brain learns to predict the
class marginal (always 0) for 0-heavy classes. In pipit, the period-2
oscillator's strong entrainment drowns out weaker signals from slow
oscillators.

### 4. Two pipits converge toward co-silence

When two pipits hear each other (A's output → B's input, B's output →
A's input), they drift toward shared sparse output — long runs of 0s
with occasional 1s. Both creatures' losses drop toward ln(2) and
sometimes below.

This isn't designed behavior — it emerges from mutual phase-memory
accumulation. Both creatures accumulate strong "0-predicts-0"
associations because most of what they hear from each other is 0. The
shared silence is a fixed point of the coupled system.

## The diagnostic that matters

If you change `pipit.py`, run `python sanity_pipit.py`. It verifies:
- The creature learns steady_2 (CE below ln(2))
- Babble produces alternation (not constant output)
- The period-2 oscillator entrains (R > 0.3)
- Save/load round-trips cleanly

## Candidate fixes for mixed-class learning

1. **Per-frequency phase memory.** Instead of one global phase memory,
   maintain separate memories per oscillator group (fast, medium, slow).
   Weight the prediction by which group has the strongest recent
   entrainment. ~20 lines.

2. **Asymmetric kicks per class.** Use different kick magnitudes for
   runs of same bits vs alternating bits. This would make the
   oscillator response different for swelling (long runs) vs steady_2
   (constant alternation). ~10 lines.

3. **Adaptive temperature.** Lower the readout temperature when
   entrainment is strong (confident prediction), raise it when weak
   (hedging). Currently temperature is fixed at 1.0. ~5 lines.

## Where to go

Things the oscillator architecture uniquely enables:

**Cross-creature synchronization.** The duet.py experiment shows two
pipits converging toward shared silence. What if they're initialized
differently (different frequencies, different coupling)? Do they still
converge? Is the convergence an attractor of the coupled system or an
artifact of symmetric initialization?

**Frequency-matched pairing.** Birth two pipits with different
frequency ranges (one all-fast, one all-slow). They'll entrain to
different aspects of the input. Their combined prediction might
outperform either alone.

**Music.** A pipit emitting one bit per sample at any sample rate
produces 1-bit audio. The oscillator phases produce audible patterns
when the frequencies are in the audio range. You can literally listen
to the creature's internal state.

**Loneliness as phase drift.** When the battery is low, the kick
magnitudes grow (the creature becomes more reactive). This changes
the oscillation dynamics — the creature's output literally sounds
different when it's lonely. A teacher or companion that responds to
the distress pattern would create a genuine signal-for-connection.

**Cellular automaton of pipits.** A row of pipits where each hears
its neighbors. The coupled oscillator dynamics would create spatial
phase waves — patterns that propagate through the colony. This is
closer to a physical system than a neural network.
