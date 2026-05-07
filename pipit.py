"""
pipit.py — a creature made of coupled oscillators and phase memory.

A pipit lives tick-by-tick. Each tick, one bit arrives, one bit is
emitted. There is no listen/babble split. There are no neural
network layers. There is no backpropagation.

Architecture:

  OscillatorBrain — two mechanisms:

  1. Oscillator bank (fixed structure):
      N oscillators with fixed natural frequencies (geometric spread
      from period-2 to period-20). Input bits kick the phases through
      learned asymmetric kicks. Kuramoto coupling (fixed) allows the
      oscillators to influence each other. The phases ARE the creature's
      sensory state — N angles rotating at different speeds, nudged by
      what it hears.

  2. Phase memory — two modes:

      'ema' (exponential moving average):
          For each oscillator i and each vocab item v, an EMA of
          cos(φ_i) and sin(φ_i) at the moment v arrives. Controlled
          by mem_alpha (0 = instant, 1 = infinite memory).
          Low alpha (~0.6) = adapts to current rhythm quickly.
          High alpha (~0.97) = long-term average, slow to change.

      'episodic' (circular buffer):
          A ring buffer of the last K (bit, cos(φ), sin(φ)) tuples.
          Prediction compares current phases to the circular mean of
          recent observations for each bit value. Naturally adapts to
          whatever rhythm is currently playing because the buffer only
          holds recent history.

  3. Kick learning (Hebbian reinforcement):
      The input kicks are adjusted based on prediction error.

  Battery:
      Decays each tick. Modulates kick magnitude.

Pure numpy. ~400 lines.
"""

from __future__ import annotations
import json
import numpy as np


VOCAB_SIZE = 2
DEFAULT_N_OSC = 16


class OscillatorBrain:
    """N coupled phase oscillators with configurable phase memory.

    Fixed: frequencies, coupling.
    Learned (Hebbian): phase-memory associations, input kick magnitudes.

    memory_mode='ema':      EMA phase memory (set mem_alpha for decay speed)
    memory_mode='episodic': circular buffer of last buf_size observations
    """

    def __init__(self, n_osc=DEFAULT_N_OSC, memory_mode='ema',
                 mem_alpha=0.97, mem_alpha_short=0.5, buf_size=32,
                 rng=None):
        self.n_osc = n_osc
        self.memory_mode = memory_mode
        self.rng = rng if rng is not None else np.random.default_rng()

        # --- Oscillator bank (fixed structure) ---
        periods = np.geomspace(2.0, 20.0, n_osc)
        self.omega = 2 * np.pi / periods
        self.omega[1::2] *= -1   # alternate rotation directions

        scale = 0.03 / np.sqrt(n_osc)
        self.K = self.rng.normal(0, scale, (n_osc, n_osc))
        np.fill_diagonal(self.K, 0.0)

        # --- Input kicks (learned, Hebbian) ---
        self.kick = np.zeros((n_osc, VOCAB_SIZE))
        self.kick[:, 0] = -0.3
        self.kick[:, 1] = +0.3
        self.kick += self.rng.normal(0, 0.1, (n_osc, VOCAB_SIZE))
        self.kick_lr = 0.001

        # --- Long-term EMA phase memory (direction) ---
        self.mem_alpha = mem_alpha
        self.mem_cos = np.zeros((n_osc, VOCAB_SIZE))
        self.mem_sin = np.zeros((n_osc, VOCAB_SIZE))

        # --- Short-term EMA (confidence / coherence) ---
        # Used by 'dual' mode: tracks recent phase-input consistency.
        # High R_short = this oscillator is entrained RIGHT NOW = trust it.
        # Low R_short = drifting = ignore its vote.
        self.mem_alpha_short = mem_alpha_short
        self.short_cos = np.zeros((n_osc, VOCAB_SIZE))
        self.short_sin = np.zeros((n_osc, VOCAB_SIZE))

        # --- Episodic buffer ---
        self.buf_size = buf_size
        self.buf_cos = np.zeros((buf_size, n_osc))
        self.buf_sin = np.zeros((buf_size, n_osc))
        self.buf_bit = np.full(buf_size, -1, dtype=np.int64)
        self.buf_ptr = 0
        self.buf_count = 0

        # Readout temperature
        self.temperature = 1.0

        # --- State ---
        self.phi = self.rng.uniform(0, 2 * np.pi, n_osc)
        self._last_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        self._has_prediction = False

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _softmax(x, temp=1.0):
        x = x / max(temp, 1e-6)
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _phase_features(self):
        f = np.empty(2 * self.n_osc)
        f[0::2] = np.cos(self.phi)
        f[1::2] = np.sin(self.phi)
        return f

    def brain_dim(self):
        return 2 * self.n_osc

    def get_state(self):
        return self._phase_features()

    # -- prediction ----------------------------------------------------

    def _predict_ema(self):
        """Predict from EMA phase memory (von-Mises similarity)."""
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        scores = np.zeros(VOCAB_SIZE)
        for v in range(VOCAB_SIZE):
            scores[v] = np.sum(cos_phi * self.mem_cos[:, v]
                               + sin_phi * self.mem_sin[:, v])
        return self._softmax(scores, self.temperature)

    def _predict_episodic(self):
        """Predict from episodic buffer (nearest-neighbor in phase space)."""
        if self.buf_count == 0:
            return np.ones(VOCAB_SIZE) / VOCAB_SIZE

        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)

        # Only look at valid entries
        valid = min(self.buf_count, self.buf_size)
        bc = self.buf_cos[:valid]
        bs = self.buf_sin[:valid]
        bb = self.buf_bit[:valid]

        scores = np.zeros(VOCAB_SIZE)
        for v in range(VOCAB_SIZE):
            mask = (bb == v)
            n_v = mask.sum()
            if n_v == 0:
                continue
            mean_cos = bc[mask].mean(axis=0)  # (n_osc,)
            mean_sin = bs[mask].mean(axis=0)  # (n_osc,)
            scores[v] = np.sum(cos_phi * mean_cos + sin_phi * mean_sin)

        return self._softmax(scores, self.temperature)

    def _predict_dual(self):
        """Predict using two timescales: long-term direction + short-term confidence.

        Long-term EMA (α=0.97): learned direction (what phase does each
        oscillator prefer for each bit value).

        Short-term EMA (α~0.5): current confidence. An oscillator is
        confident when it has:
          - High R for both bit values (consistent phase-input relationship)
          - Well-separated directions for 0 vs 1 (can discriminate)

        A slow oscillator that barely moves within a sequence has high R
        but LOW separation — it looks the same for both bits. It gets
        low confidence despite high consistency. That's the key: inertia
        is not entrainment.

        Prediction:
            score_v = Σ_i  confidence_i  *  cos(φ_i - μ_long_i[v])
        """
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)

        # Short-term confidence: R_short for each bit value
        R_short = np.sqrt(self.short_cos**2 + self.short_sin**2)  # (n_osc, 2)

        # Angular separation between short-term preferred phases for 0 vs 1
        # μ_short[v] = atan2(short_sin[:, v], short_cos[:, v])
        # separation = |sin(μ_short[0] - μ_short[1])| via cross-product
        # sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        R0_safe = np.maximum(R_short[:, 0], 1e-12)
        R1_safe = np.maximum(R_short[:, 1], 1e-12)
        sin_sep = np.abs(
            (self.short_sin[:, 0] / R0_safe) * (self.short_cos[:, 1] / R1_safe)
            - (self.short_cos[:, 0] / R0_safe) * (self.short_sin[:, 1] / R1_safe)
        )  # (n_osc,)  0 = same direction, 1 = perpendicular

        # Confidence = strong R for both bits AND well-separated directions
        confidence = R_short[:, 0] * R_short[:, 1] * sin_sep  # (n_osc,)

        scores = np.zeros(VOCAB_SIZE)
        for v in range(VOCAB_SIZE):
            # Long-term direction (normalized)
            R_long = np.sqrt(self.mem_cos[:, v]**2 + self.mem_sin[:, v]**2)
            R_long_safe = np.maximum(R_long, 1e-12)
            dir_cos = self.mem_cos[:, v] / R_long_safe
            dir_sin = self.mem_sin[:, v] / R_long_safe

            # similarity = cos(φ - μ_long)
            similarity = cos_phi * dir_cos + sin_phi * dir_sin

            # Weight by discrimination confidence
            scores[v] = np.sum(confidence * similarity)

        return self._softmax(scores, self.temperature)

    def _predict(self):
        if self.memory_mode == 'ema':
            return self._predict_ema()
        elif self.memory_mode == 'dual':
            return self._predict_dual()
        else:
            return self._predict_episodic()

    # -- core: one tick ------------------------------------------------

    def tick(self, input_bit, learn=True, kick_scale=1.0):
        """Process one input bit. Returns (prediction_probs, loss)."""

        # -- 1. Score last prediction --
        if self._has_prediction:
            loss = -np.log(self._last_probs[input_bit] + 1e-12)
        else:
            loss = -np.log(0.5)
            self._has_prediction = True

        # -- 2. Update memory --
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)

        # Short-term EMA ALWAYS updates — it's sensing, not learning.
        # "What am I hearing right now?" is needed for prediction even
        # during probing.
        if self.memory_mode == 'dual':
            a_s = self.mem_alpha_short
            self.short_cos[:, input_bit] = (a_s * self.short_cos[:, input_bit]
                                            + (1 - a_s) * cos_phi)
            self.short_sin[:, input_bit] = (a_s * self.short_sin[:, input_bit]
                                            + (1 - a_s) * sin_phi)

        # Long-term EMA and episodic buffer only update when learning.
        if learn:
            if self.memory_mode in ('ema', 'dual'):
                a = self.mem_alpha
                self.mem_cos[:, input_bit] = (a * self.mem_cos[:, input_bit]
                                              + (1 - a) * cos_phi)
                self.mem_sin[:, input_bit] = (a * self.mem_sin[:, input_bit]
                                              + (1 - a) * sin_phi)
            if self.memory_mode == 'episodic':
                self.buf_cos[self.buf_ptr] = cos_phi
                self.buf_sin[self.buf_ptr] = sin_phi
                self.buf_bit[self.buf_ptr] = input_bit
                self.buf_ptr = (self.buf_ptr + 1) % self.buf_size
                self.buf_count = min(self.buf_count + 1, self.buf_size)

        # -- 3. Kick adjustment (Hebbian reinforcement) --
        if learn and self._has_prediction:
            error = 1.0 - self._last_probs[input_bit]
            if error > 0.3:
                if self.memory_mode in ('ema', 'dual'):
                    R = np.sqrt(self.mem_cos[:, input_bit]**2
                                + self.mem_sin[:, input_bit]**2)
                else:
                    R = np.zeros(self.n_osc)
                need = (1.0 - R) * error
                other = 1 - input_bit
                self.kick[:, input_bit] += self.kick_lr * need
                self.kick[:, other] -= self.kick_lr * need
                self.kick = np.clip(self.kick, -2.0, 2.0)

        # -- 4. Evolve oscillators --
        self.phi += self.kick[:, input_bit] * kick_scale
        diff = self.phi[None, :] - self.phi[:, None]
        coupling = np.sum(self.K * np.sin(diff), axis=1)
        self.phi += self.omega + coupling

        # -- 5. Predict --
        probs = self._predict()
        self._last_probs = probs
        return probs, float(loss)

    # -- prediction (no state change) ---------------------------------

    def predict_probs(self):
        return self._predict()

    def predict_probs_from(self, bits):
        snap = self._snapshot()
        probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        for bit in bits:
            probs, _ = self.tick(bit, learn=False)
        self._restore(snap)
        return probs

    # -- entrainment report --------------------------------------------

    def entrainment(self):
        """Return per-oscillator entrainment strength for each vocab item.
        For dual mode, returns the short-term R (current confidence)."""
        if self.memory_mode in ('ema', 'dual'):
            if self.memory_mode == 'dual':
                return np.sqrt(self.short_cos**2 + self.short_sin**2)
            return np.sqrt(self.mem_cos**2 + self.mem_sin**2)
        else:
            return np.zeros((self.n_osc, VOCAB_SIZE))

    # -- state management ----------------------------------------------

    def _snapshot(self):
        return {
            'phi': self.phi.copy(),
            'kick': self.kick.copy(),
            'mem_cos': self.mem_cos.copy(),
            'mem_sin': self.mem_sin.copy(),
            'short_cos': self.short_cos.copy(),
            'short_sin': self.short_sin.copy(),
            'buf_cos': self.buf_cos.copy(),
            'buf_sin': self.buf_sin.copy(),
            'buf_bit': self.buf_bit.copy(),
            'buf_ptr': self.buf_ptr,
            'buf_count': self.buf_count,
            'last_probs': self._last_probs.copy(),
            'has_prediction': self._has_prediction,
        }

    def _restore(self, snap):
        self.phi = snap['phi']
        self.kick = snap['kick']
        self.mem_cos = snap['mem_cos']
        self.mem_sin = snap['mem_sin']
        self.short_cos = snap['short_cos']
        self.short_sin = snap['short_sin']
        self.buf_cos = snap['buf_cos']
        self.buf_sin = snap['buf_sin']
        self.buf_bit = snap['buf_bit']
        self.buf_ptr = snap['buf_ptr']
        self.buf_count = snap['buf_count']
        self._last_probs = snap['last_probs']
        self._has_prediction = snap['has_prediction']

    def reset_phases(self, rng=None):
        """Reset oscillator phases and short-term coherence.
        Long-term memory + kicks preserved."""
        rng = rng if rng is not None else self.rng
        self.phi = rng.uniform(0, 2 * np.pi, self.n_osc)
        self.short_cos[:] = 0
        self.short_sin[:] = 0
        self._last_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        self._has_prediction = False

    def reset_all(self, rng=None):
        """Full reset: phases + all memory + kicks."""
        self.reset_phases(rng)
        self.mem_cos[:] = 0
        self.mem_sin[:] = 0
        self.kick[:, 0] = -0.3
        self.kick[:, 1] = +0.3
        self.buf_cos[:] = 0
        self.buf_sin[:] = 0
        self.buf_bit[:] = -1
        self.buf_ptr = 0
        self.buf_count = 0

    # -- persistence ---------------------------------------------------

    def state_dict(self):
        return {
            '_meta': {
                'n_osc': self.n_osc,
                'memory_mode': self.memory_mode,
                'mem_alpha': self.mem_alpha,
                'mem_alpha_short': self.mem_alpha_short,
                'buf_size': self.buf_size,
            },
            'omega': self.omega, 'K': self.K,
            'kick': self.kick,
            'mem_cos': self.mem_cos, 'mem_sin': self.mem_sin,
            'short_cos': self.short_cos, 'short_sin': self.short_sin,
            'buf_cos': self.buf_cos, 'buf_sin': self.buf_sin,
            'buf_bit': self.buf_bit,
            'phi': self.phi,
        }

    def load_state(self, sd):
        if sd['_meta']['n_osc'] != self.n_osc:
            raise ValueError(f"n_osc mismatch")
        for k in ['omega', 'K', 'kick', 'mem_cos', 'mem_sin',
                   'short_cos', 'short_sin',
                   'buf_cos', 'buf_sin', 'buf_bit', 'phi']:
            if k in sd:
                setattr(self, k, sd[k])
        self.buf_ptr = 0
        self.buf_count = int((self.buf_bit >= 0).sum())
        self._last_probs = self.predict_probs()
        self._has_prediction = False


# ----------------------------------------------------------------------
#  Battery
# ----------------------------------------------------------------------

class Battery:
    """Scalar drive. Modulates kick magnitude — a lonely pipit kicks
    harder (more reactive, more erratic oscillations). This IS the
    creature's emotional state."""

    def __init__(self, level=0.8):
        self.level = float(level)
        self.decay_rate = 0.0002
        self.low_threshold = 0.3

    def decay(self):
        self.level = max(0.0, self.level - self.decay_rate)

    def replenish(self, amount):
        self.level = min(1.0, self.level + max(0.0, amount))

    def is_lonely(self):
        return self.level < self.low_threshold

    def kick_scale(self):
        """Lonely → bigger kicks. Full → normal."""
        if self.level >= 0.5:
            return 1.0
        return 1.0 + (0.5 - self.level) * 1.5

    def state_dict(self):
        return {'level': self.level}

    def load_state(self, sd):
        self.level = float(sd['level'])


# ----------------------------------------------------------------------
#  Pipit
# ----------------------------------------------------------------------

class Pipit:
    """A bit-creature. Oscillators + phase memory + battery.
    No neural network. No backprop."""

    def __init__(self, name='pipit', n_osc=DEFAULT_N_OSC,
                 memory_mode='ema', mem_alpha=0.97, mem_alpha_short=0.5,
                 buf_size=32, seed=None):
        self.name = name
        self.seed = (int(seed) if seed is not None
                     else int(np.random.SeedSequence().entropy))
        self.rng = np.random.default_rng(self.seed)
        brng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))
        self.brain = OscillatorBrain(n_osc=n_osc, memory_mode=memory_mode,
                                     mem_alpha=mem_alpha,
                                     mem_alpha_short=mem_alpha_short,
                                     buf_size=buf_size, rng=brng)
        self.battery = Battery()
        self.round = 0
        self.temperature = 0.7
        self.recent_losses = []

    def tick(self, input_bit, learn=True, emit=True):
        """One tick of creature life.
        Returns (output_bit, prediction_probs, loss)."""
        kick_scale = self.battery.kick_scale()
        probs, loss = self.brain.tick(input_bit, learn=learn,
                                     kick_scale=kick_scale)
        output_bit = None
        if emit:
            if self.temperature < 1e-6:
                output_bit = int(np.argmax(probs))
            else:
                logits = np.log(probs + 1e-12) / self.temperature
                logits = logits - logits.max()
                e = np.exp(logits)
                sp = e / e.sum()
                output_bit = int(self.rng.choice(VOCAB_SIZE, p=sp))
        self.battery.decay()
        self.recent_losses.append(loss)
        if len(self.recent_losses) > 500:
            self.recent_losses = self.recent_losses[-500:]
        self.round += 1
        return output_bit, probs, loss

    def experience(self, bits, learn=True):
        """Process a sequence from the world."""
        results = []
        for bit in bits:
            out, _, loss = self.tick(int(bit), learn=learn, emit=True)
            results.append((out, loss))
        return results

    def babble(self, prompt, n=32, learn_during=False):
        """Process prompt, then generate by hearing own output."""
        for bit in prompt:
            self.tick(int(bit), learn=learn_during, emit=False)
        out = []
        last = int(prompt[-1]) if prompt else 0
        for _ in range(n):
            output, _, _ = self.tick(last, learn=learn_during, emit=True)
            out.append(output)
            last = output
        return out

    def babble_snapshot(self, prompt, n=32):
        """Babble without modifying the creature's persistent state."""
        snap = self.brain._snapshot()
        r_bak = self.round
        rng_state = self.rng.bit_generator.state
        result = self.babble(prompt, n=n, learn_during=False)
        self.brain._restore(snap)
        self.round = r_bak
        self.rng.bit_generator.state = rng_state
        return result

    # -- persistence ---------------------------------------------------

    def save(self, path):
        brain_sd = self.brain.state_dict()
        batt_sd = self.battery.state_dict()
        arrays = {}
        for k, v in brain_sd.items():
            if isinstance(v, np.ndarray):
                arrays[f'brain__{k}'] = v
        np.savez(path + '.npz', **arrays)
        meta = {
            'name': self.name, 'seed': self.seed, 'round': self.round,
            'temperature': self.temperature,
            'brain_meta': brain_sd['_meta'],
            'battery': batt_sd,
        }
        with open(path + '.json', 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path + '.json') as f:
            meta = json.load(f)
        arrays = dict(np.load(path + '.npz'))
        bm = meta['brain_meta']
        pipit = cls(
            name=meta['name'], n_osc=int(bm['n_osc']),
            memory_mode=bm.get('memory_mode', 'ema'),
            mem_alpha=float(bm.get('mem_alpha', 0.97)),
            mem_alpha_short=float(bm.get('mem_alpha_short', 0.5)),
            buf_size=int(bm.get('buf_size', 32)),
            seed=int(meta['seed']),
        )
        pipit.round = int(meta['round'])
        pipit.temperature = float(meta.get('temperature', 0.7))
        brain_sd = {'_meta': bm}
        for k, v in arrays.items():
            if k.startswith('brain__'):
                brain_sd[k[len('brain__'):]] = v
        pipit.brain.load_state(brain_sd)
        pipit.battery.load_state(meta['battery'])
        return pipit

    def __repr__(self):
        mode = self.brain.memory_mode
        if mode == 'ema':
            extra = f'alpha={self.brain.mem_alpha}'
        elif mode == 'dual':
            extra = (f'alpha={self.brain.mem_alpha},'
                     f'short={self.brain.mem_alpha_short}')
        else:
            extra = f'buf={self.brain.buf_size}'
        return (f"Pipit('{self.name}', n_osc={self.brain.n_osc}, "
                f"{mode}({extra}), round={self.round})")


__all__ = ['OscillatorBrain', 'Battery', 'Pipit',
           'VOCAB_SIZE', 'DEFAULT_N_OSC']
