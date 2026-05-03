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

  2. Phase memory (Hebbian learning):
      For each oscillator i and each vocab item v, an exponential
      moving average of cos(φ_i) and sin(φ_i) at the moment v arrives.
      This captures:
        μ_i[v] : the mean phase of osc i when v arrives (circular mean)
        R_i[v] : the concentration (0 = no pattern, 1 = perfect lock)

      Prediction: score each vocab item by von-Mises similarity
      between current phases and the mean phases associated with that
      item. An entrained oscillator at its "preferred phase" gives a
      strong vote. A non-entrained oscillator contributes noise.

  3. Kick learning (Hebbian reinforcement):
      The input kicks are adjusted based on prediction error. If wrong,
      the kicks that would have helped the correct answer become
      slightly more distinctive. No gradient computation — just
      reinforcement of phase-input distinctiveness.

  Battery:
      Decays each tick. Modulates kick magnitude — a lonely pipit
      kicks harder (more reactive, more erratic output). This IS the
      creature's emotional state: loneliness literally changes the
      oscillation dynamics.

Why this isn't a neural network:
  - No layers. No weight matrices multiplying activations.
  - No gradient descent through a computation graph.
  - Prediction is von-Mises similarity in phase space (a physics
    kernel, not a learned function).
  - Learning is Hebbian association + reinforcement of kicks.
  - Memory is circular statistics (running means of angles), not
    a hidden state vector.
  - The computational substrate is coupled oscillation.

Pure numpy. ~300 lines.
"""

from __future__ import annotations
import json
import numpy as np


VOCAB_SIZE = 2
DEFAULT_N_OSC = 16


class OscillatorBrain:
    """N coupled phase oscillators with phase-memory prediction.

    Fixed: frequencies, coupling.
    Learned (Hebbian): phase-memory associations, input kick magnitudes.
    """

    def __init__(self, n_osc=DEFAULT_N_OSC, rng=None):
        self.n_osc = n_osc
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

        # --- Phase memory (Hebbian, EMA of circular stats) ---
        self.mem_cos = np.zeros((n_osc, VOCAB_SIZE))
        self.mem_sin = np.zeros((n_osc, VOCAB_SIZE))
        self.mem_alpha = 0.97   # EMA decay

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

    # -- phase-memory prediction ---------------------------------------

    def _predict_from_memory(self):
        """Score each vocab item by von-Mises similarity.

        R * cos(φ - μ) = cos(φ) * mem_cos + sin(φ) * mem_sin
        (the entrainment R is implicitly encoded in the magnitude of
        mem_cos and mem_sin — an entrained oscillator has large magnitude).
        """
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        scores = np.zeros(VOCAB_SIZE)
        for v in range(VOCAB_SIZE):
            scores[v] = np.sum(cos_phi * self.mem_cos[:, v]
                               + sin_phi * self.mem_sin[:, v])
        return self._softmax(scores, self.temperature)

    # -- core: one tick ------------------------------------------------

    def tick(self, input_bit, learn=True, kick_scale=1.0):
        """Process one input bit. Returns (prediction_probs, loss)."""

        # -- 1. Score last prediction --
        if self._has_prediction:
            loss = -np.log(self._last_probs[input_bit] + 1e-12)
        else:
            loss = -np.log(0.5)
            self._has_prediction = True

        # -- 2. Update phase memory (Hebbian) --
        if learn:
            a = self.mem_alpha
            self.mem_cos[:, input_bit] = (a * self.mem_cos[:, input_bit]
                                          + (1 - a) * np.cos(self.phi))
            self.mem_sin[:, input_bit] = (a * self.mem_sin[:, input_bit]
                                          + (1 - a) * np.sin(self.phi))

        # -- 3. Kick adjustment (Hebbian reinforcement) --
        if learn and self._has_prediction:
            error = 1.0 - self._last_probs[input_bit]
            if error > 0.3:
                R = np.sqrt(self.mem_cos[:, input_bit]**2
                            + self.mem_sin[:, input_bit]**2)
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
        probs = self._predict_from_memory()
        self._last_probs = probs
        return probs, float(loss)

    # -- prediction (no state change) ---------------------------------

    def predict_probs(self):
        return self._predict_from_memory()

    def predict_probs_from(self, bits):
        snap = self._snapshot()
        probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        for bit in bits:
            probs, _ = self.tick(bit, learn=False)
        self._restore(snap)
        return probs

    # -- entrainment report --------------------------------------------

    def entrainment(self):
        """Return per-oscillator entrainment strength for each vocab item."""
        R = np.sqrt(self.mem_cos**2 + self.mem_sin**2)
        return R   # (n_osc, VOCAB_SIZE)

    # -- state management ----------------------------------------------

    def _snapshot(self):
        return {
            'phi': self.phi.copy(),
            'kick': self.kick.copy(),
            'mem_cos': self.mem_cos.copy(),
            'mem_sin': self.mem_sin.copy(),
            'last_probs': self._last_probs.copy(),
            'has_prediction': self._has_prediction,
        }

    def _restore(self, snap):
        self.phi = snap['phi']
        self.kick = snap['kick']
        self.mem_cos = snap['mem_cos']
        self.mem_sin = snap['mem_sin']
        self._last_probs = snap['last_probs']
        self._has_prediction = snap['has_prediction']

    def reset_phases(self, rng=None):
        """Reset oscillator phases only. Memory + kicks preserved."""
        rng = rng if rng is not None else self.rng
        self.phi = rng.uniform(0, 2 * np.pi, self.n_osc)
        self._last_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        self._has_prediction = False

    def reset_all(self, rng=None):
        """Full reset: phases + memory + kicks."""
        self.reset_phases(rng)
        self.mem_cos[:] = 0
        self.mem_sin[:] = 0
        self.kick[:, 0] = -0.3
        self.kick[:, 1] = +0.3

    # -- persistence ---------------------------------------------------

    def state_dict(self):
        return {
            '_meta': {'n_osc': self.n_osc},
            'omega': self.omega, 'K': self.K,
            'kick': self.kick,
            'mem_cos': self.mem_cos, 'mem_sin': self.mem_sin,
            'phi': self.phi,
        }

    def load_state(self, sd):
        if sd['_meta']['n_osc'] != self.n_osc:
            raise ValueError(f"n_osc mismatch")
        for k in ['omega', 'K', 'kick', 'mem_cos', 'mem_sin', 'phi']:
            setattr(self, k, sd[k])
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

    def __init__(self, name='pipit', n_osc=DEFAULT_N_OSC, seed=None):
        self.name = name
        self.seed = (int(seed) if seed is not None
                     else int(np.random.SeedSequence().entropy))
        self.rng = np.random.default_rng(self.seed)
        brng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))
        self.brain = OscillatorBrain(n_osc=n_osc, rng=brng)
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
        pipit = cls(name=meta['name'], n_osc=int(bm['n_osc']),
                    seed=int(meta['seed']))
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
        return (f"Pipit('{self.name}', n_osc={self.brain.n_osc}, "
                f"round={self.round})")


__all__ = ['OscillatorBrain', 'Battery', 'Pipit',
           'VOCAB_SIZE', 'DEFAULT_N_OSC']
