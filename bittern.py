"""
bittern.py — a binary creature.

A bittern lives in a stream of bits. It hears bits, and it emits bits.
Vocabulary is 2 (0 and 1). Everything that mattered in glaud-i — the
brain, the firing organs, the battery, the mood — is preserved here,
but the substrate is the smallest possible alphabet.

This file owns three classes glued together by a fourth:

  Brain    — single-head causal attention, hand-derived backprop, vocab=2.
             encode(bits) -> brain_state. train_step learns next-bit
             prediction. predict_probs returns a 2-vector.

  Organs   — two organs (a 0-organ and a 1-organ) that compete to fire.
             Sensitivity vector dot brain_state, plus resting bias,
             fatigue, momentum, lateral signal, emotion bias. Hebbian
             update of lateral on observation; reward-shaped sensitivity.

  Battery  — slowly decays. Refills on positive interaction. Modulates
             emotion bias to the organs. Lonely bitterns behave
             differently because their internal state literally is
             different — the brain sees a shifted emotion, the organs
             see a shifted activation.

  Bittern  — owns one of each. listen() observes a stream, babble()
             generates one. save/load via numpy .npz and a JSON
             metadata file. Same persistence shape as glaud-i and
             clade so existing tooling is easy to mirror.

Pure numpy. No frameworks. Hand-derived backprop because that is
what is in the spirit of glaud-i.
"""

from __future__ import annotations

import json
import numpy as np


VOCAB_SIZE = 2  # the entire alphabet
DEFAULT_EMBED_DIM = 16
DEFAULT_BRAIN_DIM = 16
DEFAULT_CONTEXT = 32


# ----------------------------------------------------------------------
#  Brain — single-head causal attention with a next-bit head
# ----------------------------------------------------------------------

class Brain:
    """Tiny single-head causal attention, vocab=2.

    forward(bits) -> brain_state (brain_dim,)
    train_step(bits, target_bit, lr) -> cross-entropy loss

    Architecture:
        embedding(2, embed_dim) + positional_encoding
            -> single-head causal attention (Q,K,V projections)
            -> output projection W_o : embed_dim -> brain_dim
            -> readout W_out : brain_dim -> 2 logits
    """

    def __init__(self, embed_dim=DEFAULT_EMBED_DIM,
                 brain_dim=DEFAULT_BRAIN_DIM,
                 context=DEFAULT_CONTEXT,
                 rng=None):
        self.embed_dim = embed_dim
        self.brain_dim = brain_dim
        self.context = context
        self.rng = rng if rng is not None else np.random.default_rng()

        scale = 1.0 / np.sqrt(embed_dim)
        self.embedding = self.rng.normal(0, 0.3, (VOCAB_SIZE, embed_dim))
        self.pos_enc = self._make_pos_enc(context, embed_dim)

        self.W_q = self.rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_k = self.rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_v = self.rng.normal(0, scale, (embed_dim, embed_dim))
        self.W_o = self.rng.normal(0, scale, (embed_dim, brain_dim))

        self.W_out = self.rng.normal(0, scale, (brain_dim, VOCAB_SIZE))
        self.b_out = np.zeros(VOCAB_SIZE)

        # Pre-compute causal masks (avoids re-allocating every encode call)
        self._causal_masks = {}
        for t in range(2, context + 1):
            self._causal_masks[t] = np.triu(
                np.ones((t, t), dtype=bool), k=1)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _make_pos_enc(n, d):
        pe = np.zeros((n, d))
        pos = np.arange(n)[:, None].astype(np.float64)
        div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe[:, 0::2] = np.sin(pos * div[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = np.cos(pos * div[:pe[:, 1::2].shape[1]])
        return pe

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    # -- forward -------------------------------------------------------

    def encode(self, bits):
        """Return (brain_state, cache). Empty bits -> zero brain state."""
        bits = np.asarray(bits, dtype=np.int64)
        # Right-truncate to context if needed (matches clade's text_to_idxs)
        if bits.size > self.context:
            bits = bits[-self.context:]
        T = bits.size
        if T == 0:
            return np.zeros(self.brain_dim), None

        x = self.embedding[bits] + self.pos_enc[:T]            # (T, D)
        Q = x @ self.W_q                                       # (T, D)
        K = x @ self.W_k
        V = x @ self.W_v
        scale = 1.0 / np.sqrt(self.embed_dim)
        scores = Q @ K.T * scale                               # (T, T)
        # Causal mask (pre-computed)
        if T > 1:
            scores = scores.copy()
            scores[self._causal_masks[T]] = -1e9
        attn = self._softmax(scores, axis=-1)                  # (T, T)
        attn_out = attn @ V                                    # (T, D)
        brain_state = attn_out[-1] @ self.W_o                  # (brain_dim,)

        cache = {'bits': bits, 'x': x, 'Q': Q, 'K': K, 'V': V,
                 'attn': attn, 'attn_out': attn_out, 'T': T,
                 'scale': scale}
        return brain_state, cache

    def predict_probs(self, bits):
        bs, _ = self.encode(bits)
        logits = bs @ self.W_out + self.b_out
        return self._softmax(logits, axis=-1)

    def predict_fast(self, bits):
        """Forward pass without cache — for scoring only.
        Same math as predict_probs but skips cache dict construction."""
        bits = np.asarray(bits, dtype=np.int64)
        if bits.size > self.context:
            bits = bits[-self.context:]
        T = bits.size
        if T == 0:
            return np.array([0.5, 0.5])
        x = self.embedding[bits] + self.pos_enc[:T]
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scores = Q @ K.T / np.sqrt(self.embed_dim)
        if T > 1:
            scores = scores.copy()
            scores[self._causal_masks[T]] = -1e9
        attn = self._softmax(scores, axis=-1)
        bs = (attn @ V)[-1] @ self.W_o
        logits = bs @ self.W_out + self.b_out
        return self._softmax(logits, axis=-1)

    # -- training ------------------------------------------------------

    def train_step(self, bits, target_bit, lr=0.01):
        """One forward+backward step. Returns CE loss."""
        bs, cache = self.encode(bits)
        if cache is None:
            return 0.0

        logits = bs @ self.W_out + self.b_out
        probs = self._softmax(logits, axis=-1)
        loss = -np.log(probs[target_bit] + 1e-12)

        # ----- readout -----
        d_logits = probs.copy()
        d_logits[target_bit] -= 1.0                            # (2,)
        d_W_out = np.outer(bs, d_logits)                       # (brain_dim, 2)
        d_b_out = d_logits
        d_bs = self.W_out @ d_logits                           # (brain_dim,)

        # ----- W_o (only last position contributed to bs) -----
        attn_out_last = cache['attn_out'][-1]                  # (D,)
        d_W_o = np.outer(attn_out_last, d_bs)                  # (D, brain_dim)
        d_attn_out_last = self.W_o @ d_bs                      # (D,)

        T = cache['T']
        d_attn_out = np.zeros_like(cache['attn_out'])          # (T, D)
        d_attn_out[-1] = d_attn_out_last

        # attn_out = attn @ V
        d_attn = d_attn_out @ cache['V'].T                     # (T, T)
        d_V = cache['attn'].T @ d_attn_out                     # (T, D)

        # softmax backward, per row: d_scores[i] = attn[i] * (d_attn[i] - sum(d_attn[i]*attn[i]))
        sum_term = np.sum(d_attn * cache['attn'], axis=-1, keepdims=True)  # (T, 1)
        d_scores = cache['attn'] * (d_attn - sum_term)         # (T, T)

        # scores = Q @ K.T * scale
        scale = cache['scale']
        d_Q = (d_scores @ cache['K']) * scale                  # (T, D)
        d_K = (d_scores.T @ cache['Q']) * scale                # (T, D)

        # Q,K,V from x
        d_W_q = cache['x'].T @ d_Q                             # (D, D)
        d_W_k = cache['x'].T @ d_K
        d_W_v = cache['x'].T @ d_V
        d_x = (d_Q @ self.W_q.T
               + d_K @ self.W_k.T
               + d_V @ self.W_v.T)                             # (T, D)

        # x = embedding[bits] + pos_enc -> scatter d_x into embedding
        d_embedding = np.zeros_like(self.embedding)
        np.add.at(d_embedding, cache['bits'], d_x)

        # ----- apply -----
        self.W_out -= lr * d_W_out
        self.b_out -= lr * d_b_out
        self.W_o -= lr * d_W_o
        self.W_q -= lr * d_W_q
        self.W_k -= lr * d_W_k
        self.W_v -= lr * d_W_v
        self.embedding -= lr * d_embedding

        return float(loss)

    # -- persistence ---------------------------------------------------

    def state_dict(self):
        return {
            '_meta': {'embed_dim': self.embed_dim,
                      'brain_dim': self.brain_dim,
                      'context': self.context},
            'embedding': self.embedding,
            'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v,
            'W_o': self.W_o,
            'W_out': self.W_out, 'b_out': self.b_out,
        }

    def load_state(self, sd):
        meta = sd['_meta']
        if (meta['embed_dim'] != self.embed_dim
                or meta['brain_dim'] != self.brain_dim
                or meta['context'] != self.context):
            raise ValueError(
                f"Brain shape mismatch: this brain is "
                f"({self.embed_dim},{self.brain_dim},{self.context}) but "
                f"saved is ({meta['embed_dim']},{meta['brain_dim']},"
                f"{meta['context']})")
        self.embedding = sd['embedding']
        self.W_q, self.W_k, self.W_v = sd['W_q'], sd['W_k'], sd['W_v']
        self.W_o = sd['W_o']
        self.W_out, self.b_out = sd['W_out'], sd['b_out']


# ----------------------------------------------------------------------
#  Organs — two firing organs, glaud-i style
# ----------------------------------------------------------------------

class Organs:
    """Two organs in a 2-vocab system. Each fires according to a
    sensitivity vector dot brain_state, plus lateral, fatigue, emotion.
    Hebbian on observation, supervised update of sensitivity from heard
    bits (not just reward).

    v2 fixes (post-organ-diagnostic):
    - Scalar `momentum` removed: in vocab=2 it adds equally to both
      organs and cancels in the softmax. It was a no-op.
    - `fatigue_amount` reduced from 1.0 to 0.3 and `fatigue_decay`
      from 0.7 to 0.85. Old values had fatigue ≫ sensitivity,
      drowning out the brain.
    - The asymmetric `firing_threshold` floor is gone. It pulled
      negative activations up to 0.05 while letting positives soar,
      which flattened the softmax and erased signal.
    - `sensitivity_weight` raised from 2.0 to 4.0 so sensitivity is
      the dominant term in the activation.
    - New `learn_from_brain()` method: supervised update of sensitivity
      from a (brain_state, target_bit) pair, called during listen().
      Without this, sensitivity got no learning signal at all during
      pure-listen training (reward was never invoked).
    """

    def __init__(self, brain_dim=DEFAULT_BRAIN_DIM, rng=None):
        self.brain_dim = brain_dim
        self.rng = rng if rng is not None else np.random.default_rng()

        self.sensitivity = self.rng.normal(0, 0.1, (VOCAB_SIZE, brain_dim))
        self.resting = np.zeros(VOCAB_SIZE)
        self.lateral = np.zeros((VOCAB_SIZE, VOCAB_SIZE))

        # Hyperparams (post-diagnostic v2 values)
        self.fatigue_amount = 0.3
        self.fatigue_decay = 0.85
        self.temperature = 0.7
        self.sensitivity_weight = 4.0
        self.lateral_weight = 1.0
        self.lateral_lr = 0.02
        self.lateral_decay = 0.9995
        self.sensitivity_lr = 0.02   # supervised LR (was reward-only)

    def fire(self, brain_state, prev_bit, fatigue, emotion_bias, rng):
        """Pick a bit by competition between the two organs.

        Removed `momentum` parameter — it was a scalar added equally
        to both organs in vocab=2, which is a no-op in the softmax."""
        bs_n = np.linalg.norm(brain_state)
        bs_dir = brain_state / bs_n if bs_n > 1e-9 else brain_state
        sens = self.sensitivity @ bs_dir                       # (2,)

        act = self.sensitivity_weight * sens
        act = act + self.resting
        act = act - fatigue
        act = act + emotion_bias
        if prev_bit is not None:
            lateral_signal = self.lateral[prev_bit].copy()
            n = np.linalg.norm(lateral_signal)
            if n > 1e-9:
                lateral_signal = lateral_signal / n
            act = act + self.lateral_weight * lateral_signal

        logits = act / max(self.temperature, 1e-3)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        choice = int(rng.choice(VOCAB_SIZE, p=probs))
        return choice, probs

    def observe(self, bits):
        """Hebbian update from a heard bit sequence."""
        for i in range(len(bits) - 1):
            a, b = int(bits[i]), int(bits[i + 1])
            self.lateral[a, b] += self.lateral_lr
        self.lateral *= self.lateral_decay

    def learn_from_brain(self, brain_state, target_bit):
        """Supervised update of sensitivity vectors. Pulls the
        target's sensitivity vector toward brain_state, pushes the
        non-target's away. This is what teaches sensitivity to align
        with what the brain is encoding — without it, sensitivity is
        pure noise during a listen-only training regime.

        Equivalent to logistic-regression-on-brain-state for the two
        organs, with brain_state as the feature and target_bit as the
        label, but with hand-applied gradient via dot-product
        normalization rather than full softmax CE."""
        n = np.linalg.norm(brain_state)
        if n < 1e-9:
            return
        bs_dir = brain_state / n
        # Pull target sensitivity TOWARD brain_state direction;
        # push the other AWAY. Symmetric two-class logistic update.
        other = 1 - target_bit
        self.sensitivity[target_bit] += self.sensitivity_lr * bs_dir
        self.sensitivity[other] -= self.sensitivity_lr * bs_dir
        # Clamp row norms so sensitivity can't run away
        row_norms = np.linalg.norm(self.sensitivity, axis=1, keepdims=True)
        scale = np.minimum(1.0, 2.0 / np.maximum(row_norms, 1e-9))
        self.sensitivity *= scale

    def reward(self, target_bit, got_bit, brain_state):
        """Adjust sensitivity vectors based on whether got_bit matched
        target_bit. Kept for completeness; not used in pure-listen
        training. Returns the match scalar."""
        match = 1.0 if target_bit == got_bit else -0.2
        n = np.linalg.norm(brain_state)
        if n < 1e-9:
            return match
        bs_dir = brain_state / n
        delta = self.sensitivity_lr * match * bs_dir
        self.sensitivity[target_bit] += delta
        if got_bit != target_bit:
            self.sensitivity[got_bit] -= 0.3 * delta
        row_norms = np.linalg.norm(self.sensitivity, axis=1, keepdims=True)
        scale = np.minimum(1.0, 2.0 / np.maximum(row_norms, 1e-9))
        self.sensitivity *= scale
        return match

    def state_dict(self):
        return {
            '_meta': {'brain_dim': self.brain_dim},
            'sensitivity': self.sensitivity,
            'resting': self.resting,
            'lateral': self.lateral,
        }

    def load_state(self, sd):
        if sd['_meta']['brain_dim'] != self.brain_dim:
            raise ValueError(
                f"Organs brain_dim mismatch: this is {self.brain_dim} "
                f"but saved is {sd['_meta']['brain_dim']}")
        self.sensitivity = sd['sensitivity']
        self.resting = sd['resting']
        self.lateral = sd['lateral']


# ----------------------------------------------------------------------
#  Battery — internal drive
# ----------------------------------------------------------------------

class Battery:
    """One scalar drive. Decays each round. Refills from interaction.
    Modulates emotion bias to organs."""

    def __init__(self, level=0.8):
        self.level = float(level)
        self.decay_rate = 0.005
        self.low_threshold = 0.3
        self.emotion_weight = 0.05

    def decay(self):
        self.level = max(0.0, self.level - self.decay_rate)

    def replenish(self, amount):
        self.level = min(1.0, self.level + max(0.0, amount))

    def is_lonely(self):
        return self.level < self.low_threshold

    def emotion_bias(self):
        # Steeper on the lonely side, just like glaud-i's social_battery
        if self.level >= 0.5:
            return (self.level - 0.5) * 0.4
        return (self.level - 0.5) * 2.0

    def modulated_bias(self):
        """The actual emotion bias passed to organs (already weighted)."""
        return self.emotion_bias() * self.emotion_weight

    def state_dict(self):
        return {'level': self.level}

    def load_state(self, sd):
        self.level = float(sd['level'])


# ----------------------------------------------------------------------
#  Bittern — the creature
# ----------------------------------------------------------------------

class Bittern:
    """A bit-creature. Brain + Organs + Battery."""

    def __init__(self, name='bittern',
                 embed_dim=DEFAULT_EMBED_DIM,
                 brain_dim=DEFAULT_BRAIN_DIM,
                 context=DEFAULT_CONTEXT,
                 seed=None):
        self.name = name
        self.seed = (int(seed) if seed is not None
                     else int(np.random.SeedSequence().entropy))
        self.rng = np.random.default_rng(self.seed)

        # Sub-RNGs deterministic from seed
        brng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))
        orng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))

        self.brain = Brain(embed_dim=embed_dim, brain_dim=brain_dim,
                           context=context, rng=brng)
        self.organs = Organs(brain_dim=brain_dim, rng=orng)
        self.battery = Battery()
        self.round = 0
        self.recent_brain_losses = []

    # -- core ops ------------------------------------------------------

    def listen(self, bits, train_pairs=4, lr=0.01, balanced=False,
               self_hear_p=0.0):
        """Observe a bit stream. Brain trains via next-bit prediction
        on `train_pairs` random splits per call; organs do Hebbian on
        lateral once AND get a supervised sensitivity update at each
        split (so sensitivity learns to align with brain_state).

        Without organ supervised learning, sensitivity vectors stay
        near random initialization through pure-listen training, and
        the organs ignore (or actively oppose) the brain. Adding it
        is what makes the organs actually transmit brain signal.

        balanced=True: sample split points so target=0 and target=1
        are equally represented, preventing the brain from learning
        the class marginal instead of the rhythm structure.

        self_hear_p>0: scheduled sampling. With probability self_hear_p,
        each prefix bit is replaced by the brain's own prediction.
        Pipit's lesson: creatures that hear their own voice don't
        develop exposure bias (fr_ce rising above tf_ce)."""
        bits = list(bits)
        if len(bits) < 2:
            return

        self.organs.observe(bits)

        # Pre-compute split pools when balanced training is requested
        if balanced:
            pos_0 = [i for i in range(1, len(bits)) if bits[i] == 0]
            pos_1 = [i for i in range(1, len(bits)) if bits[i] == 1]

        for _ in range(max(1, train_pairs)):
            if balanced and pos_0 and pos_1:
                # Alternate targets: half from 0-pool, half from 1-pool
                pool = pos_1 if self.rng.random() < 0.5 else pos_0
                split = int(self.rng.choice(pool))
            else:
                split = int(self.rng.integers(1, len(bits)))
            start = max(0, split - self.brain.context)
            prefix = list(bits[start:split])
            target = int(bits[split])

            # Self-hearing: replace some prefix bits with brain's own
            # predictions. This is scheduled sampling — the creature
            # learns what its own voice sounds like during training.
            if self_hear_p > 0 and len(prefix) > 1:
                for t in range(1, len(prefix)):
                    if self.rng.random() < self_hear_p:
                        probs = self.brain.predict_probs(prefix[:t])
                        prefix[t] = int(
                            self.rng.choice(VOCAB_SIZE, p=probs))

            # Brain step
            loss = self.brain.train_step(prefix, target, lr=lr)
            self.recent_brain_losses.append(loss)
            # Organ supervised step on the SAME prefix/target.
            # Use the brain state AFTER the brain update so organs
            # learn to read the current brain, not a stale one.
            bs, _ = self.brain.encode(prefix)
            self.organs.learn_from_brain(bs, target)
        if len(self.recent_brain_losses) > 200:
            self.recent_brain_losses = self.recent_brain_losses[-200:]
        self.round += 1

    def babble(self, prompt_bits, n=32, mode='organs'):
        """Given a prompt, generate n bits.

        modes:
            'organs'       - normal path, organs decide each bit
            'brain_sample' - bypass organs, sample from brain.predict_probs
            'brain_argmax' - bypass organs, take brain's most likely bit
        """
        prompt_bits = list(prompt_bits)
        out = []
        prev = prompt_bits[-1] if prompt_bits else None
        fatigue = np.zeros(VOCAB_SIZE)
        emotion = self.battery.modulated_bias()
        context = list(prompt_bits)
        for _ in range(n):
            window = context[-self.brain.context:]
            if mode == 'organs':
                bs, _ = self.brain.encode(window)
                choice, _probs = self.organs.fire(
                    bs, prev, fatigue, emotion, self.rng)
                fatigue *= self.organs.fatigue_decay
                fatigue[choice] += self.organs.fatigue_amount
            elif mode == 'brain_sample':
                probs = self.brain.predict_probs(window)
                choice = int(self.rng.choice(VOCAB_SIZE, p=probs))
            elif mode == 'brain_argmax':
                probs = self.brain.predict_probs(window)
                choice = int(np.argmax(probs))
            else:
                raise ValueError(f"unknown babble mode: {mode!r}")
            out.append(choice)
            context.append(choice)
            prev = choice
        return out

    def step(self):
        """Per-round bookkeeping: battery decay."""
        self.battery.decay()

    # -- persistence ---------------------------------------------------

    def save(self, path):
        """Save to <path>.npz + <path>.json."""
        bs_dict = self.brain.state_dict()
        os_dict = self.organs.state_dict()
        bt_dict = self.battery.state_dict()

        arrays = {}
        for k, v in bs_dict.items():
            if isinstance(v, np.ndarray):
                arrays[f'brain__{k}'] = v
        for k, v in os_dict.items():
            if isinstance(v, np.ndarray):
                arrays[f'organs__{k}'] = v
        np.savez(path + '.npz', **arrays)

        meta = {
            'name': self.name,
            'seed': self.seed,
            'round': self.round,
            'brain_meta': bs_dict['_meta'],
            'organs_meta': os_dict['_meta'],
            'battery': bt_dict,
        }
        with open(path + '.json', 'w') as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path + '.json') as f:
            meta = json.load(f)
        arrays = dict(np.load(path + '.npz'))

        bm = meta['brain_meta']
        bittern = cls(
            name=meta['name'],
            embed_dim=int(bm['embed_dim']),
            brain_dim=int(bm['brain_dim']),
            context=int(bm['context']),
            seed=int(meta['seed']),
        )
        bittern.round = int(meta['round'])

        bs_load = {'_meta': bm}
        for k, v in arrays.items():
            if k.startswith('brain__'):
                bs_load[k[len('brain__'):]] = v
        bittern.brain.load_state(bs_load)

        os_load = {'_meta': meta['organs_meta']}
        for k, v in arrays.items():
            if k.startswith('organs__'):
                os_load[k[len('organs__'):]] = v
        bittern.organs.load_state(os_load)

        bittern.battery.load_state(meta['battery'])
        return bittern


__all__ = ['Brain', 'Organs', 'Battery', 'Bittern',
           'VOCAB_SIZE', 'DEFAULT_EMBED_DIM', 'DEFAULT_BRAIN_DIM',
           'DEFAULT_CONTEXT']
