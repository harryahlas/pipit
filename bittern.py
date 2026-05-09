"""
bittern.py — a binary creature.

A bittern lives in a stream of bits. It hears bits, and it emits bits.
Vocabulary is 2 (0 and 1). Everything that mattered in glaud-i — the
brain, the firing organs, the battery, the mood — is preserved here,
but the substrate is the smallest possible alphabet.

This file owns four classes glued together by a fifth:

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

  Pain     — the bodily layer between brain and organs. Five scalars
             (sting, hunger, nausea, itch, disquiet) that respond to
             different errors and produce different behavioral
             modulations. The brain doesn't know about pain; the
             organs don't know about pain; pain modifies HOW the
             creature uses them.

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

    v3 changes (for the pain system):
    - `fire()` accepts optional `temperature_override` and
      `lateral_weight_override` parameters. Pain consults these
      to modulate organ behavior without mutating organ state.
      A non-overridden call is identical to v2 behavior.
    - `learn_from_brain()` accepts an optional
      `sensitivity_lr_override`. Same idea — itch raises this
      when the body needs to re-align with the brain.
    - `_activations()` factored out of `fire()` so pain can
      compute "what would the organs do under condition X?"
      without consuming randomness.
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

    def _activations(self, brain_state, prev_bit, fatigue, emotion_bias,
                     temperature_override=None,
                     lateral_weight_override=None):
        """Compute fire probabilities without sampling.

        Used by fire() and by pain measurements that need to compare
        hypothetical activations (e.g., with vs. without lateral)
        without consuming randomness from the rng."""
        bs_n = np.linalg.norm(brain_state)
        bs_dir = brain_state / bs_n if bs_n > 1e-9 else brain_state
        sens = self.sensitivity @ bs_dir                       # (2,)

        act = self.sensitivity_weight * sens
        act = act + self.resting
        act = act - fatigue
        act = act + emotion_bias

        lw = (self.lateral_weight if lateral_weight_override is None
              else lateral_weight_override)
        if prev_bit is not None and lw > 0:
            lateral_signal = self.lateral[prev_bit].copy()
            n = np.linalg.norm(lateral_signal)
            if n > 1e-9:
                lateral_signal = lateral_signal / n
            act = act + lw * lateral_signal

        temp = (self.temperature if temperature_override is None
                else temperature_override)
        logits = act / max(temp, 1e-3)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return probs

    def fire(self, brain_state, prev_bit, fatigue, emotion_bias, rng,
             temperature_override=None, lateral_weight_override=None):
        """Pick a bit by competition between the two organs."""
        probs = self._activations(brain_state, prev_bit, fatigue,
                                  emotion_bias,
                                  temperature_override,
                                  lateral_weight_override)
        choice = int(rng.choice(VOCAB_SIZE, p=probs))
        return choice, probs

    def observe(self, bits):
        """Hebbian update from a heard bit sequence."""
        for i in range(len(bits) - 1):
            a, b = int(bits[i]), int(bits[i + 1])
            self.lateral[a, b] += self.lateral_lr
        self.lateral *= self.lateral_decay

    def learn_from_brain(self, brain_state, target_bit,
                         sensitivity_lr_override=None):
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
        lr = (self.sensitivity_lr if sensitivity_lr_override is None
              else sensitivity_lr_override)
        self.sensitivity[target_bit] += lr * bs_dir
        self.sensitivity[other] -= lr * bs_dir
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
#  Pain — five signals, five behavioral knobs
# ----------------------------------------------------------------------

class Pain:
    """Five pain signals — each a different error, each shaping behavior
    in a different way.

    Pain is the bodily layer between brain and organs. The brain
    doesn't know about pain; the organs don't know about pain. Pain
    modifies HOW the creature uses them.

    Five pains:
        sting    — confident wrong predictions raise organ temperature
        hunger   — sustained mediocrity shortens effective context
        nausea   — short/long brain-state EMA divergence suppresses lateral
        itch     — body habits diverging from raw response speeds sensitivity
        disquiet — immediate novelty raises brain learning rate

    Hunger is derived from recent_brain_losses; the other four are
    EMA-updated scalars stored on this object.

    Sting and itch update once per training pair. Nausea and disquiet
    update once per listen() call.

    --- Itch reformulation ---
    The original design compared brain.predict_probs against organ
    probs. But in this architecture organs faithfully transmit the
    brain (~100% argmax agreement), so itch was approximately a
    lagged echo of sting — not a separate dimension. The reformulation
    compares the organs' response WITH lateral (bigram habits) against
    WITHOUT lateral (raw sensitivity-driven response). Itch fires when
    habits and raw signal disagree — typically right after a regime
    shift, before the lateral matrix has decayed away. This is a
    body-level signal independent of brain confidence.

    --- Hunger warmup ---
    The 0.35 threshold is below ln(2)≈0.693, so a learning creature
    quickly drops below it and stops being hungry. But a brand-new
    creature has loss ≈ ln(2) and would be MAXIMALLY hungry — locked
    into a 4-bit context window before it has had a chance to learn
    anything that requires longer context (like swelling). Hunger is
    suppressed for the first 500 rounds.

    --- Nausea via short/long EMAs ---
    The original design used cosine distance between current brain
    state and a single EMA. But brain_state is a function of context,
    and a swelling sequence contains burst-phase contexts and pause-
    phase contexts that produce very different states — so the
    point-vs-EMA distance is high CONTINUOUSLY, not just at regime
    transitions. The reformulation tracks two EMAs (short window ~10
    rounds, long window ~100 rounds). In a stable regime they
    converge. After a regime shift the short tracks the new state
    while the long lags, producing a transient gap that decays as
    the long catches up.
    """

    def __init__(self, brain_dim):
        self.brain_dim = brain_dim
        # Master toggle for ablation. When False, all effective_*
        # methods short-circuit to base values so pain has no
        # behavioral influence. Pain still updates internally
        # (we can read what would have been felt) but doesn't flow
        # into the brain or the organs. Used to answer "is pain
        # contributing constructively, or just along for the ride?"
        self.enabled = True
        # Stored scalars
        self.sting_level = 0.0
        self.nausea_level = 0.0
        self.itch_level = 0.0
        self.disquiet_level = 0.0
        # hunger is derived

        # Brain-state EMAs for nausea (short window vs long window)
        self.bs_ema_short = np.zeros(brain_dim)
        self.bs_ema_long = np.zeros(brain_dim)
        self._ema_initialized = False

        # Update rates.
        #
        # alpha_sting was 0.10 (~7-pair half-life). But in a block-
        # structured habitat each listen call is one class, so the
        # sting signal alternates between ~0.8 during a confidently-
        # wrong block and ~0 between them. With a 7-pair half-life
        # the EMA tracks the within-block signal but decays close
        # to zero before the next checkpoint sample. Result: a
        # chronically-wrong creature looked un-stung when we
        # measured it. 0.02 (~35-pair half-life) integrates across
        # multiple blocks so sting is visible at any sampling
        # moment, not just during the streak.
        self.alpha_sting = 0.02
        self.alpha_itch = 0.10
        self.alpha_nausea = 0.10
        self.alpha_disquiet = 0.15      # faster than other pains
        self.alpha_bs_short = 0.10      # ~10-round window
        self.alpha_bs_long = 0.01       # ~100-round window

        # Hunger config
        self.hunger_warmup = 500
        self.hunger_low = 0.35          # below this, no hunger
        self.hunger_high = 0.70         # at this, max hunger (≈ ln 2)

        # Disquiet config
        # Different from hunger: disquiet measures THIS call's losses
        # against an absolute "comfortable" threshold, not a 50-round
        # average. So disquiet fires on the leading edge of a regime
        # change; hunger fires on the trailing edge.
        self.disquiet_baseline = 0.50

    # ── Updates ──────────────────────────────────────────────────────

    def update_after_pair(self, brain_probs_pre, target_bit,
                          organ_probs_with_lat, organ_probs_without_lat):
        """Called after each (prefix, target) training pair.

        brain_probs_pre: probs from brain BEFORE this pair's gradient step
        organ_probs_with_lat: organ activation with full lateral weight
        organ_probs_without_lat: organ activation with lateral_weight=0
        """
        # Sting: confidence × wrongness. Peaks near 1.0 when the brain
        # put nearly all probability on the wrong bit. Near zero if the
        # brain hedged (max ≈ 0.5) or was correct.
        confidence = float(np.max(brain_probs_pre))
        wrongness = float(1.0 - brain_probs_pre[target_bit])
        sting_signal = confidence * wrongness
        a = self.alpha_sting
        self.sting_level = (1 - a) * self.sting_level + a * sting_signal

        # Itch: how much the lateral matrix (bigram habits) is shifting
        # the organ output relative to the lateral-free baseline.
        # When habits and raw response are aligned, itch is low.
        itch_signal = float(abs(organ_probs_with_lat[0]
                                - organ_probs_without_lat[0]))
        a = self.alpha_itch
        self.itch_level = (1 - a) * self.itch_level + a * itch_signal

    def update_after_listen(self, current_brain_state, this_call_losses):
        """Called once per listen() call, after the training loop."""
        bs = np.asarray(current_brain_state, dtype=np.float64)

        # Update brain-state EMAs (short and long windows)
        if not self._ema_initialized:
            self.bs_ema_short = bs.copy()
            self.bs_ema_long = bs.copy()
            self._ema_initialized = True
        else:
            a_s = self.alpha_bs_short
            a_l = self.alpha_bs_long
            self.bs_ema_short = (1 - a_s) * self.bs_ema_short + a_s * bs
            self.bs_ema_long = (1 - a_l) * self.bs_ema_long + a_l * bs

        # Nausea: cosine distance between short and long EMAs
        s = self.bs_ema_short
        l = self.bs_ema_long
        ns = float(np.linalg.norm(s))
        nl = float(np.linalg.norm(l))
        if ns > 1e-9 and nl > 1e-9:
            cos_sim = float(np.dot(s, l) / (ns * nl))
            # Clip to [0, 1]. cos_sim ranges [-1, 1], so 1 - cos_sim
            # ranges [0, 2]. Without clipping, anti-aligned EMAs
            # produce nausea > 1 → effective_lateral_weight < 0
            # (lateral inversion). That can be a real mechanism —
            # the first habitat run produced a transient near-perfect
            # steady_2 specialist via this pathway — but it should be
            # a deliberate experiment with clear knobs, not an
            # overflow bug. Held for a future explicit "lateral
            # inversion" pain.
            nausea_signal = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
        else:
            nausea_signal = 0.0
        a = self.alpha_nausea
        self.nausea_level = ((1 - a) * self.nausea_level
                             + a * nausea_signal)

        # Disquiet: this call's losses against an absolute baseline.
        # Normalized so 0.50 (baseline) → 0, and ln(2) ≈ 0.69 → ~0.4,
        # values well above ln(2) → 1. Clipped to [0, 1].
        if this_call_losses:
            this_avg = float(np.mean(this_call_losses))
            raw = (this_avg - self.disquiet_baseline) / self.disquiet_baseline
            disquiet_signal = float(np.clip(raw, 0.0, 1.0))
        else:
            disquiet_signal = 0.0
        a = self.alpha_disquiet
        self.disquiet_level = ((1 - a) * self.disquiet_level
                               + a * disquiet_signal)

    # ── Hunger (derived) ─────────────────────────────────────────────

    def hunger_level(self, recent_brain_losses, round_):
        """Derived from recent_brain_losses, with warmup."""
        if round_ < self.hunger_warmup:
            return 0.0
        if not recent_brain_losses:
            return 0.0
        recent_avg = float(np.mean(recent_brain_losses[-50:]))
        span = self.hunger_high - self.hunger_low
        return float(np.clip((recent_avg - self.hunger_low) / span,
                             0.0, 1.0))

    # ── Effective behavioral knobs ───────────────────────────────────
    #
    # Each method takes the underlying base value (organ temperature,
    # brain context, etc) and returns the modulated version that the
    # creature should actually use. When self.enabled is False, every
    # method short-circuits to the base value — used for ablation.

    def effective_temperature(self, base_temp):
        """Sting raises temperature: stung creatures hedge."""
        if not self.enabled:
            return base_temp
        return base_temp * (1.0 + 2.0 * self.sting_level)

    def effective_context(self, base_context, recent_brain_losses, round_):
        """Hunger shortens context."""
        if not self.enabled:
            return base_context
        h = self.hunger_level(recent_brain_losses, round_)
        eff = int(base_context * (1.0 - 0.75 * h))
        return max(4, eff)

    def effective_lateral_weight(self, base_weight):
        """Nausea suppresses lateral: stop trusting habits during regime change."""
        if not self.enabled:
            return base_weight
        return base_weight * (1.0 - self.nausea_level)

    def effective_sensitivity_lr(self, base_lr):
        """Itch accelerates body alignment."""
        if not self.enabled:
            return base_lr
        return base_lr * (1.0 + 4.0 * self.itch_level)

    def effective_brain_lr(self, base_lr):
        """Disquiet adds urgency. Capped at 2x to prevent runaway
        when combined with the existing loneliness 2x (which is
        applied externally by habitat/evolve)."""
        if not self.enabled:
            return base_lr
        return base_lr * min(1.0 + self.disquiet_level, 2.0)

    # ── Reporting ────────────────────────────────────────────────────

    def report(self, recent_brain_losses=None, round_=0):
        return {
            'sting': float(self.sting_level),
            'hunger': float(self.hunger_level(recent_brain_losses or [],
                                              round_)),
            'nausea': float(self.nausea_level),
            'itch': float(self.itch_level),
            'disquiet': float(self.disquiet_level),
        }

    # ── Persistence ──────────────────────────────────────────────────

    def state_dict(self):
        return {
            '_meta': {'brain_dim': self.brain_dim},
            'sting_level': self.sting_level,
            'nausea_level': self.nausea_level,
            'itch_level': self.itch_level,
            'disquiet_level': self.disquiet_level,
            'bs_ema_short': self.bs_ema_short,
            'bs_ema_long': self.bs_ema_long,
            'ema_initialized': self._ema_initialized,
        }

    def load_state(self, sd):
        if sd['_meta']['brain_dim'] != self.brain_dim:
            raise ValueError(
                f"Pain brain_dim mismatch: this is {self.brain_dim} "
                f"but saved is {sd['_meta']['brain_dim']}")
        self.sting_level = float(sd['sting_level'])
        self.nausea_level = float(sd['nausea_level'])
        self.itch_level = float(sd['itch_level'])
        self.disquiet_level = float(sd['disquiet_level'])
        self.bs_ema_short = np.asarray(sd['bs_ema_short'])
        self.bs_ema_long = np.asarray(sd['bs_ema_long'])
        self._ema_initialized = bool(sd['ema_initialized'])


# ----------------------------------------------------------------------
#  PainfulMemory — durable replay buffer for past hurts
# ----------------------------------------------------------------------

class PainfulMemory:
    """A small library of confidently-wrong moments. Captured during
    listen() when sting exceeds a threshold; sampled during listen()
    to nudge training back toward past hurt.

    This addresses what Pain v1 failed to do: it gives the creature
    a durable, specific, contextual memory of past errors. Pain v1
    was a transient mood — five scalars that decayed between rounds.
    PainfulMemory is episodic — specific (prefix, target) pairs that
    persist across thousands of rounds and can be re-triggered when
    the same context recurs.

    Biology: Aplysia gill-withdrawal sensitization, hippocampal
    replay, taste aversion learning. A single painful event creates
    a durable, specific association between a context (a tail
    touch, a food taste, a bit-prefix) and an outcome.

    Design choices:

    Capture happens BEFORE the gradient step, not after. The pair
    the brain just got confidently wrong on is the useful training
    signal; capturing post-step would record a slightly-stale sting
    because the brain has already begun to correct.

    No decay. A memory's sting is fixed at capture time. If the
    creature has learned to be correct on that prefix, the memory
    won't generate new sting events to refresh — but it also won't
    fade. It stays as a "don't forget this lesson" anchor. The
    capacity-limited eviction (lowest sting displaced first) is the
    only way memories leave the buffer.

    Dedup by (prefix, target). Capturing an identical pair twice
    updates the existing entry's sting (keeping the max) rather
    than filling the buffer with copies of the same hard case.

    Replay sampling is weighted by sting. Most-painful memories
    replay most often. Within a fixed budget of train_pairs per
    listen, the buffer competes with fresh world samples.
    """

    def __init__(self, capacity=32, capture_threshold=0.4,
                 replay_p=0.25):
        self.capacity = int(capacity)
        self.capture_threshold = float(capture_threshold)
        self.replay_p = float(replay_p)
        self.entries = []   # list of {'prefix': [int], 'target': int, 'sting': float}
        # Master toggle for ablation
        self.enabled = True

    # ── Capture ──────────────────────────────────────────────────────

    def maybe_capture(self, prefix, target, brain_probs_pre):
        """If the brain was confidently wrong on this pair, store it.

        Returns True if a capture happened, False otherwise. The
        return value is for diagnostics; callers can ignore it."""
        if not self.enabled:
            return False
        confidence = float(np.max(brain_probs_pre))
        wrongness = float(1.0 - brain_probs_pre[int(target)])
        sting = confidence * wrongness
        if sting < self.capture_threshold:
            return False

        prefix_list = [int(b) for b in prefix]
        target_int = int(target)

        # Dedup: if (prefix, target) already in buffer, just update sting
        for e in self.entries:
            if e['target'] == target_int and e['prefix'] == prefix_list:
                e['sting'] = max(e['sting'], float(sting))
                return True

        entry = {
            'prefix': prefix_list,
            'target': target_int,
            'sting': float(sting),
        }

        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            return True

        # Buffer full: evict lowest-sting entry IF new entry beats it
        min_idx = min(range(len(self.entries)),
                      key=lambda i: self.entries[i]['sting'])
        if self.entries[min_idx]['sting'] < sting:
            self.entries[min_idx] = entry
            return True
        return False

    # ── Replay ───────────────────────────────────────────────────────

    def maybe_replay(self, rng):
        """Maybe return a (prefix, target) from the buffer for training.

        Returns None if no replay should happen this call (either the
        buffer is empty/disabled, or the random draw didn't fire).
        Caller should fall through to normal world-based selection."""
        if not self.enabled:
            return None
        if not self.entries:
            return None
        if rng.random() > self.replay_p:
            return None
        # Sample weighted by sting (more painful = more likely to replay)
        weights = np.array([e['sting'] for e in self.entries], dtype=np.float64)
        total = float(weights.sum())
        if total < 1e-12:
            return None
        probs = weights / total
        idx = int(rng.choice(len(self.entries), p=probs))
        e = self.entries[idx]
        return list(e['prefix']), int(e['target'])

    # ── Reporting ────────────────────────────────────────────────────

    def report(self):
        """Snapshot summary of buffer state."""
        if not self.entries:
            return {'size': 0, 'mean_sting': 0.0, 'max_sting': 0.0,
                    'min_sting': 0.0}
        stings = [e['sting'] for e in self.entries]
        return {
            'size': len(self.entries),
            'mean_sting': float(np.mean(stings)),
            'max_sting': float(np.max(stings)),
            'min_sting': float(np.min(stings)),
        }

    # ── Persistence ──────────────────────────────────────────────────

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'capture_threshold': self.capture_threshold,
            'replay_p': self.replay_p,
            'enabled': self.enabled,
            'entries': self.entries,
        }

    def load_state(self, sd):
        self.capacity = int(sd['capacity'])
        self.capture_threshold = float(sd['capture_threshold'])
        self.replay_p = float(sd['replay_p'])
        self.enabled = bool(sd['enabled'])
        self.entries = list(sd['entries'])


# ----------------------------------------------------------------------
#  Bittern — the creature
# ----------------------------------------------------------------------

class Bittern:
    """A bit-creature. Brain + Organs + Battery + Pain."""

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
        self.pain = Pain(brain_dim=brain_dim)
        self.painful_memory = PainfulMemory()
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

        --- Pain interactions ---
        Hunger shortens the prefix (effective_context replaces context).
        Disquiet raises brain lr (effective_brain_lr).
        Itch raises sensitivity lr (effective_sensitivity_lr).
        Sting and itch update after each pair; nausea and disquiet
        update once at the end of the call."""
        bits = list(bits)
        if len(bits) < 2:
            return

        self.organs.observe(bits)

        # Pre-compute split pools when balanced training is requested
        if balanced:
            pos_0 = [i for i in range(1, len(bits)) if bits[i] == 0]
            pos_1 = [i for i in range(1, len(bits)) if bits[i] == 1]

        # Hunger: shorten effective context for THIS listen call.
        # Computed once at top of call; pain doesn't update mid-loop.
        eff_context = self.pain.effective_context(
            self.brain.context, self.recent_brain_losses, self.round)

        last_bs = None             # for update_after_listen
        this_call_losses = []      # for disquiet

        for _ in range(max(1, train_pairs)):
            # Replay or fresh? PainfulMemory may return a past pair
            # to retrain on; otherwise we draw from the current world.
            replay = self.painful_memory.maybe_replay(self.rng)
            if replay is not None:
                prefix, target = replay
                # Truncate to current effective context for consistency
                # with world-based pairs (under hunger, eff_context may
                # be smaller than the captured prefix length).
                if len(prefix) > eff_context:
                    prefix = prefix[-eff_context:]
            else:
                if balanced and pos_0 and pos_1:
                    # Alternate targets: half from 0-pool, half from 1-pool
                    pool = pos_1 if self.rng.random() < 0.5 else pos_0
                    split = int(self.rng.choice(pool))
                else:
                    split = int(self.rng.integers(1, len(bits)))
                start = max(0, split - eff_context)
                prefix = list(bits[start:split])
                target = int(bits[split])

                # Self-hearing: replace some prefix bits with brain's own
                # predictions. Scheduled sampling. Only applied to
                # fresh-from-world pairs, not replays.
                if self_hear_p > 0 and len(prefix) > 1:
                    for t in range(1, len(prefix)):
                        if self.rng.random() < self_hear_p:
                            probs = self.brain.predict_probs(prefix[:t])
                            prefix[t] = int(
                                self.rng.choice(VOCAB_SIZE, p=probs))

            # Sting needs PRE-update brain probs (was the brain confident
            # about the wrong answer BEFORE this gradient step?).
            brain_probs_pre = self.brain.predict_probs(prefix)

            # Capture into PainfulMemory BEFORE the gradient step. The
            # pair the brain just got confidently wrong on is the
            # signal we want to store; capturing post-step would
            # record a slightly-stale sting because the brain has
            # already begun to correct on this pair.
            self.painful_memory.maybe_capture(prefix, target,
                                              brain_probs_pre)

            # Disquiet adds urgency to the brain's learning rate.
            eff_brain_lr = self.pain.effective_brain_lr(lr)

            loss = self.brain.train_step(prefix, target, lr=eff_brain_lr)
            this_call_losses.append(loss)
            self.recent_brain_losses.append(loss)

            # Post-update brain state — what flows to the organs.
            bs, _ = self.brain.encode(prefix)
            last_bs = bs

            # Itch: compare body's response WITH lateral vs WITHOUT.
            # Both evaluations skip fatigue (no recent firing during
            # listen) and use the actual emotion bias.
            prev_for_organs = prefix[-1] if len(prefix) > 0 else None
            fatigue_zero = np.zeros(VOCAB_SIZE)
            emotion = self.battery.modulated_bias()
            probs_with = self.organs._activations(
                bs, prev_for_organs, fatigue_zero, emotion)
            probs_without = self.organs._activations(
                bs, prev_for_organs, fatigue_zero, emotion,
                lateral_weight_override=0.0)

            self.pain.update_after_pair(brain_probs_pre, target,
                                        probs_with, probs_without)

            # Itch accelerates organ alignment with the brain.
            eff_sens_lr = self.pain.effective_sensitivity_lr(
                self.organs.sensitivity_lr)
            self.organs.learn_from_brain(
                bs, target, sensitivity_lr_override=eff_sens_lr)

        if len(self.recent_brain_losses) > 200:
            self.recent_brain_losses = self.recent_brain_losses[-200:]

        # Once-per-call pain updates (nausea, disquiet)
        if last_bs is not None:
            self.pain.update_after_listen(last_bs, this_call_losses)

        self.round += 1

    def babble(self, prompt_bits, n=32, mode='organs'):
        """Given a prompt, generate n bits.

        modes:
            'organs'       - normal path, organs decide each bit
            'brain_sample' - bypass organs, sample from brain.predict_probs
            'brain_argmax' - bypass organs, take brain's most likely bit

        --- Pain interactions (organs mode only) ---
        Hunger shortens window. Sting raises temperature. Nausea
        suppresses lateral. These are computed once per babble call;
        pain doesn't update during babble.
        """
        prompt_bits = list(prompt_bits)
        out = []
        prev = prompt_bits[-1] if prompt_bits else None
        fatigue = np.zeros(VOCAB_SIZE)
        emotion = self.battery.modulated_bias()
        context = list(prompt_bits)

        eff_context = self.pain.effective_context(
            self.brain.context, self.recent_brain_losses, self.round)
        eff_temp = self.pain.effective_temperature(self.organs.temperature)
        eff_lat = self.pain.effective_lateral_weight(
            self.organs.lateral_weight)

        for _ in range(n):
            window = context[-eff_context:]
            if mode == 'organs':
                bs, _ = self.brain.encode(window)
                choice, _probs = self.organs.fire(
                    bs, prev, fatigue, emotion, self.rng,
                    temperature_override=eff_temp,
                    lateral_weight_override=eff_lat)
                fatigue *= self.organs.fatigue_decay
                fatigue[choice] += self.organs.fatigue_amount
            elif mode == 'brain_sample':
                # Brain bypass paths use full base context — pain
                # doesn't apply to brain-only generation modes.
                window = context[-self.brain.context:]
                probs = self.brain.predict_probs(window)
                choice = int(self.rng.choice(VOCAB_SIZE, p=probs))
            elif mode == 'brain_argmax':
                window = context[-self.brain.context:]
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

    def reset_emotional_state(self):
        """Reset battery, pain, painful memory, round counter, and
        recent loss buffer.

        Used when a creature crosses a generational boundary (in
        evolve.py): the brain weights persist, but the felt-state of
        the creature starts fresh. Children of a parent get this
        automatically by being constructed via Bittern(...); the
        PARENT also needs it to enter the next generation as a peer
        of its own children rather than a senior with stale moods.

        PainfulMemory specifically resets here because it is NOT
        heritable — children don't inherit their parent's memories,
        so giving surviving parents their old buffer would mean
        selection rewards "did this parent get lucky with their
        buffer" rather than "is this brain a useful inheritance."
        Within-lifetime memory experiments should not call this."""
        self.battery = Battery()
        self.pain = Pain(brain_dim=self.brain.brain_dim)
        self.painful_memory = PainfulMemory()
        self.round = 0
        self.recent_brain_losses = []

    # -- persistence ---------------------------------------------------

    def save(self, path):
        """Save to <path>.npz + <path>.json."""
        bs_dict = self.brain.state_dict()
        os_dict = self.organs.state_dict()
        bt_dict = self.battery.state_dict()
        p_dict = self.pain.state_dict()

        arrays = {}
        for k, v in bs_dict.items():
            if isinstance(v, np.ndarray):
                arrays[f'brain__{k}'] = v
        for k, v in os_dict.items():
            if isinstance(v, np.ndarray):
                arrays[f'organs__{k}'] = v
        for k, v in p_dict.items():
            if isinstance(v, np.ndarray):
                arrays[f'pain__{k}'] = v
        np.savez(path + '.npz', **arrays)

        meta = {
            'name': self.name,
            'seed': self.seed,
            'round': self.round,
            'brain_meta': bs_dict['_meta'],
            'organs_meta': os_dict['_meta'],
            'battery': bt_dict,
            'pain_meta': p_dict['_meta'],
            'pain_scalars': {
                'sting_level': float(p_dict['sting_level']),
                'nausea_level': float(p_dict['nausea_level']),
                'itch_level': float(p_dict['itch_level']),
                'disquiet_level': float(p_dict['disquiet_level']),
                'ema_initialized': bool(p_dict['ema_initialized']),
            },
            'painful_memory': self.painful_memory.state_dict(),
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

        # Pain — backwards compatible with pre-pain saves: skip if missing.
        if 'pain_meta' in meta and 'pain_scalars' in meta:
            p_load = {'_meta': meta['pain_meta'], **meta['pain_scalars']}
            for k, v in arrays.items():
                if k.startswith('pain__'):
                    p_load[k[len('pain__'):]] = v
            bittern.pain.load_state(p_load)
        # else: bittern.pain was already constructed fresh in __init__

        # PainfulMemory — backwards compatible with pre-memory saves.
        if 'painful_memory' in meta:
            bittern.painful_memory.load_state(meta['painful_memory'])
        # else: bittern.painful_memory was already constructed fresh

        return bittern


__all__ = ['Brain', 'Organs', 'Battery', 'Pain', 'PainfulMemory', 'Bittern',
           'VOCAB_SIZE', 'DEFAULT_EMBED_DIM', 'DEFAULT_BRAIN_DIM',
           'DEFAULT_CONTEXT']
