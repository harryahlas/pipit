"""
world.py — the bit-rhythms a bittern lives in.

Each rhythm class is a generator (n, rng) -> list[int]. The corpus is
just labelled samples from these generators. Emotion labels are
borrowed from glaud-i for downstream experiments where the battery's
emotion bias is tied to which class is being heard.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
#  Rhythm generators
# ----------------------------------------------------------------------

def steady_2(n=64, rng=None):
    """0101010101..."""
    return [(i % 2) for i in range(n)]


def steady_3(n=64, rng=None):
    """001001001..."""
    return [int((i % 3) == 2) for i in range(n)]


def steady_4(n=64, rng=None):
    """00110011..."""
    return [int((i % 4) >= 2) for i in range(n)]


def swelling(n=64, rng=None):
    """Long pauses then long bursts. Variable lengths each cycle so the
    bittern has to learn the SHAPE (long-on / long-off) not a fixed
    period."""
    rng = rng if rng is not None else np.random.default_rng()
    out = []
    while len(out) < n:
        pause = int(rng.integers(4, 12))
        burst = int(rng.integers(4, 12))
        out += [0] * pause + [1] * burst
    return out[:n]


def staccato(n=64, rng=None):
    """Short bursts, long gaps. Opposite shape from swelling."""
    rng = rng if rng is not None else np.random.default_rng()
    out = []
    while len(out) < n:
        burst = int(rng.integers(1, 4))
        gap = int(rng.integers(4, 10))
        out += [1] * burst + [0] * gap
    return out[:n]


def random_bits(n=64, rng=None):
    """Pure noise — control case. A learned brain should hit fr_ce ≈ ln(2)
    on this and not lower."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.integers(0, 2, size=n).tolist()


# ----------------------------------------------------------------------
#  Class registry
# ----------------------------------------------------------------------

CLASSES = {
    # name -> (generator, emotion_label)
    'steady_2':  (steady_2,  'neutral'),
    'steady_3':  (steady_3,  'neutral'),
    'steady_4':  (steady_4,  'neutral'),
    'swelling':  (swelling,  'good'),
    'staccato':  (staccato,  'bad'),
    'random':    (random_bits, 'neutral'),
}


def make_corpus(per_class=200, length=48, classes=None, rng=None):
    """Generate a labelled corpus.

    Default `classes=None` selects the THREE most distinctive rhythms:
    steady_2 (alternating), swelling (long runs), and staccato (sharp
    bursts). These have very different bigram statistics and are
    learnable as a mix. Pass a custom list to include the other
    classes (steady_3, steady_4, random) — but `random` is
    fundamentally unlearnable (its conditional entropy is ln(2) by
    construction), so including it pollutes training gradients
    rather than helping.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if classes is None:
        classes = ['steady_2', 'swelling', 'staccato']
    items = []
    for cls in classes:
        gen, emo = CLASSES[cls]
        for _ in range(per_class):
            items.append({
                'bits': gen(n=length, rng=rng),
                'class': cls,
                'emotion': emo,
            })
    rng.shuffle(items)
    return items


def bits_to_str(bits):
    return ''.join(str(int(b)) for b in bits)


def str_to_bits(s):
    return [int(c) for c in s if c in '01']


# ----------------------------------------------------------------------
#  Self-test
# ----------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print("Sample from each rhythm class (32 bits):\n")
    for cls in CLASSES:
        gen, emo = CLASSES[cls]
        seq = gen(n=32, rng=rng)
        print(f"  {cls:10s} ({emo:7s})  {bits_to_str(seq)}")
