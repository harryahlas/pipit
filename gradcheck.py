"""Gradient check for Brain — verifies hand-derived backprop matches numerical."""
import numpy as np
import sys
sys.path.insert(0, '/home/claude/bittern')

from bittern import Brain, VOCAB_SIZE


def numerical_gradient(f, w, eps=1e-5):
    """Centered finite difference."""
    g = np.zeros_like(w)
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = w[ix]
        w[ix] = old + eps
        f_plus = f()
        w[ix] = old - eps
        f_minus = f()
        w[ix] = old
        g[ix] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return g


def check():
    rng = np.random.default_rng(42)
    brain = Brain(embed_dim=8, brain_dim=8, context=8, rng=rng)
    bits = [0, 1, 1, 0, 1]
    target = 1

    # Snapshot weights so we can recompute analytic grads on a fresh state
    # (train_step applies updates in place, so we need to capture grads
    # before applying them — easiest path: capture via lr=0)

    # Get analytic gradient by running train_step with lr=0 and comparing
    # what the update WOULD have been. We need to expose grads. Easier:
    # do a manual forward+backward and capture.

    def loss_fn():
        bs, _ = brain.encode(bits)
        logits = bs @ brain.W_out + brain.b_out
        logits = logits - logits.max()
        e = np.exp(logits)
        probs = e / e.sum()
        return -np.log(probs[target] + 1e-12)

    # Use a custom run-through that captures gradients without applying them
    # — easiest: temporarily snapshot/restore weights around train_step,
    # diff the update, divide by lr.
    lr = 1e-3

    def grad_for(w_attr):
        w_before = getattr(brain, w_attr).copy()
        # train_step applies: w -= lr * d_w  =>  d_w = (w_before - w_after) / lr
        # but we want to test ONE param at a time, so save/restore others.
        all_params = ['embedding', 'W_q', 'W_k', 'W_v', 'W_o', 'W_out', 'b_out']
        snap = {p: getattr(brain, p).copy() for p in all_params}
        brain.train_step(bits, target, lr=lr)
        w_after = getattr(brain, w_attr).copy()
        for p in all_params:
            setattr(brain, p, snap[p])
        return (w_before - w_after) / lr

    print("Gradient check (analytic vs numerical):")
    print(f"  {'param':<12}  max_abs_err   mean_abs_err   ok?")
    all_ok = True
    for pname in ['W_out', 'b_out', 'W_o', 'W_q', 'W_k', 'W_v', 'embedding']:
        g_analytic = grad_for(pname)
        w = getattr(brain, pname)
        g_numerical = numerical_gradient(loss_fn, w, eps=1e-5)
        err = np.abs(g_analytic - g_numerical)
        max_err = err.max()
        mean_err = err.mean()
        ok = max_err < 1e-4
        all_ok &= ok
        print(f"  {pname:<12}  {max_err:.2e}     {mean_err:.2e}     "
              f"{'✓' if ok else '✗'}")

    print()
    print(f"Result: {'ALL OK' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    ok = check()
    sys.exit(0 if ok else 1)
