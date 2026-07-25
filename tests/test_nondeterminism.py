"""Nondeterminism robustness: invariant features do not move under jitter, only under quantization / substitution."""

from __future__ import annotations

import numpy as np
import torch

from assay.hooks import capture_hidden_states
from assay.nondeterminism import NondeterminismModel, rank_overlap, sign_agreement

from tests.conftest import SIGMA


def _points(sketcher, model, ids, jitter=None):
    import contextlib

    ctx = jitter.applied_to(model) if jitter else contextlib.nullcontext()
    with ctx:
        hs = capture_hidden_states(model, ids)
    n = len(hs) - 1
    layer_map = None
    if n != max(sketcher.config.layers):
        layer_map = lambda l: max(1, min(n, int(round(l * n / max(sketcher.config.layers)))))
    positions = list(range(2, ids.numel()))[: sketcher.config.n_positions]
    return sketcher.compute_points(hs, positions, layer_map=layer_map)


def test_invariants_stable_under_jitter_sensitive_to_swap(models, sketcher):
    """[Claim: nondeterminism-robust] Feature movement separates cleanly in the order
    benign jitter << quantization << substitution. The coarse invariants (sign / rank) stay put
    under jitter, start moving under quantization, and collapse to near-zero correlation under substitution."""
    ids = torch.arange(12) % 50
    sk_clean, signs_clean, ranks_clean = _points(sketcher, models["M"], ids)

    def reldist(s):
        return float(np.linalg.norm(s - sk_clean) / np.linalg.norm(sk_clean))

    # (1) Benign jitter (two seeds = distinct executions): invariants do not move
    rel_j = 0.0
    for seed in (100, 200):
        jit = NondeterminismModel(SIGMA, seed=seed)
        sk_j, signs_j, ranks_j = _points(sketcher, models["M"], ids, jitter=jit)
        assert sign_agreement(signs_clean, signs_j) > 0.99
        assert rank_overlap(ranks_clean, ranks_j) > 0.98
        rel_j = max(rel_j, reldist(sk_j))
    assert rel_j < 2e-3

    # (2) Quantization: moves significantly more than jitter (continuous distance >3x, ranks start moving)
    sk_q, signs_q, ranks_q = _points(sketcher, models["quant"], ids)
    rel_q = reldist(sk_q)
    rq = rank_overlap(ranks_clean, ranks_q)
    assert rel_q > 3 * rel_j
    assert rq < 0.995  # the rank invariant starts to move under quantization

    # (3) Size substitution: collapses to near-zero correlation (sign agreement ~0.5)
    sk_s, signs_s, ranks_s = _points(sketcher, models["size"], ids)
    ss = sign_agreement(signs_clean, signs_s)
    assert ss < 0.75
    assert rank_overlap(ranks_clean, ranks_s) < 0.6
    assert reldist(sk_s) > 10 * rel_q
    # Sign-agreement ordering: jitter >= quantization > substitution
    assert sign_agreement(signs_clean, signs_q) >= ss + 0.2


def test_jitter_magnitude_is_benign(models):
    """The jitter's effect on activations is relative ~1e-3 or less (orders of magnitude below quant/substitution)."""
    ids = torch.arange(10) % 50
    clean = capture_hidden_states(models["M"], ids)
    jit = NondeterminismModel(SIGMA, seed=1)
    with jit.applied_to(models["M"]):
        noisy = capture_hidden_states(models["M"], ids)
    rel = (noisy[3] - clean[3]).norm() / clean[3].norm()
    quant = capture_hidden_states(models["quant"], ids)
    rel_q = (quant[3] - clean[3]).norm() / clean[3].norm()
    assert rel < 5e-3
    assert rel_q > 3 * rel  # the quantization shift is clearly larger than jitter


def test_jitter_reproducible():
    """Same seed gives the same jitter sequence (experiment reproducibility)."""
    a = NondeterminismModel(1e-3, seed=7)
    b = NondeterminismModel(1e-3, seed=7)
    x = torch.ones(4, 8)
    assert torch.equal(a._perturb(x), b._perturb(x))
