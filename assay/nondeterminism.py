"""NondeterminismModel: an injection model of benign numerical nondeterminism (jitter).

In a real inference stack, differences in GPU parallel-reduction order, kernel
selection, and batch shape cause hidden_states to wobble by ~1e-5 to ~1e-3
relative even for the same model and input. This module models that as a small
relative-noise injection at each block output, and uses it for both calibration
(Calibrator) and robustness tests.

[Claim: nondeterminism-robust] Assay's tolerance band is calibrated on honest
traffic under this jitter, so benign jitter produces no false alarms. Quantization
(weight relative error ~0.4%) and model substitution shift activations by orders
of magnitude more, so the same statistics separate them cleanly
(checked in tests/test_nondeterminism.py).
"""

from __future__ import annotations

import contextlib
from typing import Iterator, List

import torch
from torch import nn

from assay.hooks import layer_list


class NondeterminismModel:
    """Injects small relative-scale Gaussian noise into each transformer block output.

    sigma_rel: noise amplitude relative to the block output's standard deviation.
               3e-4 is a conservative (deliberately large) approximation of the
               wobble seen across different GPUs / execution orders; the point is
               that separation from quantization / substitution holds even when we
               calibrate at this magnitude (a smaller jitter would leave more room).
    seed:      seed for reproducible injected noise. The sequence advances per call
               (different jitter per request) yet the whole run is reproducible.
    """

    def __init__(self, sigma_rel: float = 3e-4, seed: int = 0):
        self.sigma_rel = float(sigma_rel)
        self._gen = torch.Generator().manual_seed(seed)

    def _perturb(self, h: torch.Tensor) -> torch.Tensor:
        if self.sigma_rel <= 0:
            return h
        noise = torch.randn(h.shape, generator=self._gen, dtype=torch.float32)
        return h + self.sigma_rel * h.detach().float().std() * noise.to(h.dtype)

    @contextlib.contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[None]:
        """Context manager that temporarily attaches jitter-injection hooks to every block.

        Noise is added to each block output and propagates to later blocks (like
        real nondeterminism, it accumulates with depth). It is applied only on the
        provider's generation / receipt path; the verifier recomputes cleanly, and
        that gap is absorbed by calibration.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def hook(_module, _args, output):
            if isinstance(output, tuple):
                return (self._perturb(output[0]),) + tuple(output[1:])
            return self._perturb(output)

        try:
            for block in layer_list(model):
                handles.append(block.register_forward_hook(hook))
            yield
        finally:
            for h in handles:
                h.remove()


def sign_agreement(signs_a, signs_b) -> float:
    """Sign-bit agreement rate -- stays ~1.0 under jitter, drops toward chance under substitution.

    [Claim: nondeterminism-robust] A coarse-grained read of the sketch: benign
    jitter is far too small to flip the sign of a projected coordinate, while a
    different model changes the activation direction outright.
    """
    import numpy as np

    a, b = (x.astype(bool) for x in (signs_a, signs_b))
    return float((a == b).mean())


def rank_overlap(ranks_a, ranks_b) -> float:
    """Mean Jaccard overlap of the top-rank sets -- the second invariant feature.

    [Claim: nondeterminism-robust] Ordering of the largest coordinates survives
    jitter but not quantization or substitution, which is the separation shown in
    figures/fig_invariance.png.
    """
    import numpy as np

    a = ranks_a.reshape(-1, ranks_a.shape[-1])
    b = ranks_b.reshape(-1, ranks_b.shape[-1])
    vals = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        vals.append(len(sa & sb) / max(len(sa | sb), 1))
    return float(np.mean(vals))
