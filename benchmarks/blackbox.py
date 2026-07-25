"""Black-box statistical audit baseline (RUT-style: output text / simple distribution comparison).

The external auditor only sees API response text. Two variants are implemented:
  1. CanaryAuditor: send canary prompts whose reference greedy output under M was
     recorded in advance, and run a sequential test (Bernoulli e-process) on the
     agreement of the response token sequence.
  2. DistributionAuditor: compare the unigram distribution of output tokens against a
     reference distribution (a simple distribution test; measures the sample size needed).

Structural weaknesses (implemented for contrast with Assay, while giving it a fair best effort):
  - Coverage: audit queries are a fraction of traffic (or a separate budget) -- real user responses are never checked.
  - Randomized substitution: an audit query only hits M' with probability 1-p, so detection is 1/(1-p)x slower.
  - Audit evasion: if the attacker detects a canary and answers with M, detection becomes impossible in principle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def token_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Position-wise agreement rate of two output token sequences (aligned to the shorter one)."""
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return float(np.mean([1.0 if a[i] == b[i] else 0.0 for i in range(n)]))


class BernoulliEProcess:
    """Sequential e-process on the mismatch indicator (mixture likelihood-ratio bet).

    H0: mismatch ~ Bernoulli(q0) (calibrated on honest operation). At each step it
    averages the likelihood ratios of several alternatives q1 (mixing preserves E[e|H0]=1).
    """

    def __init__(self, q0: float = 0.02, alpha: float = 0.01, alts: Tuple[float, ...] = (0.1, 0.3, 0.8)):
        self.q0 = min(max(q0, 1e-4), 0.5)
        self.alpha = alpha
        self.alts = alts
        self.log_e = 0.0
        self.t = 0
        self.t_alarm: Optional[int] = None
        self.history: List[float] = []

    @property
    def threshold(self) -> float:
        return math.log(1.0 / self.alpha)

    @property
    def alarmed(self) -> bool:
        return self.t_alarm is not None

    def update(self, mismatch: bool) -> None:
        x = 1.0 if mismatch else 0.0
        lrs = [
            (q1 / self.q0) ** x * ((1 - q1) / (1 - self.q0)) ** (1 - x)
            for q1 in self.alts
        ]
        self.t += 1
        self.log_e += math.log(max(float(np.mean(lrs)), 1e-300))
        self.log_e = max(self.log_e, -50.0)
        self.history.append(self.log_e)
        if self.t_alarm is None and self.log_e >= self.threshold:
            self.t_alarm = self.t


class CanaryAuditor:
    """External sampling audit via canary prompts + reference greedy outputs.

    coverage = number of audit queries / total traffic. Recorded for contrast with
    Assay (=1.0).
    """

    def __init__(
        self,
        canaries: Sequence[torch.Tensor],
        references: Sequence[List[int]],
        q0: float = 0.02,
        alpha: float = 0.01,
        match_threshold: float = 0.9,
        seed: int = 0,
    ):
        assert len(canaries) == len(references)
        self.canaries = list(canaries)
        self.references = list(references)
        self.match_threshold = match_threshold
        self.eproc = BernoulliEProcess(q0=q0, alpha=alpha)
        self._rng = np.random.default_rng(seed)

    def pick_canary(self) -> Tuple[int, torch.Tensor]:
        i = int(self._rng.integers(len(self.canaries)))
        return i, self.canaries[i]

    def observe(self, canary_idx: int, output_ids: Sequence[int]) -> None:
        agree = token_agreement(self.references[canary_idx], list(output_ids))
        self.eproc.update(agree < self.match_threshold)

    @property
    def alarmed(self) -> bool:
        return self.eproc.alarmed

    @property
    def t_alarm(self) -> Optional[int]:
        return self.eproc.t_alarm


@torch.no_grad()
def reference_outputs(model: nn.Module, canaries: Sequence[torch.Tensor], gen_len: int) -> List[List[int]]:
    """Reference greedy outputs the auditor prepares in advance (computed on clean M)."""
    refs = []
    for c in canaries:
        ids = c.flatten().unsqueeze(0)
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=gen_len,
            do_sample=False,
            pad_token_id=getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", 0) or 0,
        )
        refs.append([int(x) for x in out[0, ids.shape[1]:].tolist()])
    return refs


class DistributionAuditor:
    """Simple distribution audit via the L1 distance of the output unigram distribution.

    The reference distribution is built from M's sampled outputs. The detection
    threshold is calibrated by the spread of M's own resampling (bootstrap). Returns
    an estimate of the number of queries needed.
    """

    def __init__(self, ref_tokens: Sequence[int], vocab_size: int, n_boot: int = 200, seed: int = 0):
        self.vocab = vocab_size
        self.ref = np.bincount(np.asarray(ref_tokens), minlength=vocab_size).astype(float)
        self.ref /= max(self.ref.sum(), 1.0)
        self._rng = np.random.default_rng(seed)
        self._ref_tokens = np.asarray(ref_tokens)
        self._n_boot = n_boot

    def _l1(self, tokens: np.ndarray) -> float:
        h = np.bincount(tokens, minlength=self.vocab).astype(float)
        h /= max(h.sum(), 1.0)
        return float(np.abs(h - self.ref).sum())

    def threshold(self, n_tokens: int, quantile: float = 0.99) -> float:
        """Quantile of the L1 distance for an M resample of the same size (calibrated detection threshold)."""
        vals = []
        for _ in range(self._n_boot):
            sub = self._rng.choice(self._ref_tokens, size=n_tokens, replace=True)
            vals.append(self._l1(sub))
        return float(np.quantile(vals, quantile))

    def detects(self, observed_tokens: Sequence[int]) -> bool:
        obs = np.asarray(list(observed_tokens))
        return self._l1(obs) > self.threshold(len(obs))
