"""SequentialAudit: combine per-request verdicts with an e-value / SPRT.

Each request's p-value (from the verifier's calibrated z-value) is turned into an
anytime-valid e-process and accumulated. An alarm fires when the product crosses
the 1/alpha threshold ("substitution present").

[Claim: unforgeable] Even under probabilistic substitution (M with prob p),
evidence accumulates every time a cheating request slips in -- unlike sampling
audits, there is no "the audit happened not to hit it" escape hatch (every request
contributes to the e-process).

[Claim: nondeterminism-robust] By Ville's inequality, if honest (H0: p-values are
super-uniform), the false-alarm probability is <= alpha at any stopping time.
Jitter is calibrated away, so it belongs to H0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuditState:
    t: int
    stat: float          # cumulative log e-value (or cumulative LLR)
    alarmed: bool
    t_alarm: Optional[int]


class EValueAudit:
    """Mixture e-value: e(p) = (1-lam) + lam * kappa * p^(kappa-1).

    - kappa * p^(kappa-1) is a p-value calibrator (kappa in (0,1) gives E[e] = 1 under H0).
    - Mixing with 1 (via lam) keeps the honest-case drift near ~0 while still
      giving a huge e to a small p (clear cheating).
      -> retains power against "rare" cheating, e.g. probabilistic substitution.
    - Alarm when the cumulative product exceeds 1/alpha (anytime-valid, Ville).
    """

    def __init__(self, alpha: float = 0.01, lam: float = 0.5, kappa: float = 0.5):
        assert 0 < alpha < 1 and 0 < lam <= 1 and 0 < kappa < 1
        self.alpha = alpha
        self.lam = lam
        self.kappa = kappa
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

    def update(self, p: float) -> AuditState:
        """Ingest one request's p-value and advance the cumulative log e-value.

        [Claim: unforgeable] Every request feeds the same e-process, so sparse
        cheating (randomized substitution) still accumulates evidence monotonically
        in expectation -- there is no subset of traffic the provider can cheat on
        "for free".
        [Claim: nondeterminism-robust] Under H0 the increment has E[e] = 1, so
        honest jitter-driven p-values cannot push the process across the threshold
        except with probability <= alpha (Ville).
        """
        p = min(max(p, 1e-300), 1.0)
        e = (1.0 - self.lam) + self.lam * self.kappa * p ** (self.kappa - 1.0)
        self.t += 1
        self.log_e += math.log(e)
        self.history.append(self.log_e)
        if self.t_alarm is None and self.log_e >= self.threshold:
            self.t_alarm = self.t
        return AuditState(self.t, self.log_e, self.alarmed, self.t_alarm)


class SPRTAudit:
    """Wald SPRT: sequential likelihood-ratio test of H0: z ~ N(0,1) vs H1: z ~ N(delta,1).

    LLR_t = sum(delta * z_t - delta^2 / 2). Alarm at the upper threshold log((1-beta)/alpha).
    Implemented as a contrast to the e-value version (delta is the smallest assumed
    cheating shift). For a monitoring use case we do not stop at the lower (accept)
    threshold but keep accumulating.
    """

    def __init__(self, delta: float = 4.0, alpha: float = 0.01, beta: float = 0.05, z_cap: float = 20.0):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.z_cap = z_cap   # per-request contribution cap (insurance against a single outlier detonating)
        self.llr = 0.0
        self.t = 0
        self.t_alarm: Optional[int] = None
        self.history: List[float] = []

    @property
    def threshold(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def alarmed(self) -> bool:
        return self.t_alarm is not None

    def update(self, z: float) -> AuditState:
        """Advance the log-likelihood ratio with one request's z-value.

        [Claim: unforgeable] Same all-request coverage as the e-value audit; kept as
        a parametric contrast so the headline result does not depend on one
        particular sequential test.
        """
        z = min(max(float(z), -self.z_cap), self.z_cap)
        self.t += 1
        self.llr += self.delta * z - 0.5 * self.delta**2
        self.llr = max(self.llr, -50.0)  # monitoring: don't sink too far on the low side
        self.history.append(self.llr)
        if self.t_alarm is None and self.llr >= self.threshold:
            self.t_alarm = self.t
        return AuditState(self.t, self.llr, self.alarmed, self.t_alarm)
