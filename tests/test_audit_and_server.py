"""Statistical soundness of the sequential audit, and detection of the randomized / evasive scenarios (test version of the headline claims)."""

from __future__ import annotations

import numpy as np
import torch

from assay.sequential_audit import EValueAudit, SPRTAudit
from benchmarks.blackbox import CanaryAuditor, reference_outputs, token_agreement

from tests.conftest import make_server


def test_evalue_anytime_validity_monte_carlo():
    """[Claim: nondeterminism-robust (false-alarm control)] The sequential false-alarm rate under H0
    (uniform p-values) does not exceed alpha=0.05 (Monte Carlo; a demonstration of Ville's inequality)."""
    rng = np.random.default_rng(0)
    alarms = 0
    n_runs, horizon = 400, 200
    for _ in range(n_runs):
        audit = EValueAudit(alpha=0.05)
        for _ in range(horizon):
            audit.update(float(rng.uniform()))
            if audit.alarmed:
                alarms += 1
                break
    assert alarms / n_runs <= 0.05 + 0.02  # Monte Carlo slack


def test_evalue_power_on_sparse_cheating():
    """[Claim: unforgeable] Even when only 10% of requests cheat, evidence accumulates monotonically
    (statistical version of power against randomized substitution)."""
    rng = np.random.default_rng(1)
    audit = EValueAudit(alpha=0.01)
    for t in range(200):
        p = 1e-10 if rng.uniform() < 0.1 else float(rng.uniform())
        audit.update(p)
        if audit.alarmed:
            break
    assert audit.alarmed and audit.t_alarm <= 60


def test_sprt_updates():
    sprt = SPRTAudit(alpha=0.01)
    for _ in range(3):
        sprt.update(8.0)
    assert sprt.alarmed


def test_random_substitution_detected_by_assay(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable] Even for randomized substitution that uses the real model with prob p=0.9,
    all-request verification detects it from just the cheating fraction."""
    server = make_server(models, sketcher, "random", seed=30)
    audit = EValueAudit(alpha=0.01)
    for i in range(80):
        p = prompts[i % len(prompts)]
        resp = server.handle(f"rnd-{i}", p)
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        audit.update(v.p)
        if audit.alarmed:
            break
    assert audit.alarmed, "randomized substitution not detected"


def test_evasive_server_fools_blackbox_but_not_assay(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable (headline)] A canary-detecting evasive server completely fools the
    black-box audit, but is caught by Assay within a few requests."""
    canaries = prompts[:4]
    refs = reference_outputs(models["M"], canaries, gen_len=8)
    server = make_server(models, sketcher, "evasive", canaries=canaries, seed=31)

    # (a) Black box: canaries are always answered with M -> zero evidence
    bb = CanaryAuditor(canaries, refs, q0=0.05, alpha=0.01, seed=0)
    for t in range(30):
        idx, c = bb.pick_canary()
        resp = server.handle(f"bb-{t}", c)
        assert resp.model_used == "M"  # evasion is working
        bb.observe(idx, resp.output_ids)
    assert not bb.alarmed

    # (b) Assay: ordinary user requests also carry a receipt -> immediate detection
    audit = EValueAudit(alpha=0.01)
    for i in range(10):
        p = prompts[8 + (i % 8)]  # a real user prompt, not a canary
        resp = server.handle(f"user-{i}", p)
        assert resp.model_used == "M'"
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        audit.update(v.p)
        if audit.alarmed:
            break
    assert audit.alarmed and audit.t_alarm <= 5


def test_token_agreement():
    assert token_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert token_agreement([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5
