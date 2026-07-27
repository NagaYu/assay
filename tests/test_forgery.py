"""The teacher-forcing forgery, and the check that closes it.

The sketch comparison proves that M was run over a token sequence. It does not prove that
M *produced* that sequence -- and those are not the same claim. A provider can decode
cheaply with M', push (prompt + M' output) through M once with teacher forcing, and read a
completely genuine sketch off that pass. Every structural check passes, the sketches agree,
and the user still received M' text.

These tests pin down both halves: that the attack really does defeat the sketch check on its
own, and that the generation-consistency check catches it.
"""

from __future__ import annotations

import numpy as np
import torch

from assay.sequential_audit import EValueAudit
from assay.verifier import Verifier

from tests.conftest import make_server


def _audit(models, sketcher, calibration, mode, deep_rate, prompts, n=40, seed=3):
    server = make_server(models, sketcher, mode, seed=seed)
    verifier = Verifier(models["M"], sketcher, sketcher.signer, calibration,
                        verifier_seed=99, deep_check_rate=deep_rate)
    audit = EValueAudit(alpha=0.01)
    deep = mismatches = 0
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        resp = server.handle(f"fg-{i}", prompt)
        verdict = verifier.verify(resp.receipt, prompt, resp.output_ids,
                                  expected_request_id=f"fg-{i}")
        assert verdict.valid_receipt, verdict.reason  # the forgery is structurally perfect
        deep += verdict.deep_checked
        mismatches += verdict.gen_mismatch
        audit.update(verdict.p)
        if audit.alarmed:
            break
    return audit, deep, mismatches


def test_substituted_model_actually_changes_the_output(models, sketcher, prompts):
    """The premise of every other test here: M and M' must disagree on what to emit.

    A randomly-initialized GPT-2 pair does not, unless the fixture forces it to -- both
    models just echo the last prompt token, which makes M' a perfect stand-in for M and
    makes the whole substitution question vacuous.
    """
    honest = make_server(models, sketcher, "honest", seed=1)
    cheat = make_server(models, sketcher, "cheat", seed=1)
    differing = sum(honest.handle(f"a-{i}", p).output_ids != cheat.handle(f"b-{i}", p).output_ids
                    for i, p in enumerate(prompts[:6]))
    assert differing >= 5


def test_forgery_defeats_the_sketch_check_alone(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unevadable] Without the generation check, the forgery is invisible.

    This is the negative result the defense exists for; if it ever starts passing, the
    generation check has stopped being the thing that catches this.
    """
    audit, deep, mismatches = _audit(models, sketcher, calibrated_verifier.calibration,
                                     "forge", 0.0, prompts, n=40)
    assert deep == 0 and mismatches == 0
    assert not audit.alarmed, f"forgery was caught without the deep check (log_e={audit.log_e:.1f})"
    assert audit.log_e < 0  # the evidence actually drifts *away* from cheating


def test_generation_check_catches_the_forgery(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unevadable] Sampling the full stack on a minority of requests closes it."""
    audit, deep, mismatches = _audit(models, sketcher, calibrated_verifier.calibration,
                                     "forge", 0.25, prompts, n=40)
    assert audit.alarmed, f"forgery survived {deep} deep checks (log_e={audit.log_e:.1f})"
    assert mismatches > 0


def test_honest_traffic_has_no_generation_mismatches(models, sketcher, prompts, calibrated_verifier):
    """[Claim: nondeterminism-robust] The deep check must not fire on an honest provider.

    Greedy decoding under jitter can flip a near-tie, which is why the null rate is
    calibrated rather than assumed to be zero -- but it should stay rare.
    """
    server = make_server(models, sketcher, "honest", seed=12)
    verifier = Verifier(models["M"], sketcher, sketcher.signer,
                        calibrated_verifier.calibration, verifier_seed=7, deep_check_rate=1.0)
    audit = EValueAudit(alpha=0.01)
    checked = mismatched = 0
    for i in range(20):
        prompt = prompts[i % len(prompts)]
        resp = server.handle(f"hg-{i}", prompt)
        verdict = verifier.verify(resp.receipt, prompt, resp.output_ids)
        assert verdict.deep_checked
        checked += verdict.gen_checked
        mismatched += verdict.gen_mismatch
        audit.update(verdict.p)
    assert checked > 0
    assert mismatched / checked < 0.05, f"{mismatched}/{checked} honest positions mismatched"
    assert not audit.alarmed


def test_deep_check_never_costs_more_than_a_full_recompute(models, sketcher, prompts, calibrated_verifier):
    """[Claim: low-cost] The deep check reuses its own forward for the sketch comparison.

    Run as two separate passes it cost 1.75x a full recompute on a deep-checked request --
    worse than just recomputing everything, which would have made the cost claim false for
    those requests.
    """
    server = make_server(models, sketcher, "honest", seed=14)
    verifier = Verifier(models["M"], sketcher, sketcher.signer,
                        calibrated_verifier.calibration, verifier_seed=5, deep_check_rate=1.0)
    ratios = []
    for i in range(8):
        prompt = prompts[i]
        resp = server.handle(f"dc-{i}", prompt)
        ratios.append(verifier.verify(resp.receipt, prompt, resp.output_ids).cost_ratio)
    assert max(ratios) <= 1.0, f"deep check exceeded a full recompute: {max(ratios):.2f}"
