"""Verifier correctness: exact truncated-forward match, low cost, no false alarms, substitution detection."""

from __future__ import annotations

import numpy as np
import torch

from assay.hooks import capture_hidden_states
from assay.sequential_audit import EValueAudit
from assay.verifier import truncated_hidden_state

from tests.conftest import make_server


def test_truncated_forward_matches_full(models):
    """[Claim: low-cost] The truncated forward matches the full forward's hidden_states exactly
    at every layer (so partial recompute does not change what verification means)."""
    M = models["M"]
    ids = torch.arange(10) % 50
    full = capture_hidden_states(M, ids)
    n = len(full) - 1
    for layer in range(1, n + 1):
        h, blocks_run = truncated_hidden_state(M, ids, layer)
        assert torch.allclose(full[layer], h, atol=1e-6), f"layer {layer} mismatch"
        expected_blocks = n if layer == n else layer + 1
        assert blocks_run == expected_blocks


def test_causal_token_truncation(models):
    """Under causal attention, truncating the sequence to the inspected positions does not change
    the activations (justifies the token-axis cost reduction)."""
    M = models["M"]
    ids = torch.arange(12) % 50
    full = capture_hidden_states(M, ids)
    h_trunc, _ = truncated_hidden_state(M, ids[:7], 2)
    assert torch.allclose(full[2][:7], h_trunc, atol=1e-6)


def test_honest_no_false_alarm(models, sketcher, prompts, calibrated_verifier):
    """[Claim: nondeterminism-robust] No sequential alarm fires over 60 honest requests with jitter."""
    server = make_server(models, sketcher, "honest", seed=5, jitter_seed=11)
    audit = EValueAudit(alpha=0.01)
    zs = []
    for i in range(60):
        p = prompts[i % len(prompts)]
        resp = server.handle(f"h-{i}", p)
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        assert v.valid_receipt
        zs.append(v.z)
        audit.update(v.p)
    assert not audit.alarmed, f"false alarm at t={audit.t_alarm}, log_e={audit.log_e:.1f}"
    # No extreme z at the per-request level either (calibration sanity)
    assert float(np.mean(np.asarray(zs) > 4.0)) < 0.05


def test_detect_size_substitution(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable] Size substitution (4 layers -> 2) alarms within a few requests."""
    server = make_server(models, sketcher, "cheat", mp_key="size", seed=6)
    audit = EValueAudit(alpha=0.01)
    for i in range(10):
        p = prompts[i % len(prompts)]
        resp = server.handle(f"s-{i}", p)
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        audit.update(v.p)
        if audit.alarmed:
            break
    assert audit.alarmed and audit.t_alarm <= 5


def test_detect_quantization(models, sketcher, prompts, calibrated_verifier):
    """[Claim: nondeterminism-robust] INT8 quantization shifts activations orders of magnitude more
    than jitter and is detected within a few requests by the calibrated tolerance band."""
    server = make_server(models, sketcher, "cheat", mp_key="quant", seed=7)
    audit = EValueAudit(alpha=0.01)
    zs = []
    for i in range(20):
        p = prompts[i % len(prompts)]
        resp = server.handle(f"q-{i}", p)
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        zs.append(v.z)
        audit.update(v.p)
        if audit.alarmed:
            break
    assert audit.alarmed and audit.t_alarm <= 10, f"zs={zs}"


def test_verifier_cost_sublinear(models, sketcher, prompts, calibrated_verifier):
    """[Claim: low-cost] The verification layer-token cost is strictly smaller than a full recompute."""
    server = make_server(models, sketcher, "honest", seed=8)
    ratios = []
    for i in range(12):
        p = prompts[i]
        resp = server.handle(f"c-{i}", p)
        v = calibrated_verifier.verify(resp.receipt, p, resp.output_ids)
        ratios.append(v.cost_ratio)
    assert 0 < float(np.mean(ratios)) < 1.0
    assert max(ratios) <= 1.0
