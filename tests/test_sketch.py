"""Receipt binding and signing: tampering, replay, and protocol deviation fail the structural check."""

from __future__ import annotations

import numpy as np
import torch

from assay.sketch import Ed25519Signer, HmacSigner

from tests.conftest import make_server


def test_signature_roundtrip_and_tamper(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable] A correct receipt passes signature verification, and flipping a
    single byte of the sketch makes it fail (mock TEE signature)."""
    server = make_server(models, sketcher, "honest", seed=20)
    p = prompts[0]
    resp = server.handle("sig-0", p)
    ok, reason = calibrated_verifier.structural_check(resp.receipt, p, resp.output_ids)
    assert ok, reason

    resp.receipt.sketch[0, 0, 0] += 1.0
    ok, reason = calibrated_verifier.structural_check(resp.receipt, p, resp.output_ids)
    assert not ok and reason == "bad_signature"


def test_replay_rejected(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable] Reusing a past honest receipt for a different request (replay) is
    rejected by the prompt/output hash binding."""
    server = make_server(models, sketcher, "honest", seed=21)
    p0, p1 = prompts[0], prompts[1]
    old = server.handle("r-0", p0)
    new = server.handle("r-1", p1)
    verdict = calibrated_verifier.verify(old.receipt, p1, new.output_ids)
    assert not verdict.valid_receipt
    assert verdict.p <= 1e-12  # becomes maximal evidence immediately in the sequential audit


def test_position_derivation_enforced(models, sketcher, prompts, calibrated_verifier):
    """[Claim: unforgeable] A receipt where the provider re-picked the inspection positions is
    rejected by the PRF re-derivation check."""
    server = make_server(models, sketcher, "honest", seed=22)
    p = prompts[2]
    resp = server.handle("pos-0", p)
    r = resp.receipt
    shifted = tuple(max(0, x - 1) for x in r.positions)
    if shifted == r.positions:
        shifted = tuple(x + 1 for x in r.positions)
    r.positions = shifted
    r.signature = sketcher.signer.sign(r.canonical_bytes())  # even assuming the attacker holds the key
    ok, reason = calibrated_verifier.structural_check(r, p, resp.output_ids)
    assert not ok and reason == "position_derivation_mismatch"


def test_ed25519_signer():
    """The ed25519 signer (public-key verification) works through the same interface as HMAC."""
    s = Ed25519Signer()
    data = b"receipt-bytes"
    sig = s.sign(data)
    assert s.verify(data, sig)
    assert not s.verify(b"receipt-bytez", sig)


def test_sketch_deterministic(models, sketcher):
    """Projection and position selection are deterministic from the seed (re-derivable identically by provider and verifier)."""
    R1 = sketcher.projection(2, 64)
    R2 = sketcher.projection(2, 64)
    assert torch.equal(R1, R2)
    pos1 = sketcher.select_positions("req-x", 8, 16)
    pos2 = sketcher.select_positions("req-x", 8, 16)
    assert pos1 == pos2
    assert pos1 != sketcher.select_positions("req-y", 8, 16)
