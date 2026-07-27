"""Assay: a research prototype for per-request internal-receipt model-substitution detection.

Central claims:
  1. [unevadable]            Every request is in scope, so canary-based evasion and
                             probabilistic substitution still leave the cheating
                             request itself as evidence. Two things this does *not* say.
                             The provider owns the signing key (the TEE is simulated), so
                             it is not a claim of cryptographic unforgeability. And a
                             matching sketch only shows M was *run over* the text, not
                             that M *produced* it -- closing that gap needs the
                             generation-consistency check (see verifier.py and
                             ProviderServer(mode="forge"), which breaks the sketch check
                             outright and is caught only by that check).
  2. [nondeterminism-robust] Uses invariant features and a calibrated tolerance band
                             that do not move under benign numerical jitter.
  3. [low-cost]              Verification runs a slice of a full recompute (up to the
                             drawn layer, over the inspected positions): a constant-factor
                             saving, ~0.74x measured, including the full-depth passes the
                             generation check needs on a minority of requests.
"""

from assay.hooks import (
    capture_hidden_states,
    fake_quantize_int8_,
    load_model,
    load_tokenizer,
    tiny_random_pair,
)
from assay.sketch import ActivationSketcher, Receipt, SketchConfig, HmacSigner, Ed25519Signer
from assay.verifier import Verifier, Calibrator, Calibration, VerdictRecord
from assay.nondeterminism import NondeterminismModel
from assay.sequential_audit import EValueAudit, SPRTAudit
from assay.server import ProviderServer, ServerResponse

__all__ = [
    "capture_hidden_states",
    "fake_quantize_int8_",
    "load_model",
    "load_tokenizer",
    "tiny_random_pair",
    "ActivationSketcher",
    "Receipt",
    "SketchConfig",
    "HmacSigner",
    "Ed25519Signer",
    "Verifier",
    "Calibrator",
    "Calibration",
    "VerdictRecord",
    "NondeterminismModel",
    "EValueAudit",
    "SPRTAudit",
    "ProviderServer",
    "ServerResponse",
]
