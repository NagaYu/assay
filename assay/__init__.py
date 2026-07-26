"""Assay: a research prototype for per-request internal-receipt model-substitution detection.

Central claims:
  1. [unevadable]            Every request is in scope, so canary-based evasion and
                             probabilistic substitution still leave the cheating
                             request itself as evidence. Note the provider owns the
                             signing key (the TEE is simulated), so this is *not* a
                             claim of cryptographic unforgeability: it is that a receipt
                             agreeing with M's activations cannot be produced without
                             running M, and that no request escapes the check.
  2. [nondeterminism-robust] Uses invariant features and a calibrated tolerance band
                             that do not move under benign numerical jitter.
  3. [low-cost]              Verification runs a slice of a full recompute (up to the
                             drawn layer, over the inspected positions, never the
                             lm_head): a constant-factor saving, ~0.63x measured.
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
