"""Assay: a research prototype for per-request internal-receipt model-substitution detection.

Central claims:
  1. [unforgeable]           Every request is in scope, so canary-based evasion and
                             probabilistic substitution still leave the cheating
                             request itself as evidence.
  2. [nondeterminism-robust] Uses invariant features and a calibrated tolerance band
                             that do not move under benign numerical jitter.
  3. [low-cost]              Verification runs a slice of a full recompute (up to the
                             selected layer, over the selected positions).
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
