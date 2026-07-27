"""ActivationSketch: JL-projected sketch of hidden_states + rank/sign invariants + signed receipt.

Protocol (shared between provider and verifier, public):
  - From a fixed seed, per-layer JL random projection matrices R_l (H x k) are
    derived deterministically.
  - The inspected token positions are derived deterministically from
    PRF(seed, request_id) (drawn from the response-token region).
  - Each response is turned into a receipt (projected sketch, sign bits, top ranks)
    and signed (mocking a TEE signature).

[Claim: unevadable] The inspected token positions are a deterministic PRF keyed
by request_id, so the provider cannot tell which request is "the audited one" --
every request carries a receipt, and all of them are verifiable after the fact.
Evasion by canary-detection is structurally impossible. (The set of layers is a
fixed public protocol constant; the verifier picks which layer to recompute using
its own secret randomness -- see verifier.py.)

[Claim: nondeterminism-robust] The sign bits and top ranks are a coarse-grained
view of the continuous sketch: they do not move under benign numerical jitter
(~1e-4 relative), only under quantization / substitution
(checked in tests/test_nondeterminism.py).
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from assay.hooks import prf_int


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SketchConfig:
    """Public parameters of the sketch protocol."""

    seed: int = 20260718
    layers: Tuple[int, ...] = (3, 6, 9, 11)   # indices into the hidden_states tuple
    n_positions: int = 6                       # number of positions drawn from the response region
    sketch_dim: int = 32                       # JL projection output dim k
    rank_top_m: int = 8                        # number of top coordinates kept for the rank invariant

    @staticmethod
    def auto_layers(n_layer: int, count: int = 4) -> Tuple[int, ...]:
        """Choose inspection layers spread from shallow to deep by depth (avoid the final ln_f layer).

        [Claim: unevadable] Spreading the layer set means a substitute model has to match
        M at several depths at once, not just at one convenient point in the stack.
        [Claim: low-cost] Layers are capped below the last one so the verifier's truncated
        forward can usually stop early instead of running the whole stack.
        """
        if n_layer < count + 1:
            return tuple(range(1, n_layer))
        idxs = np.linspace(n_layer * 0.25, n_layer - 1, count)
        return tuple(sorted({int(round(i)) for i in idxs}))


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def opaque_request_id(purpose: str, seed: int, index: int) -> str:
    """Build a request id that carries no visible hint about why the request was made.

    [Claim: unevadable] Any traffic the auditor originates -- calibration runs, spot checks
    -- travels through the provider under audit. If those requests are recognizable (they
    used to be literally "cal-0", "cal-1", ...), a provider can answer them honestly and
    cheat on everything else, which is the canary-evasion attack wearing a different hat.
    The digest is opaque to the provider while staying reproducible for the auditor.
    """
    return hashlib.sha256(f"{purpose}|{seed}|{index}".encode()).hexdigest()[:24]


def token_sha(ids: Sequence[int]) -> str:
    """Hash a token sequence for receipt binding.

    [Claim: unevadable] Binding the prompt and output token hashes into the signed
    receipt is what makes replay and post-hoc substitution of a receipt detectable.
    """
    return hashlib.sha256(np.asarray(list(ids), dtype=np.int64).tobytes()).hexdigest()


@dataclass
class Receipt:
    """One request's signed internal receipt.

    sketch:  (n_layers, n_pos, k) float16 -- JL projected sketch
    signs:   (n_layers, n_pos, k) uint8   -- sign bits (invariant feature)
    ranks:   (n_layers, n_pos, m) int16   -- indices of the largest coordinates (invariant feature)
    """

    request_id: str
    model_claim: str
    prompt_sha: str
    output_sha: str
    prompt_len: int
    total_len: int
    layers: Tuple[int, ...]
    positions: Tuple[int, ...]
    sketch: np.ndarray
    signs: np.ndarray
    ranks: np.ndarray
    seed: int
    sig_alg: str = ""
    signature: str = ""

    def canonical_bytes(self) -> bytes:
        """Canonical byte string to sign (binds every field except the signature).

        [Claim: unevadable] Because it binds the prompt/output hashes and the
        request_id, "replaying an old honest receipt" or "swapping fields after
        the fact" fail signature verification
        (tests/test_sketch.py::test_replay_rejected).
        """
        meta = {
            "request_id": self.request_id,
            "model_claim": self.model_claim,
            "prompt_sha": self.prompt_sha,
            "output_sha": self.output_sha,
            "prompt_len": self.prompt_len,
            "total_len": self.total_len,
            "layers": list(self.layers),
            "positions": list(self.positions),
            "seed": self.seed,
            "sketch": base64.b64encode(np.ascontiguousarray(self.sketch, dtype=np.float16).tobytes()).decode(),
            "signs": base64.b64encode(np.ascontiguousarray(self.signs, dtype=np.uint8).tobytes()).decode(),
            "ranks": base64.b64encode(np.ascontiguousarray(self.ranks, dtype=np.int16).tobytes()).decode(),
        }
        return json.dumps(meta, sort_keys=True).encode()


# ---------------------------------------------------------------------------
# Signing (mock of a TEE signature)
# ---------------------------------------------------------------------------

class HmacSigner:
    """HMAC-SHA256 signer. Mocks a TEE attestation signature with a symmetric key.

    Its job is to prevent tampering, back-dating, and replay of receipts. The
    substitution detection itself is done by the sketch-consistency check
    (verifier.py). In a real deployment the TEE signs with a key bound to the
    code identity.
    """

    alg = "hmac-sha256"

    def __init__(self, key: bytes):
        self._key = key

    def sign(self, data: bytes) -> str:
        """Sign the receipt bytes with the shared key (stands in for a TEE attestation)."""
        return hmac_mod.new(self._key, data, hashlib.sha256).hexdigest()

    def verify(self, data: bytes, signature: str) -> bool:
        """Constant-time check that the receipt bytes were not altered after signing."""
        return hmac_mod.compare_digest(self.sign(data), signature)


class Ed25519Signer:
    """ed25519 public-key signer (cryptography). Mocks the setting where the verifier holds no secret key."""

    alg = "ed25519"

    def __init__(self, private_key=None):
        from cryptography.hazmat.primitives.asymmetric import ed25519

        self._sk = private_key or ed25519.Ed25519PrivateKey.generate()
        self._pk = self._sk.public_key()

    def sign(self, data: bytes) -> str:
        """Sign with the private key; the verifier only ever needs the public half."""
        return base64.b64encode(self._sk.sign(data)).decode()

    def verify(self, data: bytes, signature: str) -> bool:
        from cryptography.exceptions import InvalidSignature

        try:
            self._pk.verify(base64.b64decode(signature), data)
            return True
        except InvalidSignature:
            return False


# ---------------------------------------------------------------------------
# Sketcher
# ---------------------------------------------------------------------------

class ActivationSketcher:
    """Sketches hidden_states to low dimension and issues / recomputes signed receipts."""

    def __init__(self, config: SketchConfig, signer=None):
        self.config = config
        self.signer = signer
        self._proj_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # -- protocol derivation (identical on provider and verifier) ------------

    def projection(self, layer: int, hidden_dim: int) -> torch.Tensor:
        """Per-layer fixed-seed JL projection matrix (H, k). Entries N(0, 1/k), roughly length-preserving.

        [Claim: low-cost] Sketch comparison happens in k=32 dimensions and the
        receipt is a few KB. The projection is re-derivable from the seed, so no
        matrix needs to be distributed.
        """
        key = (layer, hidden_dim)
        if key not in self._proj_cache:
            gen = torch.Generator().manual_seed(prf_int(self.config.seed, "proj", layer, hidden_dim) % (2**63))
            R = torch.randn(hidden_dim, self.config.sketch_dim, generator=gen, dtype=torch.float32)
            self._proj_cache[key] = R / np.sqrt(self.config.sketch_dim)
        return self._proj_cache[key]

    def select_positions(self, request_id: str, prompt_len: int, total_len: int) -> Tuple[int, ...]:
        """Deterministically choose inspection positions in the response-token region from PRF(seed, request_id).

        [Claim: unevadable] Positions are request-specific and re-derivable -- a
        provider that re-picks positions to sketch only "convenient" places is
        rejected at verification.

        The candidate window runs from the last prompt token (the state that produced the
        first response token) through the last response token *inclusive*. Including the
        final position matters: with it excluded, a one-token response left exactly one
        candidate -- the prompt's last token -- whose hidden state is a function of the
        prompt alone, so the receipt said nothing at all about the response. One-token
        replies are ordinary in practice (classification, routing, an immediate EOS), and
        that gap let a provider serve them from anywhere while still passing. Every
        candidate from prompt_len onward depends on the response tokens.
        """
        lo = max(prompt_len - 1, 0)
        hi = total_len  # exclusive; total_len-1 is the last response token's own state
        if total_len <= 0 or hi <= lo:
            raise ValueError(
                f"no inspectable positions for prompt_len={prompt_len}, total_len={total_len}"
            )
        candidates = list(range(lo, hi))
        rng = np.random.default_rng(prf_int(self.config.seed, "pos", request_id, prompt_len, total_len))
        n = min(self.config.n_positions, len(candidates))
        picked = rng.choice(len(candidates), size=n, replace=False)
        return tuple(sorted(int(candidates[i]) for i in picked))

    # -- sketch computation --------------------------------------------------

    def sketch_vector(self, h: torch.Tensor, layer: int) -> np.ndarray:
        """Project one (layer, position) hidden state h (H,) into a k-dim sketch.

        [Claim: low-cost] The receipt carries k=32 floats per inspected point instead of
        the full hidden state (H=768 for gpt2), and the projection is re-derived from the
        seed, so nothing but the sketch has to be transmitted or stored.
        """
        R = self.projection(layer, h.shape[-1])
        s = (h.to(torch.float32) @ R).cpu().numpy()
        return s.astype(np.float32)

    def invariants(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return sign bits and top ranks (indices of the largest values) -- jitter-invariant coarse features.

        [Claim: nondeterminism-robust] These coarse features quantize away the
        continuous jitter, so they stay put under benign numerical noise and only
        move under quantization / substitution.
        """
        signs = (s > 0).astype(np.uint8)
        ranks = np.argsort(-s, kind="stable")[: self.config.rank_top_m].astype(np.int16)
        return signs, ranks

    def compute_points(
        self,
        hidden_states: Sequence[torch.Tensor],
        positions: Sequence[int],
        layers: Optional[Sequence[int]] = None,
        layer_map=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the (layers x positions) sketch / sign / rank tensors.

        layer_map: claimed-layer index -> actual index into hidden_states.
        Honest provider and verifier use the identity map. An attacker using a
        cheaper model with a different layer count attempts a "best-effort"
        forgery here (e.g. proportional mapping) -- see server.py.

        [Claim: unevadable] The receipt is computed from whatever model actually
        ran; a receipt built from M''s activations cannot match M's recomputation.
        """
        layers = list(layers if layers is not None else self.config.layers)
        layer_map = layer_map or (lambda l: l)
        k, m = self.config.sketch_dim, self.config.rank_top_m
        sketch = np.zeros((len(layers), len(positions), k), dtype=np.float32)
        signs = np.zeros((len(layers), len(positions), k), dtype=np.uint8)
        ranks = np.zeros((len(layers), len(positions), m), dtype=np.int16)
        for li, layer in enumerate(layers):
            hs = hidden_states[layer_map(layer)]
            for pi, pos in enumerate(positions):
                s = self.sketch_vector(hs[pos], layer)
                sketch[li, pi] = s
                signs[li, pi], ranks[li, pi] = self.invariants(s)
        return sketch, signs, ranks

    # -- receipt issuance ----------------------------------------------------

    def build_receipt(
        self,
        request_id: str,
        model_claim: str,
        prompt_ids: Sequence[int],
        output_ids: Sequence[int],
        hidden_states: Sequence[torch.Tensor],
        layer_map=None,
    ) -> Receipt:
        """Issue one signed receipt for a response (mocking the provider side / inside a TEE).

        [Claim: unevadable] Every response gets a receipt -- there is no sampling step and
        no way for the provider to tell an audited request from an ordinary one, so
        canary-detection evasion has nothing to detect. The receipt is computed from the
        activations of whichever model actually ran, which is exactly what gives the
        cheating request away.
        """
        prompt_len = len(prompt_ids)
        total_len = prompt_len + len(output_ids)
        positions = self.select_positions(request_id, prompt_len, total_len)
        sketch, signs, ranks = self.compute_points(hidden_states, positions, layer_map=layer_map)
        receipt = Receipt(
            request_id=request_id,
            model_claim=model_claim,
            prompt_sha=token_sha(prompt_ids),
            output_sha=token_sha(output_ids),
            prompt_len=prompt_len,
            total_len=total_len,
            layers=tuple(self.config.layers),
            positions=positions,
            sketch=sketch.astype(np.float16),
            signs=signs,
            ranks=ranks,
            seed=self.config.seed,
        )
        if self.signer is not None:
            receipt.sig_alg = self.signer.alg
            receipt.signature = self.signer.sign(receipt.canonical_bytes())
        return receipt
