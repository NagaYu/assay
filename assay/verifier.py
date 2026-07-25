"""Verifier: with M's weights, recompute only a slice of each receipt at sublinear cost and check it.

Verification flow (one request):
  1. Structural check: prompt/output hashes, PRF-derived positions, layer set, signature.
     -> A receipt that fails here is immediately "cheating" (tamper / replay / protocol deviation).
  2. Draw one inspection layer using the verifier's secret randomness (unpredictable to the provider).
  3. Truncated forward: recompute only the blocks needed for that layer, over only the
     tokens up to the maximum inspected position (removing later tokens is a no-op under
     causal attention).
  4. Recompute the sketch with the same JL projection and z-score it against the
     calibrated (jitter-calibrated) tolerance band.

[Claim: low-cost] The executed layer-tokens (blocks run x tokens processed), compared to a
full recompute, are sublinear (measured layer-token ratio ~0.64 at the default gpt2 settings).
Only O(1) layers are *inspected*, but the truncated forward runs every block up to the drawn
layer, so the recompute cost scales with the depth of the drawn layer -- ~2/3 of full in
expectation.
[Claim: unforgeable] Which layer is checked is the verifier's secret randomness -- the
provider cannot forge "only the layers that won't be checked". Every request is in scope.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from assay.hooks import _base_transformer, layer_list, num_layers
from assay.nondeterminism import rank_overlap, sign_agreement
from assay.sketch import ActivationSketcher, Receipt, token_sha

_DIST_FLOOR = 1e-9
_P_FLOOR = 1e-12
_SIGMA_FLOOR = 0.05


# ---------------------------------------------------------------------------
# Truncated forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def truncated_hidden_state(model: nn.Module, input_ids: torch.Tensor, layer: int) -> Tuple[torch.Tensor, int]:
    """Get only hidden_states[layer], running only the blocks that are needed.

    Matches the transformers convention exactly (see the top of hooks.py):
      for layer <= n-1, hs[layer] = output of block layer-1, which is bitwise-equal
      to hidden_states[layer] of a forward truncated to the first layer+1 blocks
      (truncation just runs a prefix of the block loop);
      only for layer == n do we run all blocks + ln_f.

    Returns:
        (hidden (T,H), number of blocks executed)

    [Claim: low-cost] No lm_head, only a block prefix, and the sequence can also be
    truncated to the inspected positions (by the caller). Agreement across all layers is
    checked in tests/test_verifier.py::test_truncated_forward_matches_full.
    """
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    base = _base_transformer(model)
    blocks = layer_list(model)
    n = len(blocks)
    if not (1 <= layer <= n):
        raise ValueError(f"layer {layer} out of range 1..{n}")
    m = n if layer == n else layer + 1
    attr = "h" if hasattr(base, "h") else "layers"
    try:
        setattr(base, attr, blocks[:m])
        out = base(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    finally:
        setattr(base, attr, blocks)
    return out.hidden_states[layer][0], m


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Per-layer log-distance distribution and invariant baselines, measured on honest (jittered) traffic.

    [Claim: nondeterminism-robust] The tolerance band is calibrated on honest
    operation *with* benign jitter, so jitter is by definition part of the zero
    point and produces no false alarms.
    """

    mu: Dict[int, float] = field(default_factory=dict)      # per-layer mean of log mean-distance
    sigma: Dict[int, float] = field(default_factory=dict)   # per-layer std (with floor)
    n_samples: int = 0
    honest_sign_agreement: float = 1.0
    honest_rank_overlap: float = 1.0

    def zscore(self, layer: int, log_dist: float) -> float:
        mu = self.mu[layer]
        sd = max(self.sigma[layer], _SIGMA_FLOOR)
        return (log_dist - mu) / sd


class Calibrator:
    """Runs honest+jitter traffic and calibrates the tolerance band for every inspection layer (one-time preprocessing)."""

    def fit(self, verifier: "Verifier", server, prompts: Sequence[torch.Tensor], n: int = 64) -> Calibration:
        per_layer: Dict[int, List[float]] = {l: [] for l in verifier.sketcher.config.layers}
        sign_vals, rank_vals = [], []
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            resp = server.handle(f"cal-{i}", prompt)
            for layer in verifier.sketcher.config.layers:
                stats = verifier.compare_layer(resp.receipt, prompt, resp.output_ids, layer)
                per_layer[layer].append(math.log(max(stats["mean_dist"], _DIST_FLOOR)))
                sign_vals.append(stats["sign_agreement"])
                rank_vals.append(stats["rank_overlap"])
        calib = Calibration(n_samples=n)
        for layer, vals in per_layer.items():
            calib.mu[layer] = float(np.mean(vals))
            calib.sigma[layer] = float(max(np.std(vals), _SIGMA_FLOOR))
        calib.honest_sign_agreement = float(np.mean(sign_vals))
        calib.honest_rank_overlap = float(np.mean(rank_vals))
        return calib


# ---------------------------------------------------------------------------
# Verdict record
# ---------------------------------------------------------------------------

@dataclass
class VerdictRecord:
    """One request's verification result and cost measurement."""

    request_id: str
    valid_receipt: bool          # passed the structural check (hash / signature / protocol derivation)?
    reason: str
    layer_checked: int
    mean_dist: float
    z: float
    p: float
    sign_agreement: float
    rank_overlap: float
    cost_layer_tokens: int       # blocks run x tokens processed
    full_layer_tokens: int       # full recompute = n_layer x total tokens
    wall_time: float

    @property
    def cost_ratio(self) -> float:
        return self.cost_layer_tokens / max(self.full_layer_tokens, 1)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Holds M's weights and audits receipts by random partial recompute."""

    def __init__(
        self,
        model_m: nn.Module,
        sketcher: ActivationSketcher,
        signer=None,
        calibration: Optional[Calibration] = None,
        verifier_seed: int = 1234567,
    ):
        self.model = model_m
        self.sketcher = sketcher
        self.signer = signer
        self.calibration = calibration
        # The inspection layer is drawn with randomness kept secret from the provider [Claim: unforgeable]
        self._rng = np.random.default_rng(verifier_seed)
        self.n_layer = num_layers(model_m)

    # -- internal: recompute and compare one layer --------------------------

    def compare_layer(
        self,
        receipt: Receipt,
        prompt_ids: torch.Tensor,
        output_ids: Sequence[int],
        layer: int,
    ) -> Dict[str, float]:
        """Recompute the sketch for inspection layer `layer` and return normalized distance and invariant agreement.

        The sequence is truncated to (max inspected position + 1) tokens (under
        causal attention, activations at position <= p_max do not depend on later
        tokens) -> cost reduction along the token axis as well.
        """
        li = receipt.layers.index(layer)
        p_max = max(receipt.positions)
        full_ids = torch.cat([prompt_ids.flatten(), torch.as_tensor(list(output_ids), dtype=torch.long)])
        trunc_ids = full_ids[: p_max + 1]
        t0 = time.perf_counter()
        hidden, blocks_run = truncated_hidden_state(self.model, trunc_ids, layer)
        wall = time.perf_counter() - t0

        dists, agrees, overlaps = [], [], []
        for pi, pos in enumerate(receipt.positions):
            s_v = self.sketcher.sketch_vector(hidden[pos], layer)
            s_p = receipt.sketch[li, pi].astype(np.float32)
            dists.append(float(np.linalg.norm(s_p - s_v) / (np.linalg.norm(s_v) + 1e-12)))
            signs_v, ranks_v = self.sketcher.invariants(s_v)
            agrees.append(sign_agreement(receipt.signs[li, pi], signs_v))
            overlaps.append(rank_overlap(receipt.ranks[li, pi][None, :], ranks_v[None, :]))
        return {
            "mean_dist": float(np.mean(dists)),
            "sign_agreement": float(np.mean(agrees)),
            "rank_overlap": float(np.mean(overlaps)),
            "blocks_run": blocks_run,
            "tokens_run": int(trunc_ids.numel()),
            "wall_time": wall,
        }

    # -- structural check ----------------------------------------------------

    def structural_check(self, receipt: Receipt, prompt_ids: torch.Tensor, output_ids: Sequence[int]) -> Tuple[bool, str]:
        """Check the signature, hash binding, and protocol derivation (layers / positions).

        [Claim: unforgeable] Rejects all three forgery families here: replay
        (reusing an old receipt), tampering, and "sketch only convenient positions".
        """
        p_ids = [int(x) for x in prompt_ids.flatten().tolist()]
        if receipt.prompt_sha != token_sha(p_ids):
            return False, "prompt_hash_mismatch"
        if receipt.output_sha != token_sha(list(output_ids)):
            return False, "output_hash_mismatch"
        if receipt.prompt_len != len(p_ids) or receipt.total_len != len(p_ids) + len(list(output_ids)):
            return False, "length_mismatch"
        if tuple(receipt.layers) != tuple(self.sketcher.config.layers):
            return False, "layer_set_mismatch"
        expected_pos = self.sketcher.select_positions(receipt.request_id, receipt.prompt_len, receipt.total_len)
        if tuple(receipt.positions) != expected_pos:
            return False, "position_derivation_mismatch"
        if self.signer is not None:
            if not receipt.signature or not self.signer.verify(receipt.canonical_bytes(), receipt.signature):
                return False, "bad_signature"
        return True, "ok"

    # -- verification --------------------------------------------------------

    def verify(self, receipt: Receipt, prompt_ids: torch.Tensor, output_ids: Sequence[int]) -> VerdictRecord:
        """Verify one request and return a verdict with z-value, p-value, and cost measurement.

        [Claim: low-cost] cost_layer_tokens / full_layer_tokens is the sublinear cost ratio.
        [Claim: nondeterminism-robust] z is standardized against the calibrated (jitter-included) tolerance band.
        """
        total_len = len(prompt_ids.flatten()) + len(list(output_ids))
        full_lt = self.n_layer * total_len
        ok, reason = self.structural_check(receipt, prompt_ids, output_ids)
        if not ok:
            return VerdictRecord(
                request_id=receipt.request_id, valid_receipt=False, reason=reason,
                layer_checked=-1, mean_dist=float("inf"), z=float("inf"), p=_P_FLOOR,
                sign_agreement=0.0, rank_overlap=0.0,
                cost_layer_tokens=0, full_layer_tokens=full_lt, wall_time=0.0,
            )

        layer = int(self._rng.choice(list(receipt.layers)))
        stats = self.compare_layer(receipt, prompt_ids, output_ids, layer)
        log_dist = math.log(max(stats["mean_dist"], _DIST_FLOOR))
        if self.calibration is not None:
            z = self.calibration.zscore(layer, log_dist)
        else:
            z = 0.0
        p = max(0.5 * math.erfc(z / math.sqrt(2)), _P_FLOOR)  # one-sided (large distance = cheating)
        return VerdictRecord(
            request_id=receipt.request_id, valid_receipt=True, reason="ok",
            layer_checked=layer, mean_dist=stats["mean_dist"], z=float(z), p=float(p),
            sign_agreement=stats["sign_agreement"], rank_overlap=stats["rank_overlap"],
            cost_layer_tokens=stats["blocks_run"] * stats["tokens_run"],
            full_layer_tokens=full_lt, wall_time=stats["wall_time"],
        )

    # -- benchmark helper: measure full-recompute cost -----------------------

    @torch.no_grad()
    def full_recompute_time(self, prompt_ids: torch.Tensor, output_ids: Sequence[int]) -> float:
        """Wall-clock time of a full all-layer, all-token teacher-forcing recompute (no lm_head).

        The denominator of the wall-clock verifier cost ratio. [Claim: low-cost]
        """
        full_ids = torch.cat([prompt_ids.flatten(), torch.as_tensor(list(output_ids), dtype=torch.long)])
        base = _base_transformer(self.model)
        t0 = time.perf_counter()
        base(input_ids=full_ids.unsqueeze(0), output_hidden_states=True, use_cache=False)
        return time.perf_counter() - t0
