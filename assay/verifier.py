"""Verifier: with M's weights, recompute only a slice of each receipt and check it.

Verification flow (one request):
  1. Structural check: prompt/output hashes, PRF-derived positions, layer set, signature,
     and (when supplied) the request id this receipt is being presented for.
     -> A receipt that fails here is immediately "cheating" (tamper / replay / protocol deviation).
  2. Draw one inspection layer, and whether to run the deep check, using the verifier's
     secret randomness (both unpredictable to the provider).
  3. Truncated forward: recompute only the blocks needed for that layer, over only the
     tokens up to the maximum inspected position (removing later tokens is a no-op under
     causal attention).
  4. Recompute the sketch with the same JL projection and z-score it against the
     calibrated (jitter-calibrated) tolerance band.
  5. On the deep-check share: run the full depth once and ask whether M's own next-token
     argmax matches what the provider actually returned (generation_check). This is the
     only step that distinguishes "M was run over this text" from "M produced this text".

[Claim: low-cost] The executed layer-tokens (blocks run x tokens processed) are a *fraction*
of a full recompute at the default gpt2 settings. Be precise about what this is and is not:
only O(1) layers and O(1) positions are *inspected*, but the truncated forward still runs
every block up to the drawn layer, so the recompute cost scales with that layer's depth --
a constant-factor saving, not an asymptotic one. Two things are deliberately not counted as
savings: lm_head is excluded from the numerator *and* the denominator, so skipping it earns
nothing in this ratio; and a deep-checked request pays a full-depth pass, which is why that
pass is shared between both checks rather than run twice.
[Claim: unevadable] Which layer is checked is the verifier's secret randomness -- the
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
from assay.sketch import ActivationSketcher, Receipt, opaque_request_id, token_sha

_DIST_FLOOR = 1e-9
_P_FLOOR = 1e-12
_SIGMA_FLOOR = 0.05
# Floor on the honest next-token mismatch rate. Calibration usually measures exactly zero
# mismatches, and a literal q=0 would turn a single benign jitter-induced tie-flip into
# p=0 -- an instant false alarm. Add-one smoothing in Calibrator keeps it off zero; this
# is the backstop when no calibration is supplied at all.
_GEN_RATE_FLOOR = 0.01


def _binom_sf(k: int, n: int, q: float) -> float:
    """P(Binomial(n, q) >= k) -- exact upper tail, used for the generation-consistency p-value."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    q = min(max(q, 1e-6), 0.5)
    tail = sum(math.comb(n, i) * q**i * (1.0 - q) ** (n - i) for i in range(k, n + 1))
    return float(min(max(tail, _P_FLOOR), 1.0))


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
    hidden_states, blocks_run = truncated_hidden_states(model, input_ids, layer)
    return hidden_states[layer], blocks_run


@torch.no_grad()
def truncated_hidden_states(
    model: nn.Module, input_ids: torch.Tensor, upto_layer: int
) -> Tuple[Tuple[torch.Tensor, ...], int]:
    """Run the block prefix needed for `upto_layer` and return *all* hidden_states from it.

    [Claim: low-cost] Returning the whole tuple lets one pass serve two checks. The
    generation-consistency check needs the final layer, and the sketch check needs the drawn
    layer; done as two calls that is two forwards, and a single request could then cost more
    than the full recompute it is supposed to be cheaper than (measured 1.75x before this
    was shared).
    """
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    base = _base_transformer(model)
    blocks = layer_list(model)
    n = len(blocks)
    if not (1 <= upto_layer <= n):
        raise ValueError(f"layer {upto_layer} out of range 1..{n}")
    m = n if upto_layer == n else upto_layer + 1
    attr = "h" if hasattr(base, "h") else "layers"
    try:
        setattr(base, attr, blocks[:m])
        out = base(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    finally:
        setattr(base, attr, blocks)
    return tuple(h[0] for h in out.hidden_states), m


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
    # Rate at which an honest greedy provider's emitted token differs from M's own argmax
    # at an inspected position (jitter can flip a near-tie). Null rate of generation_check.
    gen_mismatch_rate: float = _GEN_RATE_FLOOR

    def zscore(self, layer: int, log_dist: float) -> float:
        """Standardize a log-distance against this layer's honest band.

        [Claim: nondeterminism-robust] mu/sigma come from honest traffic *with* jitter, so
        benign numerical noise sits at z ~ 0 by construction. The sigma floor keeps a layer
        whose honest spread is tiny from turning that noise into large z values.
        """
        mu = self.mu[layer]
        sd = max(self.sigma[layer], _SIGMA_FLOOR)
        return (log_dist - mu) / sd


class Calibrator:
    """Runs honest+jitter traffic and calibrates the tolerance band for every inspection layer (one-time preprocessing)."""

    def fit(self, verifier: "Verifier", server, prompts: Sequence[torch.Tensor], n: int = 64,
            id_seed: int = 0) -> Calibration:
        """Measure the honest log-distance band per layer by running honest traffic with jitter.

        [Claim: nondeterminism-robust] This is where benign nondeterminism is absorbed: the
        band is fit on traffic that already contains it. Fit this on prompts that are
        disjoint from the ones later scored, or the false-positive rate comes out optimistic
        (benchmarks/run.py splits the pool for exactly this reason).

        [Claim: unevadable] Calibration traffic goes through the provider being audited, so
        its request ids must be indistinguishable from ordinary ones. They used to be
        literally `cal-0`, `cal-1`, ...: a provider that answered those honestly and cheated
        everywhere else would have shifted the whole tolerance band to its liking -- the
        canary problem, reintroduced through the calibration channel. Ids are now opaque
        digests drawn from the same shape as production ids.
        """
        per_layer: Dict[int, List[float]] = {l: [] for l in verifier.sketcher.config.layers}
        sign_vals, rank_vals = [], []
        gen_checked = gen_mismatch = 0
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            resp = server.handle(opaque_request_id("calibration", id_seed, i), prompt)
            for layer in verifier.sketcher.config.layers:
                stats = verifier.compare_layer(resp.receipt, prompt, resp.output_ids, layer)
                per_layer[layer].append(math.log(max(stats["mean_dist"], _DIST_FLOOR)))
                sign_vals.append(stats["sign_agreement"])
                rank_vals.append(stats["rank_overlap"])
            gen = verifier.generation_check(resp.receipt, prompt, resp.output_ids)
            gen_checked += gen["n_checked"]
            gen_mismatch += gen["n_mismatch"]
        calib = Calibration(n_samples=n)
        for layer, vals in per_layer.items():
            calib.mu[layer] = float(np.mean(vals))
            calib.sigma[layer] = float(max(np.std(vals), _SIGMA_FLOOR))
        # Add-one smoothing: honest traffic usually shows zero mismatches, and a literal
        # zero null rate would make one benign tie-flip look like proof of cheating.
        calib.gen_mismatch_rate = max((gen_mismatch + 1) / (gen_checked + 2), _GEN_RATE_FLOOR)
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
    deep_checked: bool = False   # did the generation-consistency check run on this request?
    gen_checked: int = 0         # inspected positions that had a successor token
    gen_mismatch: int = 0        # of those, how many disagreed with M's own argmax

    @property
    def cost_ratio(self) -> float:
        """[Claim: low-cost] Executed layer-tokens divided by a full recompute's (< 1)."""
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
        deep_check_rate: float = 0.25,
    ):
        self.model = model_m
        self.sketcher = sketcher
        self.signer = signer
        self.calibration = calibration
        # The inspection layer is drawn with randomness kept secret from the provider [Claim: unevadable]
        self._rng = np.random.default_rng(verifier_seed)
        self.n_layer = num_layers(model_m)
        self.deep_check_rate = float(deep_check_rate)

    # -- internal: generation-consistency check -----------------------------

    @torch.no_grad()
    def generation_check(
        self,
        receipt: Receipt,
        prompt_ids: torch.Tensor,
        output_ids: Sequence[int],
    ) -> Dict[str, float]:
        """Check that M itself would have emitted the response tokens at the inspected positions.

        [Claim: unevadable] The sketch check alone proves only that *some* run of M passed
        over this exact token sequence -- not that M produced it. That gap is a real attack,
        not a hypothetical: generate cheaply with M', then push (prompt + M' output) through
        M once with teacher forcing and read the sketch off that pass. The receipt is then
        genuinely M's, agrees perfectly, and the response the user received is still M''s.
        `ProviderServer(mode="forge")` implements exactly that, and it defeats every other
        check in this file.

        What closes it is that a teacher-forced pass reveals what M *would* have said. At an
        inspected position p, M's own next-token argmax is compared against the token the
        provider actually returned at p+1. For an honest greedy provider these agree except
        where jitter flips a near-tie (calibrated as `gen_mismatch_rate`); for M'-generated
        text they diverge on a large fraction of positions.

        This is the one check that needs the full stack -- next-token logits come from the
        final layer through lm_head -- so it runs on a random `deep_check_rate` share of
        requests rather than all of them, and the provider cannot tell which. Sampling
        rather than greedy decoding would need a log-probability statistic here instead of
        an argmax match.
        """
        full_ids = torch.cat([prompt_ids.flatten(), torch.as_tensor(list(output_ids), dtype=torch.long)])
        total = int(full_ids.numel())
        # Only positions that have a successor token in this response predict anything.
        checkable = [p for p in receipt.positions if p + 1 < total]
        if not checkable:
            return {"n_checked": 0, "n_mismatch": 0, "blocks_run": 0, "tokens_run": 0, "wall_time": 0.0}
        # The sketch check needs the *same* token span, so truncate to the widest of the two.
        trunc_ids = full_ids[: max(max(checkable), max(receipt.positions)) + 1]
        head = self.model.get_output_embeddings()
        t0 = time.perf_counter()
        hidden_states, blocks_run = truncated_hidden_states(self.model, trunc_ids, self.n_layer)
        final = hidden_states[self.n_layer]
        mism = 0
        for p in checkable:
            predicted = int(torch.argmax(head(final[p])))
            if predicted != int(full_ids[p + 1]):
                mism += 1
        wall = time.perf_counter() - t0
        return {
            "n_checked": len(checkable),
            "n_mismatch": mism,
            "blocks_run": blocks_run,
            "tokens_run": int(trunc_ids.numel()),
            "wall_time": wall,
            # Handed back so verify() can reuse this pass for the sketch comparison.
            "hidden_at_layer": hidden_states,
        }

    # -- internal: recompute and compare one layer --------------------------

    def compare_layer(
        self,
        receipt: Receipt,
        prompt_ids: torch.Tensor,
        output_ids: Sequence[int],
        layer: int,
        hidden: Optional[torch.Tensor] = None,
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
        if hidden is None:
            t0 = time.perf_counter()
            hidden, blocks_run = truncated_hidden_state(self.model, trunc_ids, layer)
            wall = time.perf_counter() - t0
        else:
            # Reusing a pass the caller already paid for; charge it there, not here.
            blocks_run, wall = 0, 0.0

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

    def structural_check(
        self,
        receipt: Receipt,
        prompt_ids: torch.Tensor,
        output_ids: Sequence[int],
        expected_request_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check the signature, hash binding, and protocol derivation (layers / positions).

        [Claim: unevadable] Rejects all three forgery families here: replay
        (reusing an old receipt), tampering, and "sketch only convenient positions".

        Pass `expected_request_id` (the id the client actually saw) to also bind the
        receipt to *this* request. Without it, a receipt stays valid for any request
        with a byte-identical (prompt, output) pair, so one attested response could
        cover many served ones -- which would poke a hole in the 100%-coverage claim.
        """
        if expected_request_id is not None and receipt.request_id != expected_request_id:
            return False, "request_id_mismatch"
        p_ids = [int(x) for x in prompt_ids.flatten().tolist()]
        if receipt.prompt_sha != token_sha(p_ids):
            return False, "prompt_hash_mismatch"
        if receipt.output_sha != token_sha(list(output_ids)):
            return False, "output_hash_mismatch"
        if receipt.prompt_len != len(p_ids) or receipt.total_len != len(p_ids) + len(list(output_ids)):
            return False, "length_mismatch"
        if tuple(receipt.layers) != tuple(self.sketcher.config.layers):
            return False, "layer_set_mismatch"
        if not receipt.positions:
            # A receipt with no inspected positions attests to nothing, and reaching
            # compare_layer with it raises from max(). Reject it as malformed rather than
            # letting a hand-crafted receipt crash the audit loop.
            return False, "no_positions"
        expected_pos = self.sketcher.select_positions(receipt.request_id, receipt.prompt_len, receipt.total_len)
        if tuple(receipt.positions) != expected_pos:
            return False, "position_derivation_mismatch"
        if self.signer is not None:
            if not receipt.signature or not self.signer.verify(receipt.canonical_bytes(), receipt.signature):
                return False, "bad_signature"
        return True, "ok"

    # -- verification --------------------------------------------------------

    def verify(
        self,
        receipt: Receipt,
        prompt_ids: torch.Tensor,
        output_ids: Sequence[int],
        expected_request_id: Optional[str] = None,
    ) -> VerdictRecord:
        """Verify one request and return a verdict with z-value, p-value, and cost measurement.

        [Claim: low-cost] cost_layer_tokens / full_layer_tokens is the cost ratio against a
        full recompute (< 1 by construction; ~0.74 measured on the default gpt2 settings).
        [Claim: nondeterminism-robust] z is standardized against the calibrated (jitter-included) tolerance band.
        [Claim: unevadable] `expected_request_id`, when supplied, binds the receipt to this
        exact request so one receipt cannot cover several served responses. On a random
        `deep_check_rate` share of requests the generation-consistency check also runs,
        which is what stops a provider from serving M' text under a genuine M receipt.
        """
        total_len = len(prompt_ids.flatten()) + len(list(output_ids))
        full_lt = self.n_layer * total_len
        ok, reason = self.structural_check(receipt, prompt_ids, output_ids, expected_request_id)
        if not ok:
            return VerdictRecord(
                request_id=receipt.request_id, valid_receipt=False, reason=reason,
                layer_checked=-1, mean_dist=float("inf"), z=float("inf"), p=_P_FLOOR,
                sign_agreement=0.0, rank_overlap=0.0,
                cost_layer_tokens=0, full_layer_tokens=full_lt, wall_time=0.0,
            )

        # Both draws come from the verifier's secret RNG, before any data is looked at.
        layer = int(self._rng.choice(list(receipt.layers)))
        deep = bool(self._rng.random() < self.deep_check_rate)

        gen = {"n_checked": 0, "n_mismatch": 0, "blocks_run": 0, "tokens_run": 0, "wall_time": 0.0}
        shared_hidden = None
        if deep:
            # One full-depth pass serves both checks: the drawn layer for the sketch, the
            # final layer for next-token logits. Charged once, here.
            gen = self.generation_check(receipt, prompt_ids, output_ids)
            shared_hidden = gen.pop("hidden_at_layer", None)
        stats = self.compare_layer(receipt, prompt_ids, output_ids, layer,
                                   hidden=shared_hidden[layer] if shared_hidden is not None else None)
        log_dist = math.log(max(stats["mean_dist"], _DIST_FLOOR))
        z = self.calibration.zscore(layer, log_dist) if self.calibration is not None else 0.0
        p_sketch = max(0.5 * math.erfc(z / math.sqrt(2)), _P_FLOOR)  # one-sided: large distance = cheating

        p = p_sketch
        cost_lt = stats["blocks_run"] * stats["tokens_run"] + gen["blocks_run"] * gen["tokens_run"]
        wall = stats["wall_time"] + gen["wall_time"]
        if deep:
            q_gen = self.calibration.gen_mismatch_rate if self.calibration is not None else _GEN_RATE_FLOOR
            p_gen = _binom_sf(gen["n_mismatch"], gen["n_checked"], q_gen)
            # Bonferroni over the two tests that ran. The deep draw is data-independent,
            # so combining this way keeps the per-request p-value valid under H0.
            p = max(min(1.0, 2.0 * min(p_sketch, p_gen)), _P_FLOOR)
        return VerdictRecord(
            request_id=receipt.request_id, valid_receipt=True, reason="ok",
            layer_checked=layer, mean_dist=stats["mean_dist"], z=float(z), p=float(p),
            sign_agreement=stats["sign_agreement"], rank_overlap=stats["rank_overlap"],
            cost_layer_tokens=cost_lt, full_layer_tokens=full_lt, wall_time=wall,
            deep_checked=deep, gen_checked=int(gen["n_checked"]), gen_mismatch=int(gen["n_mismatch"]),
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
