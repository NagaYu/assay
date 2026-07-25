"""Thin hook layer for model loading and hidden_states capture.

Utilities to pull hidden_states from any layer of a transformers model, plus
helpers to construct the model pairs used in the substitution experiments
(size substitution / precision substitution).

hidden_states index convention (matching transformers):
    hs[0]       = embedding output (wte + wpe, post-dropout; identity in eval)
    hs[l]       = output of block l-1 (1 <= l <= n_layer-1, before final LayerNorm)
    hs[n_layer] = final block output with ln_f applied
This convention is matched exactly by the truncated forward in verifier.py
(bitwise agreement is checked in tests/test_verifier.py::test_truncated_forward_matches_full).
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# hidden_states capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_hidden_states(model: nn.Module, input_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Run (prompt+output) through the model once (teacher forcing) and return all hidden_states.

    [Claim: low-cost] In production the provider can reuse activations already
    computed during serving, so no extra forward pass is needed. This prototype
    recomputes a teacher-forcing pass for simplicity; that cost is deliberately
    excluded from the denominator of the verifier cost ratio.

    Returns:
        tuple of length n_layer+1, each element shape (T, H).
    """
    model.eval()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    base = _base_transformer(model)
    out = base(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return tuple(h[0] for h in out.hidden_states)


def _base_transformer(model: nn.Module) -> nn.Module:
    """Extract the backbone (without lm_head) from a CausalLM wrapper.

    [Claim: low-cost] Verification only compares hidden_states, so we never run
    lm_head (the largest matmul, vocab_size x hidden_size).
    """
    for attr in ("transformer", "model", "gpt_neox"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model  # a bare base model was passed


def layer_list(model: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer blocks, architecture-agnostically.

    Handles GPT-2 style (transformer.h), Llama style (model.layers), and
    GPT-NeoX style (gpt_neox.layers). Used by the verifier's truncated forward
    to count layers.
    """
    base = _base_transformer(model)
    for attr in ("h", "layers"):
        if hasattr(base, attr) and isinstance(getattr(base, attr), nn.ModuleList):
            return getattr(base, attr)
    raise ValueError(f"unsupported architecture: {type(model).__name__}")


def num_layers(model: nn.Module) -> int:
    return len(layer_list(model))


def hidden_size(model: nn.Module) -> int:
    return int(model.config.hidden_size)


# ---------------------------------------------------------------------------
# Model-pair loading
# ---------------------------------------------------------------------------

def load_model(name: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load a CausalLM from transformers (eval mode, no gradients)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def fake_quantize_int8_(model: nn.Module) -> nn.Module:
    """Quantize->dequantize weights to per-channel absmax INT8 (in-place). Builds the precision-substitution M'.

    Implements the same absmax symmetric quantization as bitsandbytes LLM.int8()
    as a CUDA-free, weight-only fake-quant (required by the CPU-first constraint).
    On CUDA it can be swapped for the real load_in_8bit=True model.

    [Claim: nondeterminism-robust (contrast)] Quantization produces a systematic
    activation shift orders of magnitude larger than benign jitter (~1e-4 relative),
    so it is caught by both the calibrated tolerance band and the invariant features.
    """
    targets = (nn.Linear,)
    try:
        from transformers.pytorch_utils import Conv1D

        targets = (nn.Linear, Conv1D)
    except ImportError:
        pass

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, targets):
                w = module.weight.data
                # Linear: (out, in) -> per row / Conv1D: (in, out) -> per column
                dim = 1 if isinstance(module, nn.Linear) else 0
                scale = w.abs().amax(dim=dim, keepdim=True).clamp_min(1e-12) / 127.0
                module.weight.data = torch.round(w / scale).clamp(-127, 127) * scale
    return model


def load_int8_model(name: str, device: str = "cpu") -> nn.Module:
    """Load the INT8 version of M'. Real INT8 with CUDA+bitsandbytes, else fake-quant."""
    if device.startswith("cuda"):
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig

            return AutoModelForCausalLM.from_pretrained(
                name, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map=device
            ).eval()
        except ImportError:
            pass
    return fake_quantize_int8_(load_model(name, device))


def tiny_random_pair(
    seed: int = 0,
    vocab_size: int = 256,
    hidden: int = 64,
    layers_m: int = 4,
    layers_mp: int = 2,
) -> Tuple[nn.Module, nn.Module]:
    """Build a seed-fixed randomly-initialized GPT-2 pair (M: deep, M': shallow) for tests.

    Network-free, runs in seconds, hermetic size-substitution pair. Hidden size
    and vocab are shared so the attacker can build a dimensionally-valid fake
    receipt from M''s activations (a strong setting -- and it is still detected).
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    def build(n_layer: int, tag: int) -> nn.Module:
        torch.manual_seed(seed * 1000 + tag)
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=128,
            n_embd=hidden,
            n_layer=n_layer,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        model = GPT2LMHeadModel(cfg)
        model.eval()
        model.requires_grad_(False)
        return model

    return build(layers_m, 1), build(layers_mp, 2)


# ---------------------------------------------------------------------------
# Deterministic PRF (protocol-shared randomness derivation)
# ---------------------------------------------------------------------------

def prf_int(*parts: object, nbytes: int = 8) -> int:
    """SHA-256-based deterministic PRF. Used for projection-matrix seeds and position selection.

    [Claim: unforgeable] The *token positions* that get sketched, and the
    projection-matrix seeds, are derived deterministically from the protocol
    (public seed + request_id), so the provider cannot pick "positions that
    won't be inspected" to cheat. (The set of *layers* is a fixed public protocol
    constant; which layer subset the verifier recomputes is chosen by the
    verifier's secret randomness.)
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:nbytes], "big")
