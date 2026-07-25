"""Shared fixtures: verify every claim on a network-free, seed-fixed tiny model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from assay.hooks import fake_quantize_int8_, tiny_random_pair
from assay.nondeterminism import NondeterminismModel
from assay.server import ProviderServer
from assay.sketch import ActivationSketcher, HmacSigner, SketchConfig
from assay.verifier import Calibrator, Verifier


VOCAB = 256
SIGMA = 3e-4


@pytest.fixture(scope="session")
def models():
    M, Mp_size = tiny_random_pair(seed=0)
    Mp_quant, _ = tiny_random_pair(seed=0)  # quantize from weights identical to M
    fake_quantize_int8_(Mp_quant)
    return {"M": M, "size": Mp_size, "quant": Mp_quant}


@pytest.fixture(scope="session")
def sketcher():
    cfg = SketchConfig(layers=(1, 2, 3), n_positions=4, sketch_dim=16, rank_top_m=6)
    return ActivationSketcher(cfg, HmacSigner(b"test-key-0123456789abcdef-32byte"))


@pytest.fixture(scope="session")
def prompts():
    rng = np.random.default_rng(42)
    return [torch.tensor(rng.integers(0, VOCAB, size=8), dtype=torch.long) for _ in range(24)]


@pytest.fixture(scope="session")
def calibrated_verifier(models, sketcher, prompts):
    """A Verifier calibrated on the honest server + jitter (shared across tests)."""
    jitter = NondeterminismModel(SIGMA, seed=1)
    server = ProviderServer(models["M"], None, sketcher, mode="honest", jitter=jitter, gen_len=8, seed=0)
    verifier = Verifier(models["M"], sketcher, sketcher.signer)
    calib = Calibrator().fit(verifier, server, prompts, n=16)
    verifier.calibration = calib
    return verifier


def make_server(models, sketcher, mode, mp_key="size", canaries=None, seed=0, jitter_seed=2):
    jitter = NondeterminismModel(SIGMA, seed=jitter_seed)
    return ProviderServer(
        models["M"], models.get(mp_key), sketcher, mode=mode,
        p_honest=0.9, canaries=canaries, jitter=jitter, gen_len=8, seed=seed,
    )
