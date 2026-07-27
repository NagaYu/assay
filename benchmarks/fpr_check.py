"""Measure Assay's *actual* sequential false-alarm rate under honest operation.

Usage: python benchmarks/fpr_check.py [reps] [horizon] [n_cal]   (~2 min for the defaults)


The alpha guarantee is stated for exact null p-values. Here mu/sigma are estimated from a
finite calibration sample and then plugged in, and every request in a run shares that one
estimate -- so calibration error is a common-mode shift, not independent noise. If mu_hat
lands low, every z in the run is shifted up together and the e-process can drift across the
threshold with no cheating anywhere. This measures how much that costs, with an independent
calibration per replicate (which is the honest way to ask the question).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from assay.hooks import tiny_random_pair
from assay.nondeterminism import NondeterminismModel
from assay.sequential_audit import EValueAudit
from assay.server import ProviderServer
from assay.sketch import ActivationSketcher, HmacSigner, SketchConfig
from assay.verifier import Calibrator, Verifier

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 80
N_CAL = int(sys.argv[3]) if len(sys.argv) > 3 else 48
ALPHA = 0.01

M, _ = tiny_random_pair(seed=0)
cfg = SketchConfig(layers=(1, 2, 3), n_positions=6, sketch_dim=32, rank_top_m=8)
sk = ActivationSketcher(cfg, HmacSigner(b"k" * 32))
rng = np.random.default_rng(0)
pool = [torch.tensor(rng.integers(0, 256, size=10), dtype=torch.long) for _ in range(48)]
cal_prompts, eval_prompts = pool[0::2], pool[1::2]

alarms, max_log_e, zbar = 0, [], []
for rep in range(REPS):
    jitter = NondeterminismModel(3e-4, seed=1000 + rep)
    cal_srv = ProviderServer(M, None, sk, mode="honest", jitter=jitter, gen_len=10, seed=2000 + rep)
    calib = Calibrator().fit(Verifier(M, sk, sk.signer), cal_srv, cal_prompts, n=N_CAL, id_seed=rep)

    srv = ProviderServer(M, None, sk, mode="honest", jitter=jitter, gen_len=10, seed=3000 + rep)
    ver = Verifier(M, sk, sk.signer, calib, verifier_seed=4000 + rep, deep_check_rate=0.25)
    audit = EValueAudit(alpha=ALPHA)
    zs = []
    for t in range(HORIZON):
        prompt = eval_prompts[t % len(eval_prompts)]
        rid = f"sim-{rep}-{t}"
        resp = srv.handle(rid, prompt)
        v = ver.verify(resp.receipt, prompt, resp.output_ids, expected_request_id=rid)
        zs.append(v.z)
        audit.update(v.p)
    alarms += audit.alarmed
    max_log_e.append(max(audit.history))
    zbar.append(float(np.mean(zs)))
    print(f"rep {rep:3d}: alarmed={audit.alarmed} log_e_max={max(audit.history):7.2f} mean_z={np.mean(zs):+.2f}",
          flush=True)

rate = alarms / REPS
se = (rate * (1 - rate) / REPS) ** 0.5
print(f"\nsequential false-alarm rate: {rate:.3f} +/- {se:.3f}  (alpha={ALPHA}, "
      f"{REPS} reps x {HORIZON} honest requests, independent calibration each)")
print(f"mean z across reps: {np.mean(zbar):+.3f}  (spread across reps: {np.std(zbar):.3f})")
