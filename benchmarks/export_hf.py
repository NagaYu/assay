"""Build the Hugging Face bundle: a receipt corpus (dataset) and a reproducibility bundle (model).

The point of publishing receipts rather than only summary numbers is that a receipt is
checkable. Every row carries the prompt and output token ids, the signed sketch, and the
verdict Assay reached -- so anyone with the gpt2 weights can recompute the sketch and get
the same answer, or disagree with it. Aggregate tables cannot be argued with; corpora can.

Outputs (under --outdir):
  dataset/receipts.jsonl   one row per request, all scenarios, with the receipt inline
  dataset/protocol.json    sketch protocol, calibration band, and the demo signing key
  model/tiny_pair/{M,Mp}   the seeded hermetic pair the tests and the Space run on
  model/calibration.json   the gpt2 tolerance band, so the demo needs no calibration pass

Usage:
  python benchmarks/export_hf.py --pair gpt2 --n 60 --trials 2 --outdir /tmp/assay-hf
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from assay.hooks import tiny_random_pair
from assay.nondeterminism import NondeterminismModel
from assay.sequential_audit import EValueAudit
from assay.server import ProviderServer
from assay.verifier import Verifier
from benchmarks.run import SCENARIO_DEFS, setup

_DEMO_KEY = "assay-demo-key-0123456789abcdef!"


def _b64(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def receipt_row(scenario: str, trial: int, t: int, resp, verdict, prompt_ids, log_e: float) -> Dict:
    """One dataset row: the receipt as issued, the verdict as reached, and the ground truth.

    `served_by` is the label a real auditor never gets to see; it is here so the corpus can
    be used to score detectors, including detectors that are not this one.
    """
    r = resp.receipt
    return {
        "scenario": scenario,
        "trial": trial,
        "step": t,
        "request_id": r.request_id,
        "served_by": resp.model_used,          # ground truth: "M" or "M'"
        "is_substituted": resp.model_used != "M",
        "prompt_ids": [int(x) for x in prompt_ids.flatten().tolist()],
        "output_ids": [int(x) for x in resp.output_ids],
        "prompt_len": r.prompt_len,
        "total_len": r.total_len,
        "layers": list(r.layers),
        "positions": list(r.positions),
        "sketch_f16_b64": _b64(r.sketch, np.float16),
        "signs_u8_b64": _b64(r.signs, np.uint8),
        "ranks_i16_b64": _b64(r.ranks, np.int16),
        "sig_alg": r.sig_alg,
        "signature": r.signature,
        "verdict_valid_receipt": verdict.valid_receipt,
        "verdict_reason": verdict.reason,
        "layer_checked": verdict.layer_checked,
        "mean_dist": float(verdict.mean_dist),
        "z": float(verdict.z),
        "p": float(verdict.p),
        "sign_agreement": float(verdict.sign_agreement),
        "rank_overlap": float(verdict.rank_overlap),
        "deep_checked": bool(verdict.deep_checked),
        "gen_checked": int(verdict.gen_checked),
        "gen_mismatch": int(verdict.gen_mismatch),
        "cost_ratio": float(verdict.cost_ratio),
        "audit_log_e": float(log_e),
    }


def export(args) -> Dict:
    ctx = setup(args)
    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for scenario in ["honest", "size", "quant", "random", "evasive", "forge"]:
        d = SCENARIO_DEFS[scenario]
        mp = ctx["models"].get(d["mp"]) if d["mp"] else None
        alarms = []
        for trial in range(args.trials):
            jitter = NondeterminismModel(ctx["sigma"], seed=5000 + trial)
            server = ProviderServer(ctx["M"], mp, ctx["sketcher"], mode=d["mode"],
                                    p_honest=ctx["p_honest"], canaries=ctx["canaries"],
                                    jitter=jitter, gen_len=ctx["gen_len"], seed=trial * 31 + 5)
            verifier = Verifier(ctx["M"], ctx["sketcher"], ctx["signer"], ctx["calib"],
                                verifier_seed=trial * 77 + 3,
                                deep_check_rate=ctx["deep_check_rate"])
            audit = EValueAudit(alpha=ctx["alpha"])
            rng = np.random.default_rng(trial * 13 + 1)
            order = rng.permutation(len(ctx["prompts"]))
            for t in range(args.n):
                prompt = ctx["prompts"][order[t % len(order)]]
                rid = f"{scenario}-{trial}-{t}"
                resp = server.handle(rid, prompt)
                verdict = verifier.verify(resp.receipt, prompt, resp.output_ids,
                                          expected_request_id=rid)
                audit.update(verdict.p)
                rows.append(receipt_row(scenario, trial, t, resp, verdict, prompt, audit.log_e))
            alarms.append(audit.t_alarm)
            print(f"[export] {scenario} trial {trial}: alarm at {audit.t_alarm}", flush=True)
        summary[scenario] = {
            "alarm_steps": alarms,
            "n_rows": sum(1 for r in rows if r["scenario"] == scenario),
        }

    os.makedirs(os.path.join(args.outdir, "dataset"), exist_ok=True)
    receipts_path = os.path.join(args.outdir, "dataset", "receipts.jsonl")
    with open(receipts_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cfg = ctx["sketcher"].config
    calib = ctx["calib"]
    protocol = {
        "sketch": {
            "seed": cfg.seed, "layers": list(cfg.layers), "n_positions": cfg.n_positions,
            "sketch_dim": cfg.sketch_dim, "rank_top_m": cfg.rank_top_m,
        },
        "calibration": {
            "mu": {str(k): v for k, v in calib.mu.items()},
            "sigma": {str(k): v for k, v in calib.sigma.items()},
            "n_samples": calib.n_samples,
            "gen_mismatch_rate": calib.gen_mismatch_rate,
        },
        "verifier": {"deep_check_rate": ctx["deep_check_rate"], "alpha": ctx["alpha"]},
        "generation": {"gen_len": ctx["gen_len"], "decoding": "greedy", "jitter_sigma_rel": ctx["sigma"]},
        "model_m": args.model_m if args.pair == "gpt2" else "assay tiny_random_pair(seed=0)",
        # Published on purpose: without it the signatures in the corpus cannot be checked.
        # It is a demo key for a research prototype, not a secret and not a TEE key.
        "hmac_demo_key": _DEMO_KEY,
        "signature_note": "HMAC-SHA256 over Receipt.canonical_bytes(); simulates a TEE attestation",
    }
    with open(os.path.join(args.outdir, "dataset", "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)

    # Model bundle: the hermetic pair (so the Space and the tests need no download of ours)
    # plus the gpt2 tolerance band (so a demo can skip the calibration pass).
    model_dir = os.path.join(args.outdir, "model")
    os.makedirs(model_dir, exist_ok=True)
    M_tiny, Mp_tiny = tiny_random_pair(seed=0)
    M_tiny.save_pretrained(os.path.join(model_dir, "tiny_pair", "M"))
    Mp_tiny.save_pretrained(os.path.join(model_dir, "tiny_pair", "Mp"))
    with open(os.path.join(model_dir, "calibration.json"), "w") as f:
        json.dump(protocol, f, indent=2)

    stats = {
        "n_rows": len(rows),
        "n_substituted": sum(r["is_substituted"] for r in rows),
        "scenarios": summary,
        "config": {k: getattr(args, k) for k in ("pair", "n", "trials", "gen_len", "alpha",
                                                 "sigma", "p_honest", "n_cal", "deep_check_rate")},
    }
    with open(os.path.join(args.outdir, "dataset", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[export] {len(rows)} receipts -> {receipts_path}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", choices=["gpt2", "tiny"], default="gpt2")
    ap.add_argument("--model-m", default="gpt2")
    ap.add_argument("--model-mp", default="distilgpt2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--n-cal", type=int, default=48)
    ap.add_argument("--gen-len", type=int, default=12)
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=3e-4)
    ap.add_argument("--audit-rate", type=float, default=0.1)
    ap.add_argument("--p-honest", type=float, default=0.9)
    ap.add_argument("--deep-check-rate", type=float, default=0.25)
    ap.add_argument("--n-bb", type=int, default=400)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "hf_export"))
    args = ap.parse_args()
    torch.manual_seed(0)
    stats = export(args)
    print(json.dumps({k: v for k, v in stats.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
