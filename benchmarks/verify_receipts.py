"""Re-verify a published receipt corpus against the real weights.

This is the part that makes the dataset worth publishing: the rows are not summary numbers
to be taken on trust, they are signed receipts that can be recomputed. Point this at
receipts.jsonl and it rebuilds each Receipt, checks the signature, re-runs the verifier
against M, and reports whether the published verdict is reproduced.

  python benchmarks/verify_receipts.py --receipts receipts.jsonl --protocol protocol.json \
      --model gpt2 --limit 50
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from assay.hooks import load_model, tiny_random_pair
from assay.sketch import ActivationSketcher, HmacSigner, Receipt, SketchConfig
from assay.verifier import Calibration, Verifier


def load_protocol(path: str):
    with open(path) as f:
        proto = json.load(f)
    sk = proto["sketch"]
    config = SketchConfig(seed=sk["seed"], layers=tuple(sk["layers"]), n_positions=sk["n_positions"],
                          sketch_dim=sk["sketch_dim"], rank_top_m=sk["rank_top_m"])
    cal = proto["calibration"]
    calibration = Calibration(
        mu={int(k): v for k, v in cal["mu"].items()},
        sigma={int(k): v for k, v in cal["sigma"].items()},
        n_samples=cal["n_samples"],
        gen_mismatch_rate=cal.get("gen_mismatch_rate", 0.01),
    )
    signer = HmacSigner(proto["hmac_demo_key"].encode())
    return proto, config, calibration, signer


def row_to_receipt(row: dict, config: SketchConfig) -> Receipt:
    n_layers, n_pos, k, m = len(row["layers"]), len(row["positions"]), config.sketch_dim, config.rank_top_m
    def arr(key, dtype, width):
        return np.frombuffer(base64.b64decode(row[key]), dtype=dtype).reshape(n_layers, n_pos, width)
    from assay.sketch import token_sha
    return Receipt(
        request_id=row["request_id"], model_claim="M",
        prompt_sha=token_sha(row["prompt_ids"]), output_sha=token_sha(row["output_ids"]),
        prompt_len=row["prompt_len"], total_len=row["total_len"],
        layers=tuple(row["layers"]), positions=tuple(row["positions"]),
        sketch=arr("sketch_f16_b64", np.float16, k),
        signs=arr("signs_u8_b64", np.uint8, k),
        ranks=arr("ranks_i16_b64", np.int16, m),
        seed=config.seed, sig_alg=row["sig_alg"], signature=row["signature"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--model", default="gpt2", help="'gpt2' etc., or 'tiny' for the seeded pair")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--deep-check-rate", type=float, default=0.0,
                    help="0 reproduces the sketch verdict deterministically; the published "
                         "deep checks depend on the verifier's own secret draw and are not "
                         "reproducible from the corpus alone")
    args = ap.parse_args()

    proto, config, calibration, signer = load_protocol(args.protocol)
    model = tiny_random_pair(seed=0)[0] if args.model == "tiny" else load_model(args.model)
    sketcher = ActivationSketcher(config, signer)
    verifier = Verifier(model, sketcher, signer, calibration, deep_check_rate=args.deep_check_rate)

    rows = []
    with open(args.receipts) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    sig_ok = struct_ok = 0
    z_err = []
    import math

    for row in rows:
        receipt = row_to_receipt(row, config)
        if signer.verify(receipt.canonical_bytes(), receipt.signature):
            sig_ok += 1
        prompt = torch.tensor(row["prompt_ids"], dtype=torch.long)
        ok, _ = verifier.structural_check(receipt, prompt, row["output_ids"],
                                          expected_request_id=row["request_id"])
        struct_ok += ok
        if not ok:
            continue
        # Recompute the *published* layer. Calling verify() instead would draw a layer from
        # this verifier's own secret RNG, which is the whole point of that draw -- it is not
        # meant to be reproducible, and a different layer gives a legitimately different z.
        stats = verifier.compare_layer(receipt, prompt, row["output_ids"], row["layer_checked"])
        z = calibration.zscore(row["layer_checked"], math.log(max(stats["mean_dist"], 1e-9)))
        z_err.append(abs(z - row["z"]))

    print(f"rows checked            : {len(rows)}")
    print(f"signatures valid        : {sig_ok}/{len(rows)}")
    print(f"structural check passed : {struct_ok}/{len(rows)}")
    if z_err:
        print(f"z reproduced on the published layer: max |delta| = {max(z_err):.2e} "
              f"(median {np.median(z_err):.2e})")
    labels = np.array([r["is_substituted"] for r in rows])
    print(f"ground truth in corpus  : {labels.sum()}/{len(labels)} rows served by M'")


if __name__ == "__main__":
    main()
