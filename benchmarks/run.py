"""Benchmark: Assay (per-request internal receipts) vs black-box statistical audit.

Four detection tasks:
  size    full substitution (M=gpt2 12 layers -> M'=distilgpt2 6 layers; CPU stand-in for Llama-3.2 3B->1B)
  quant   quantization substitution (FP32 -> INT8 fake-quant; CPU stand-in for BF16->INT8)
  random  randomized substitution (M with prob p=0.9, else M')
  evasive audit evasion (respond with M when a known canary is detected)

Metrics: queries to detection / ROC-AUC / false-positive rate / verifier cost ratio /
coverage (fraction of responses inspected; Assay=100%, BB=audit query rate).

Examples:
  python benchmarks/run.py                        # gpt2 pair (needs HF cache, ~10 min on CPU)
  python benchmarks/run.py --pair tiny --quick    # randomly-initialized tiny pair (~1 min)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assay.hooks import load_int8_model, load_model, load_tokenizer, num_layers, tiny_random_pair
from assay.nondeterminism import NondeterminismModel
from assay.sequential_audit import EValueAudit, SPRTAudit
from assay.server import ProviderServer
from assay.sketch import ActivationSketcher, HmacSigner, SketchConfig
from assay.verifier import Calibrator, Verifier
from benchmarks.blackbox import CanaryAuditor, DistributionAuditor, reference_outputs
from benchmarks import figures


# ---------------------------------------------------------------------------
# Prompt pool
# ---------------------------------------------------------------------------

_TOPICS = [
    "the history of paper money", "how volcanoes form", "the rules of chess",
    "why leaves change color", "the invention of the telephone", "life in a coral reef",
    "how bridges are built", "the first trip to the moon", "why the sky looks blue",
    "how bees make honey", "the story of the printing press", "what makes thunder",
    "how trains changed cities", "the deep ocean floor", "why rivers bend",
    "the oldest known maps", "how glass is made", "the busiest airport in the world",
    "why cats purr", "the tallest mountain on earth", "how clocks keep time",
    "the great wall and its builders", "what ants carry home", "how radios send music",
]
_TEMPLATES = [
    "Tell me about {}.", "Explain {} in simple terms.", "Write a short note on {}.",
]

_CANARY_TOPICS = [
    "a violet zeppelin over quiet harbors", "seventeen marble owls at midnight",
    "the recipe for luminous river stones", "a clockwork whale in the desert",
    "the library of forgotten umbrellas", "six copper bells under the ice",
    "a chess match between two lighthouses", "the last tram to the observatory",
    "a garden of magnetic tulips", "the museum of misplaced shadows",
    "a telegraph line across the moon", "the choir of paper lanterns",
]


def build_prompts(tokenizer, max_tokens: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build user-traffic prompts and the auditor's canary prompts (a distinct family)."""
    user, canary = [], []
    for i, topic in enumerate(_TOPICS):
        text = _TEMPLATES[i % len(_TEMPLATES)].format(topic)
        ids = tokenizer(text, return_tensors="pt").input_ids[0][:max_tokens]
        user.append(ids)
    for topic in _CANARY_TOPICS:
        text = f"Describe {topic}."
        ids = tokenizer(text, return_tensors="pt").input_ids[0][:max_tokens]
        canary.append(ids)
    return user, canary


def build_random_prompts(vocab: int, n_user: int, n_canary: int, max_tokens: int, seed: int):
    rng = np.random.default_rng(seed)
    user = [torch.tensor(rng.integers(0, vocab, size=max_tokens), dtype=torch.long) for _ in range(n_user)]
    canary = [torch.tensor(rng.integers(0, vocab, size=max_tokens), dtype=torch.long) for _ in range(n_canary)]
    return user, canary


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties averaged) -- a numpy reimplementation compatible with scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc_score(neg: List[float], pos: List[float]) -> float:
    x = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
    x = np.clip(x, -100, 100)
    ranks = _rankdata(x)
    r_pos = ranks[len(neg):].sum()
    n0, n1 = len(neg), len(pos)
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((r_pos - n1 * (n1 + 1) / 2) / (n0 * n1))


def roc_curve(neg: List[float], pos: List[float]) -> Dict[str, list]:
    scores = np.clip(np.concatenate([neg, pos]), -100, 100)
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(-scores)
    labels = labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    tpr = np.r_[0, tps / max(labels.sum(), 1)]
    fpr = np.r_[0, fps / max((1 - labels).sum(), 1)]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_score(neg, pos)}


def median_or_none(vals: List[Optional[int]]) -> Optional[float]:
    """None if a majority were not detected; otherwise the median over the detected trials."""
    detected = [v for v in vals if v is not None]
    if len(detected) * 2 <= len(vals):
        return None
    return float(np.median(detected))


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

SCENARIO_DEFS = {
    "honest": dict(mode="honest", mp=None),
    "size": dict(mode="cheat", mp="size"),
    "quant": dict(mode="cheat", mp="quant"),
    "random": dict(mode="random", mp="size"),
    "evasive": dict(mode="evasive", mp="size"),
}


def _name_hash(name: str) -> int:
    """Deterministic per-scenario hash (unlike Python's salted str hash), so the whole
    benchmark is reproducible across processes without setting PYTHONHASHSEED."""
    import hashlib

    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def run_scenario(ctx: dict, name: str, trial: int, n_requests: int, collect_full_cost: bool = False) -> dict:
    """Run one scenario x one trial of traffic, driving Assay and the (interleaved) BB audit together."""
    d = SCENARIO_DEFS[name]
    mp = ctx["models"].get(d["mp"]) if d["mp"] else None
    jitter = NondeterminismModel(ctx["sigma"], seed=trial * 7919 + _name_hash(name) % 4096)
    server = ProviderServer(
        ctx["M"], mp, ctx["sketcher"], mode=d["mode"], p_honest=ctx["p_honest"],
        canaries=ctx["canaries"], jitter=jitter, gen_len=ctx["gen_len"], seed=trial * 131 + 7,
    )
    verifier = Verifier(ctx["M"], ctx["sketcher"], ctx["signer"], ctx["calib"],
                        verifier_seed=trial * 991 + 13)
    audit = EValueAudit(alpha=ctx["alpha"])
    sprt = SPRTAudit(alpha=ctx["alpha"])
    bb = CanaryAuditor(ctx["canaries"], ctx["refs"], q0=ctx["q0"], alpha=ctx["alpha"], seed=trial * 17 + 3)
    rng = np.random.default_rng(trial * 31 + _name_hash(name) % 1000)
    order = rng.permutation(len(ctx["prompts"]))

    rec = dict(
        name=name, trial=trial, n_requests=n_requests, audit_rate=ctx["audit_rate"],
        assay_history=[], bb_history_x=[], bb_history_y=[],
        zs=[], labels=[], sign=[], rank=[], cost_ratios=[], verify_walls=[], full_walls=[],
        user_tokens=[],
    )
    bb_t_alarm_traffic = None
    for t in range(n_requests):
        is_audit = rng.random() < ctx["audit_rate"]
        if is_audit:
            c_idx, prompt = bb.pick_canary()
        else:
            prompt = ctx["prompts"][order[t % len(order)]]
        resp = server.handle(f"{name}-{trial}-{t}", prompt)
        verdict = verifier.verify(resp.receipt, prompt, resp.output_ids)
        audit.update(verdict.p)
        sprt.update(verdict.z)
        rec["assay_history"].append(audit.log_e)
        rec["zs"].append(min(verdict.z, 100.0))
        rec["labels"].append(1 if resp.model_used == "M'" else 0)
        rec["sign"].append(verdict.sign_agreement)
        rec["rank"].append(verdict.rank_overlap)
        rec["cost_ratios"].append(verdict.cost_ratio)
        rec["verify_walls"].append(verdict.wall_time)
        if not is_audit:
            rec["user_tokens"].extend(resp.output_ids)
        if collect_full_cost and t % 10 == 0:
            rec["full_walls"].append(verifier.full_recompute_time(prompt, resp.output_ids))
        if is_audit:
            bb.observe(c_idx, resp.output_ids)
            rec["bb_history_x"].append(t + 1)
            rec["bb_history_y"].append(bb.eproc.log_e)
            if bb_t_alarm_traffic is None and bb.alarmed:
                bb_t_alarm_traffic = t + 1
    rec["assay_t_alarm"] = audit.t_alarm
    rec["sprt_t_alarm"] = sprt.t_alarm
    rec["bb_t_alarm_traffic"] = bb_t_alarm_traffic
    rec["bb_n_queries"] = len(rec["bb_history_x"])
    return rec


def run_bb_dense(ctx: dict, name: str, trial: int, n_queries: int) -> dict:
    """BB dedicated-query mode: the auditor keeps firing canaries only, no Assay -- its upper-bound performance."""
    d = SCENARIO_DEFS[name]
    mp = ctx["models"].get(d["mp"]) if d["mp"] else None
    jitter = NondeterminismModel(ctx["sigma"], seed=trial * 733 + _name_hash(name) % 512 + 99)
    server = ProviderServer(
        ctx["M"], mp, ctx["sketcher"], mode=d["mode"], p_honest=ctx["p_honest"],
        canaries=ctx["canaries"], jitter=jitter, gen_len=ctx["gen_len"], seed=trial * 401 + 11,
    )
    bb = CanaryAuditor(ctx["canaries"], ctx["refs"], q0=ctx["q0"], alpha=ctx["alpha"], seed=trial * 53 + 29)
    for t in range(n_queries):
        c_idx, prompt = bb.pick_canary()
        resp = server.handle(f"bbdense-{name}-{trial}-{t}", prompt)
        bb.observe(c_idx, resp.output_ids)
        if bb.alarmed:
            break
    return {"history": list(bb.eproc.history), "t_alarm": bb.t_alarm}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(args) -> dict:
    print(f"[setup] pair={args.pair}")
    if args.pair == "gpt2":
        M = load_model(args.model_m)
        M_size = load_model(args.model_mp)
        M_quant = load_int8_model(args.model_m)
        tok = load_tokenizer(args.model_m)
        prompts, canaries = build_prompts(tok, max_tokens=args.prompt_len)
    else:
        M, M_size = tiny_random_pair(seed=0)
        M_quant = tiny_random_pair(seed=0)[0]
        from assay.hooks import fake_quantize_int8_

        fake_quantize_int8_(M_quant)
        tok = None
        prompts, canaries = build_random_prompts(M.config.vocab_size, 24, 12, args.prompt_len, seed=42)

    n_layer = num_layers(M)
    cfg = SketchConfig(layers=SketchConfig.auto_layers(n_layer), n_positions=6, sketch_dim=32)
    signer = HmacSigner(b"assay-demo-key-0123456789abcdef!")
    sketcher = ActivationSketcher(cfg, signer)
    print(f"[setup] n_layer={n_layer} sketch_layers={cfg.layers}")

    # Calibration (honest operation + jitter, one-time preprocessing)
    jitter = NondeterminismModel(args.sigma, seed=12345)
    srv_cal = ProviderServer(M, None, sketcher, mode="honest", jitter=jitter, gen_len=args.gen_len, seed=999)
    verifier = Verifier(M, sketcher, signer)
    t0 = time.time()
    calib = Calibrator().fit(verifier, srv_cal, prompts, n=args.n_cal)
    print(f"[setup] calibration n={args.n_cal} took {time.time()-t0:.1f}s "
          f"mu={ {k: round(v, 2) for k, v in calib.mu.items()} }")

    # BB auditor reference outputs (clean M greedy) and honest mismatch-rate q0 calibration
    refs = reference_outputs(M, canaries, args.gen_len)
    from benchmarks.blackbox import token_agreement

    mism = 0
    n_q0 = 2 * len(canaries)
    for i in range(n_q0):
        c = i % len(canaries)
        resp = srv_cal.handle(f"q0-{i}", canaries[c])
        if token_agreement(refs[c], resp.output_ids) < 0.9:
            mism += 1
    q0 = max((mism + 1) / (n_q0 + 2), 0.005)
    print(f"[setup] BB honest mismatch q0={q0:.3f} ({mism}/{n_q0})")

    return dict(
        M=M, models={"size": M_size, "quant": M_quant}, tokenizer=tok,
        prompts=prompts, canaries=canaries, refs=refs, sketcher=sketcher, signer=signer,
        calib=calib, q0=q0, sigma=args.sigma, gen_len=args.gen_len,
        alpha=args.alpha, audit_rate=args.audit_rate, p_honest=args.p_honest,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", choices=["gpt2", "tiny"], default="gpt2")
    ap.add_argument("--model-m", default="gpt2")
    ap.add_argument("--model-mp", default="distilgpt2")
    ap.add_argument("--n", type=int, default=80, help="requests per scenario")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--n-bb", type=int, default=150, help="max queries for BB dedicated mode")
    ap.add_argument("--n-cal", type=int, default=48)
    ap.add_argument("--gen-len", type=int, default=12)
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--sigma", type=float, default=3e-4)
    ap.add_argument("--audit-rate", type=float, default=0.1)
    ap.add_argument("--p-honest", type=float, default=0.9)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--quick", action="store_true", help="reduced settings (n=40, trials=2, n_cal=24)")
    args = ap.parse_args()
    if args.quick:
        args.n, args.trials, args.n_cal, args.n_bb = 40, 2, 24, 60

    torch.manual_seed(0)
    os.makedirs(args.outdir, exist_ok=True)
    ctx = setup(args)

    scenario_names = ["honest", "size", "quant", "random", "evasive"]
    all_trials: Dict[str, List[dict]] = {s: [] for s in scenario_names}
    dense: Dict[str, dict] = {}

    for name in scenario_names:
        n_trials = 1 if name == "honest" else args.trials
        for trial in range(n_trials):
            t0 = time.time()
            rec = run_scenario(ctx, name, trial, args.n, collect_full_cost=(name == "honest"))
            all_trials[name].append(rec)
            print(f"[run] {name} trial {trial}: assay_t={rec['assay_t_alarm']} "
                  f"sprt_t={rec['sprt_t_alarm']} bb_t={rec['bb_t_alarm_traffic']} "
                  f"({time.time()-t0:.1f}s)")
        if name != "honest":
            dense[name] = run_bb_dense(ctx, name, 0, args.n_bb)
            print(f"[run] {name} bb_dense: t={dense[name]['t_alarm']}")

    # ---- Aggregate --------------------------------------------------------
    honest = all_trials["honest"][0]
    neg_z = honest["zs"]
    fpr_seq = 1.0 if honest["assay_t_alarm"] else 0.0
    fpr_req = float(np.mean(np.asarray(neg_z) > 3.09))

    results: Dict[str, dict] = {}
    rocs: Dict[str, dict] = {}
    qtd: Dict[str, dict] = {}
    inv = {"honest": {"sign": honest["sign"], "rank": honest["rank"]}}
    dist_detect: Dict[str, Optional[int]] = {}

    # Simple distribution-comparison BB (a passive observer of all traffic text)
    ref_pool = honest["user_tokens"]
    vocab = int(ctx["M"].config.vocab_size)

    for name in scenario_names[1:]:
        trials = all_trials[name]
        first = trials[0]
        pos_mask = np.asarray(first["labels"]) == 1
        pos_z = list(np.asarray(first["zs"])[pos_mask])
        rocs[name] = roc_curve(neg_z, pos_z)
        qtd[name] = {
            "assay": median_or_none([t["assay_t_alarm"] for t in trials]),
            "bb": median_or_none([t["bb_t_alarm_traffic"] for t in trials]),
            "bb_dense": dense[name]["t_alarm"],
        }
        if name in ("quant", "size"):
            inv[name] = {
                "sign": list(np.asarray(first["sign"])[pos_mask]),
                "rank": list(np.asarray(first["rank"])[pos_mask]),
            }
        # Distribution comparison: how many queries' worth of text does it take to detect?
        da = DistributionAuditor(ref_pool, vocab, seed=7)
        toks = first["user_tokens"]
        found = None
        for nq in (5, 10, 20, 40, 60):
            upto = nq * ctx["gen_len"]
            if upto <= len(toks) and da.detects(toks[:upto]):
                found = nq
                break
        dist_detect[name] = found
        results[name] = {
            "assay_history": first["assay_history"],
            "assay_t_alarm": first["assay_t_alarm"],
            "bb_history_x": first["bb_history_x"],
            "bb_history_y": first["bb_history_y"],
            "bb_t_alarm_traffic": first["bb_t_alarm_traffic"],
        }

    cost_ratios = [r for t in all_trials.values() for rec in t for r in rec["cost_ratios"]]
    verify_walls = [w for t in all_trials.values() for rec in t for w in rec["verify_walls"]]
    full_walls = honest["full_walls"]
    cost = {
        "layer_token_ratio": float(np.mean(cost_ratios)),
        "wall_ratio": float(np.mean(verify_walls) / max(np.mean(full_walls), 1e-9)),
        "mean_verify_wall_s": float(np.mean(verify_walls)),
        "mean_full_wall_s": float(np.mean(full_walls)),
    }

    threshold = math.log(1.0 / args.alpha)
    summary = {
        "config": {k: getattr(args, k) for k in ("pair", "n", "trials", "gen_len", "alpha",
                                                  "sigma", "audit_rate", "p_honest", "n_cal")},
        "honest": {"fpr_per_request_z3": fpr_req, "sequential_false_alarm": fpr_seq,
                   "assay_final_log_e": honest["assay_history"][-1]},
        "queries_to_detection": qtd,
        "auc": {k: v["auc"] for k, v in rocs.items()},
        "distribution_bb_queries": dist_detect,
        "coverage": {"assay": 1.0, "bb_interleaved": args.audit_rate},
        "verifier_cost": cost,
        "invariants_honest": {"sign": float(np.mean(honest["sign"])), "rank": float(np.mean(honest["rank"]))},
    }

    # ---- Output -----------------------------------------------------------
    figures.fig_headline(results, threshold, os.path.join(args.outdir, "fig_headline.png"))
    figures.fig_roc(rocs, os.path.join(args.outdir, "fig_roc.png"))
    figures.fig_cost(cost, os.path.join(args.outdir, "fig_cost.png"))
    figures.fig_invariance(inv, os.path.join(args.outdir, "fig_invariance.png"))
    figures.fig_queries(qtd, args.n, os.path.join(args.outdir, "fig_queries.png"))

    with open(os.path.join("benchmarks", "results.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "| Scenario | Assay detect (req) | BB 10% interleaved | BB dedicated | Distribution BB | AUC |",
        "|---|---|---|---|---|---|",
    ]
    label = {"size": "Full substitution", "quant": "Quantization",
             "random": "Randomized p=0.9", "evasive": "Audit evasion"}
    fmt = lambda v: ("**not detected**" if v is None else f"{v:g}")
    for name in scenario_names[1:]:
        q = qtd[name]
        lines.append(
            f"| {label[name]} | {fmt(q['assay'])} | {fmt(q['bb'])} | {fmt(q['bb_dense'])} "
            f"| {fmt(dist_detect[name])} | {rocs[name]['auc']:.3f} |"
        )
    lines += [
        "",
        f"- Honest operation: sequential false alarm = {'yes' if fpr_seq else 'no'}, "
        f"per-request FPR(z>3.09) = {fpr_req:.3f}",
        f"- Verifier cost ratio: layer-token {cost['layer_token_ratio']:.2f} / "
        f"wall {cost['wall_ratio']:.2f} (full recompute = 1.0)",
        f"- Coverage: Assay 100% / BB {args.audit_rate:.0%}",
    ]
    lines.append(f"- Config: pair={args.pair}, n={args.n}/scenario, trials={args.trials}, alpha={args.alpha} (CPU)")
    with open(os.path.join("benchmarks", "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    update_readme_table("\n".join(lines))
    print("\n".join(lines))
    print(f"[done] figures -> {args.outdir}/, summary -> benchmarks/results.json")


def update_readme_table(table_md: str, readme_path: str = "README.md") -> None:
    """Auto-insert the benchmark table into the README's marker span (refreshed on every run)."""
    start, end = "<!-- BENCH_TABLE_START -->", "<!-- BENCH_TABLE_END -->"
    try:
        with open(readme_path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return
    if start not in text or end not in text:
        return
    pre = text.split(start)[0]
    post = text.split(end, 1)[1]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(pre + start + "\n" + table_md + "\n" + end + post)


if __name__ == "__main__":
    main()
