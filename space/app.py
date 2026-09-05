"""Assay demo: watch a cheating provider get caught, one request at a time.

The interesting control is "generation-consistency check". Turn it off and pick the
"forge" provider: the receipts are genuinely M's, every signature and hash checks out, and
the audit never fires -- because a matching sketch only proves M was *run over* the text,
not that M produced it. Turn it back on and the same provider is caught in a few requests.
"""

import math
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from assay.hooks import fake_quantize_int8_, tiny_random_pair
from assay.nondeterminism import NondeterminismModel
from assay.sequential_audit import EValueAudit
from assay.server import ProviderServer
from assay.sketch import ActivationSketcher, HmacSigner, SketchConfig
from assay.verifier import Calibrator, Verifier

DEMO_KEY = b"assay-demo-key-0123456789abcdef!"
SIGMA = 3e-4
GEN_LEN = 8
ALPHA = 0.01

# UI label -> (ProviderServer mode, which cheap model stands in for M')
MODES = {
    "honest": ("honest", None),
    "cheat (smaller model)": ("cheat", "size"),
    "quantized (INT8)": ("cheat", "quant"),
    "randomized (10% cheap)": ("random", "size"),
    "audit-evasive": ("evasive", "size"),
    "receipt forgery": ("forge", "size"),
}

MODE_HELP = {
    "honest": "Serves the real M on every request. Nothing should fire.",
    "cheat (smaller model)": "Serves the cheap M' on every request. The crudest substitution.",
    "quantized (INT8)": "Serves an INT8 copy of M. The text barely changes, so an audit that "
                        "only reads output text struggles; the activations move immediately.",
    "randomized (10% cheap)": "Serves M' on 10% of requests. This is what defeats a sampling "
              "audit: the canary queries mostly land on honest answers.",
    "audit-evasive": "Recognizes the auditor's canary prompts and answers those with M, "
               "cheating on everyone else. A black-box audit can never see it.",
    "receipt forgery": "Decodes with M', then runs M once over M''s own output to build a genuine "
             "receipt. Sketches match perfectly. Only the generation-consistency check "
             "catches this one -- switch it off and watch the audit go blind.",
}

C_ASSAY, C_MUTED, C_INK, C_SURFACE, C_HIT = "#2a78d6", "#898781", "#0b0b0b", "#fcfcfb", "#d03b3b"


class Demo:
    """Holds the seeded models and the calibrated band so each run is just the audit."""

    def __init__(self):
        torch.manual_seed(0)
        self.M, self.Mp = tiny_random_pair(seed=0)
        self.Mq = tiny_random_pair(seed=0)[0]
        fake_quantize_int8_(self.Mq)
        cfg = SketchConfig(layers=(1, 2, 3), n_positions=6, sketch_dim=32, rank_top_m=8)
        self.sketcher = ActivationSketcher(cfg, HmacSigner(DEMO_KEY))
        rng = np.random.default_rng(42)
        pool = [torch.tensor(rng.integers(0, 256, size=10), dtype=torch.long) for _ in range(48)]
        # Disjoint halves: calibrating and scoring on the same prompts makes the
        # false-positive rate look better than it is.
        self.cal_prompts, self.prompts = pool[0::2], pool[1::2]
        jitter = NondeterminismModel(SIGMA, seed=1)
        cal_server = ProviderServer(self.M, None, self.sketcher, mode="honest",
                                    jitter=jitter, gen_len=GEN_LEN, seed=0)
        self.calibration = Calibrator().fit(
            Verifier(self.M, self.sketcher, self.sketcher.signer), cal_server,
            self.cal_prompts, n=24)

    def run(self, ui_mode, n_requests, deep_check, seed):
        mode, which = MODES[ui_mode]
        mp = None if which is None else (self.Mq if which == "quant" else self.Mp)
        jitter = NondeterminismModel(SIGMA, seed=int(seed))
        server = ProviderServer(self.M, mp, self.sketcher, mode=mode, p_honest=0.9,
                                canaries=self.cal_prompts[:4], jitter=jitter,
                                gen_len=GEN_LEN, seed=int(seed))
        verifier = Verifier(self.M, self.sketcher, self.sketcher.signer, self.calibration,
                            verifier_seed=int(seed) + 7,
                            deep_check_rate=0.25 if deep_check else 0.0)
        audit = EValueAudit(alpha=ALPHA)
        history, rows, costs = [], [], []
        served_cheap = deep_runs = 0
        for t in range(int(n_requests)):
            prompt = self.prompts[t % len(self.prompts)]
            rid = f"demo-{seed}-{t}"
            resp = server.handle(rid, prompt)
            v = verifier.verify(resp.receipt, prompt, resp.output_ids, expected_request_id=rid)
            audit.update(v.p)
            history.append(audit.log_e)
            costs.append(v.cost_ratio)
            served_cheap += resp.model_used != "M"
            deep_runs += v.deep_checked
            rows.append([t + 1, resp.model_used, f"{v.z:+.2f}", f"{v.p:.2e}",
                         "yes" if v.deep_checked else "-",
                         f"{v.gen_mismatch}/{v.gen_checked}" if v.deep_checked else "-",
                         f"{audit.log_e:+.2f}"])
        return self._render(ui_mode, audit, history, rows, costs, served_cheap, deep_runs, deep_check)

    def _render(self, ui_mode, audit, history, rows, costs, served_cheap, deep_runs, deep_check):
        threshold = math.log(1.0 / ALPHA)
        fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=140)
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_SURFACE)
        ax.axhline(threshold, color=C_MUTED, ls="--", lw=1.2)
        ax.annotate(f"alarm threshold  log(1/α)", (0.01, threshold), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 4), fontsize=8, color=C_MUTED)
        cap = threshold * 4
        ax.plot(range(1, len(history) + 1), np.minimum(history, cap), color=C_ASSAY, lw=2.2)
        if audit.t_alarm:
            ax.plot([audit.t_alarm], [min(threshold, cap)], "o", color=C_HIT, ms=9, zorder=5)
            ax.annotate(f"detected at request {audit.t_alarm}", (audit.t_alarm, threshold),
                        textcoords="offset points", xytext=(8, 10), fontsize=9,
                        color=C_HIT, fontweight="bold")
        ax.set_xlabel("requests")
        ax.set_ylabel("cumulative log e-value")
        ax.set_ylim(min(-2, min(history, default=0) - 1), cap * 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e1e0d9", lw=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()

        n = len(history)
        if audit.t_alarm:
            verdict = f"### 🔴 Substitution detected at request {audit.t_alarm} of {n}"
        elif ui_mode == "honest":
            verdict = f"### 🟢 No alarm over {n} honest requests — as it should be"
        else:
            verdict = f"### ⚪ Not detected in {n} requests"
        notes = [
            f"- Provider actually served the cheap model on **{served_cheap}/{n}** requests",
            f"- Generation-consistency check ran on **{deep_runs}/{n}** requests"
            + ("" if deep_check else " — **switched off**"),
            f"- Verifier cost: **{np.mean(costs):.2f}×** a full recompute (mean over requests)",
        ]
        if ui_mode == "receipt forgery" and not deep_check:
            notes.append("- Every receipt here is genuinely M's. That is the point: the sketch "
                         "check cannot tell you who *generated* the text.")
        return fig, verdict + "\n" + "\n".join(notes), rows


demo_state = Demo()

with gr.Blocks(title="Assay — catching model substitution per request") as demo:
    gr.Markdown(
        "# Assay — catching model substitution, one receipt at a time\n"
        "An inference provider promises model **M** and quietly serves a cheaper **M'**. "
        "Every response here carries a signed sketch of its own `hidden_states` — an "
        "*internal receipt* — and a verifier recomputes a small, secretly-chosen slice of it. "
        "Coverage is 100% by construction: there is no request the audit happened to miss.\n\n"
        "[Code](https://github.com/NagaYu/assay) · "
        "[Receipt corpus](https://huggingface.co/datasets/NagaYu/assay-receipts) · "
        "[Model bundle](https://huggingface.co/NagaYu/assay-tiny-pair)"
    )
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(list(MODES), value="receipt forgery", label="Provider behaviour")
            mode_help = gr.Markdown(MODE_HELP["receipt forgery"])
            n_requests = gr.Slider(10, 120, value=40, step=5, label="Requests to audit")
            deep_check = gr.Checkbox(
                value=True, label="Generation-consistency check (25% of requests)",
                info="Off = sketch check only. Try that with the 'receipt forgery' provider.")
            seed = gr.Slider(0, 50, value=3, step=1, label="Run seed")
            run_btn = gr.Button("Run audit", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="Cumulative evidence")
            summary = gr.Markdown()
    table = gr.Dataframe(
        headers=["#", "served by", "z", "p", "deep check", "token mismatches", "cumulative log e"],
        label="Per-request verdicts ('served by' is ground truth, invisible to a real auditor)",
        wrap=True)

    mode.change(lambda m: MODE_HELP[m], mode, mode_help)
    run_btn.click(demo_state.run, [mode, n_requests, deep_check, seed], [plot, summary, table])
    demo.load(demo_state.run, [mode, n_requests, deep_check, seed], [plot, summary, table])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
