# Space sources

Two versions of the demo live here.

- `index.html` — the **static** Space that is actually deployed at
  [NagaYu/assay-demo](https://huggingface.co/spaces/NagaYu/assay-demo). It replays the
  published receipt corpus, so it loads instantly and shows real gpt2 / distilgpt2 numbers.
  Its `runs.json` payload is produced from `benchmarks/export_hf.py` output (see
  `make_runs_json` in this repo's history, or regenerate with the snippet in the dataset card).
- `app.py` — a **Gradio** version that runs the audit live on the seeded tiny pair. Hosting a
  Gradio Space needs a paid HF tier, so it is kept here rather than deployed; run it locally
  with `pip install gradio && python space/app.py`.

Both demonstrate the same thing: pick the `forge` provider, switch the generation-consistency
check off, and watch a provider serving M' text under genuine M receipts go completely
undetected.
