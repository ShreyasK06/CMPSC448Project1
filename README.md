# CMPSC448 Project 1: attributing LLM text to its generator (RAID)

Five-way classification (chatgpt, cohere-chat, llama-chat, mistral-chat, mpt-chat) with TF-IDF+LR baselines,
a TextCNN and a BiLSTM. See `data/DATA_CARD.md` for the data.

## Running on a GPU machine

The processed data (`data/processed/raid_main.parquet`, `raid_rq3.parquet`, `vocab.json`, ~25 MB) is tracked
in git, so a clone is all the data you need. Training only reads `data/processed/`; `data/interim/` and
`data/raw/` (used only to rebuild the processed files) are not needed.

```bash
git clone <your-repo-url> CMPSC448Project1 && cd CMPSC448Project1
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) CUDA build of torch FIRST (pick the index matching your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu124     # or .../cu121
# 2) the rest, without touching torch
grep -v '^torch==' requirements.txt > req-notorch.txt && pip install -r req-notorch.txt
#    Windows PowerShell: (Get-Content requirements.txt) -notmatch '^torch==' | Set-Content req-notorch.txt; pip install -r req-notorch.txt

python -c "import torch; print(torch.cuda.is_available())"   # must print True
python run_gpu.py --quick     # optional smoke test, writes only to quick_out/ (git-ignored)
python run_gpu.py             # full pipeline
```

`run_gpu.py` runs, in order: (a) CNN then BiLSTM hyper-parameter tuning on the `output` variant (validation
only); (b) final training for {cnn, bilstm} x {output, both}, seeds 0 1 2, test evaluated once at the best-val
checkpoint; (c) the same on the RQ3 split (`--rq3`, tag `rq3`); (d) TF-IDF+LR baselines (word, word+char, and
RQ3) with test evaluation; (e) `python -m src.aggregate --out reports/results.md`. Logs go to `logs/`.
It is resumable: each step is skipped if its output file exists, so after an interruption just re-run it
(delete a file in `runs/` or `reports/` to redo that step). Flags: `--skip-tune`, `--quick`, `--device`,
`--seeds`, `--max-epochs`.

Notes:
- `--device auto` uses CUDA if available. On CUDA the BiLSTM is the standard packed bidirectional `nn.LSTM`;
  on CPU it is the faster two-LSTM formulation (tests assert the two give identical logits and gradients).
- The `prompt` variant is not trained for the neural models: the prompt is identical across the 5 generator
  models, so it is at chance by construction (the TF-IDF baseline still reports it as a sanity check).
- Variant `both` reuses the hyper-parameters tuned on `output`.
- The baseline C grid now includes 1000 (best C=100 sat at the old grid edge), so the existing val-only
  baseline JSONs are regenerated (with test evaluation) by the full run.

### Copying results back and committing

`reports/` and `runs/` are tracked (only `logs/`, `checkpoints/`, `*.pt` and the data cache are ignored):

```bash
git add reports runs
git commit -m "GPU run results"
git push origin HEAD            # or copy reports/ and runs/ back with scp / a zip
```

Key files: `reports/results.md` (aggregate table), `reports/tuning/*.json`, `reports/baselines/*.json`,
`runs/*.json` (per-seed metrics) and `runs/*.npz` (test predictions).
