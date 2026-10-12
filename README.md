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

> GPU job instructions for the later rounds (all completed) are in GPU_PHASE3.md to GPU_PHASE6.md; findings are in reports/results.md.

## Phase 2 on the GPU machine

Phase 2 adds three studies on top of the finished `run_gpu.py` run. It reuses the tuned base configs in
`reports/tuning/{cnn,bilstm}_output.json`, so run (or pull the results of) `run_gpu.py` first. It needs only tracked files:
`data/processed/raid_main.parquet` and, for the secondary scaling study, `data/processed/raid_scaled.parquet` (28 MB,
built locally by `python -m src.make_scaling_data`; see the DATA_CARD addendum). T11 (pretrained transformer) is intentionally not included.

- **T8 neural ablations** (variant `output`, 3 seeds, test at the best-val checkpoint): lowercase (CNN, BiLSTM), output
  cap 64/128/512 tokens (CNN) and 128/512 (BiLSTM), format-normalised input (CNN: markdown/bullets stripped, no `<NL>`).
  Also `t8_base_valprobs`: the base CNN/BiLSTM, `output`+`both`, re-run as tag `base_vp` with identical configs so that VAL
  probabilities are stored (`val_y/val_pred/val_prob/val_source_id` in the npz). They should match `base` within seed noise
  and replace nothing. From now on every neural run stores VAL probabilities.
- **T9 extended tuning** (VAL only, test untouched): CNN grid dropout {0.1,0.2,0.3} x nf {200,300} x emb 256 x weight decay
  {0,1e-4} (12 configs; `--with-emb-dropout` adds the same 12 with embedding dropout 0.2); BiLSTM 6 configs
  (hid 256: dropout {0.3,0.5} x wd {0,1e-4}; hid 384: dropout 0.5 x wd {0,1e-4}). Stage 1 = whole grid at seed 0, stage 2 = top 3
  re-run with seeds 0,1,2, pick the best mean val macro-F1 -> `reports/tuning/{cnn,bilstm}_output_ext.json`; then final 3-seed
  runs (`output` and `both`, tag `ext`, `--eval-test`).
- **T10 data scaling (secondary, beyond spec)**: CNN with the T9-chosen config (fallback: stage-1 best) at 100% of the
  2,240 train ids (tag `scale100`) and on the expanded train set (tag `scaled`, 6,240 train ids; val/test identical).
  The TF-IDF learning curve (25/50/100% + expanded) is CPU-only and already committed
  (`python -m src.analysis.t10_scaling_tfidf`, a few minutes; it is also a step in the script and skipped when
  `reports/baselines/scaling_tfidf.json` exists). `--scaling-cnn-fractions 25 50` adds CNN points at nested subsets.

```bash
git pull                                   # data/processed/raid_scaled.parquet and the code arrive with the pull
# environment: as above (CUDA torch first, then requirements without torch)
python run_gpu_phase2.py --quick           # ALWAYS FIRST: 1 epoch, 1 seed, tiny configs, writes only to quick_out/phase2/ (~10 min on CPU)
python run_gpu_phase2.py                   # everything: T8 -> T9 -> T10, then the report (resumable: just re-run after an interruption)
# useful variants
python run_gpu_phase2.py --only t8         # or t9, t10, or several: --only t8 t9
python run_gpu_phase2.py --skip-tune       # reuse reports/tuning/*_output_ext.json
python run_gpu_phase2.py --with-emb-dropout --scaling-cnn-fractions 25 50
python run_gpu_phase2.py --device cuda:1
```

**Runtime estimate** (not measured on a GPU): about 60 CNN-sized fits (base_vp, ablations, 18 tuning fits, ext, scaling) and
about 35 BiLSTM fits. On the CPU this project was developed on the stage-1 tuning fits took ~25-35 s (CNN) and ~80-100 s
(BiLSTM) each, and a modern GPU is the same order or faster, so expect roughly **1.5-3 hours** in total, dominated by the BiLSTM
and by the expanded-data runs (2.8x more training rows). `--with-emb-dropout` adds ~18 CNN-sized and ~12 BiLSTM-sized fits.
Per-step logs are in `logs/` (git-ignored); the final table lists the time of every step.

**Bringing results back** (new files are in `runs/` and `reports/`; the report step has already rewritten the results
sections, and `src.aggregate` no longer overwrites them):

```bash
git add runs reports data/processed
git commit -m "Phase 2 GPU results"
git push origin HEAD
```

The step `report_phase2` (`python -m src.analysis.t8_report`) writes three clearly labelled sections to `reports/results.md`
(and `reports/analysis.md`): `## Ablations`, `## Extended tuning`, `## Secondary: learning curve (beyond spec)`. Each shows
mean +/- std over seeds with the split of every number, a 3-seed ensemble with cluster-bootstrap CIs, and exact McNemar vs base
(Holm-corrected within the section). `python -m src.aggregate ...` keeps every `## ` section of an existing `results.md`
and keeps phase-2 tags out of the main table. Re-run only the report with
`python -m src.analysis.t8_report` (cheap, idempotent).
