# Phase 3 GPU tasks (output-length follow-up)

Run these on the GPU machine after `git pull`. Environment setup is the same as in `README.md` ("Running on a GPU machine").
Everything below uses only the existing `python -m src.train` CLI and tracked data (`data/processed/raid_main.parquet`).

## Why

Phase 2 (T8) showed the 256-token output cap is the main limiter. Test accuracy of the 3-seed ensemble vs the 256-token base:

| model | cap 64 | cap 128 | cap 256 (base) | cap 512 |
|---|---|---|---|---|
| CNN | -12.7 pp | -7.2 pp | 71.8% | +1.8 pp (Holm p=0.11, not significant) |
| BiLSTM | n/a | -4.4 pp | 69.4% | +2.8 pp (Holm p=0.004, significant) |

Accuracy was still rising at 512, so the open questions are (a) where it saturates and (b) whether the tuned `ext` configs
(T9) stack with a longer cap. Extended tuning on its own gave no significant gain (CNN +0.7 pp, BiLSTM -0.3 pp, `output` variant).

## Tasks

| id | what | models | cap(s) | config | tags | fits |
|---|---|---|---|---|---|---|
| T12 | cap sweep beyond 512 | CNN, BiLSTM | 1024, 2048 | stage-1 base | `abl_len1024`, `abl_len2048` | 12 |
| T13 | tuned config x longer cap | CNN, BiLSTM | 512, 1024 | T9 `ext` | `ext_len512`, `ext_len1024` | 12 |

All runs: variant `output`, seeds 0 1 2, `--eval-test`, `--max-epochs 20`, early stopping on VAL (train.py default).

Hyper-parameters (from `reports/tuning/`):

- base CNN: `--dropout 0.3 --nf 200 --emb 256` (weight decay: train.py default)
- base BiLSTM: `--dropout 0.5 --hid 256` (weight decay: train.py default)
- ext CNN: `--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001`
- ext BiLSTM: `--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001`

## Commands

Smoke test first (one seed, one epoch, writes to `quick_out/`, git-ignored):

```bash
python -m src.train --model cnn --variant output --seeds 0 --max-epochs 1 --max-out-len 1024 \
  --dropout 0.3 --nf 200 --emb 256 --tag smoke --runs-dir quick_out --device auto
```

Then the full set. Each run is skipped if its result file already exists, so it is safe to re-run after an interruption:

```bash
set -e
run() {  # run <model> <tag> <cap> <hp flags...>
  model=$1; tag=$2; cap=$3; shift 3
  for s in 0 1 2; do
    f="runs/${model}_output_${tag}_s${s}.json"
    [ -f "$f" ] && { echo "skip $f"; continue; }
    python -m src.train --model "$model" --variant output --seeds "$s" --eval-test \
      --tag "$tag" --max-out-len "$cap" "$@" --device auto 2>&1 | tee -a "logs/phase3_${model}_${tag}.log"
  done
}
mkdir -p logs
BASE_CNN="--dropout 0.3 --nf 200 --emb 256"
BASE_LSTM="--dropout 0.5 --hid 256"
EXT_CNN="--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001"
EXT_LSTM="--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001"

# T12: cap sweep (base configs)
for cap in 1024 2048; do
  run cnn    abl_len$cap $cap $BASE_CNN
  run bilstm abl_len$cap $cap $BASE_LSTM
done

# T13: tuned config x longer cap
for cap in 512 1024; do
  run cnn    ext_len$cap $cap $EXT_CNN
  run bilstm ext_len$cap $cap $EXT_LSTM
done
```

Tag prefixes `abl_` and `ext` are already treated as phase-2 tags, so `src.aggregate` keeps these runs out of the main table.

## Runtime

Not measured. Cost scales roughly linearly with the cap (CNN) and at least linearly for the BiLSTM; the 2048-token BiLSTM
runs are the expensive ones (expect several times the 256-token fit time of ~100 s on CPU). Budget a few hours in total; the
loop order above puts the cheaper runs first within each task. If time is short, drop `2048` from T12 first.

## Reporting rules (so the numbers stay honest)

- Choose any "best cap" by mean **VAL** macro-F1 only. TEST is written for every run by `--eval-test`; do not pick the cap by looking at TEST.
- Report VAL and TEST numbers side by side, labelled by split, for every row, with the 3-seed ensemble CI against the 256-token `base` of the same model.
- Caveat for the write-up: more rows see long outputs, so also report the share of test outputs truncated at each cap.
- `t8_report` only knows the Phase 2 ablation list, so these tags are not in the generated tables yet. Do not run the report step expecting them; the raw `runs/*.json` and `.npz` files are what matters.

## Bringing results back

```bash
git add runs
git commit -m "Phase 3 GPU results: cap sweep and tuned config x cap"
git push origin HEAD
```

If the push is rejected, `git pull --rebase` first (the run files have unique names, so there should be no conflicts).

> Follow-up (Phase 4, confirm the winning config): see [GPU_PHASE4.md](GPU_PHASE4.md).
