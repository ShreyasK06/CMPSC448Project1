# Phase 5 GPU tasks (best-vs-best against TF-IDF)

Run on the GPU machine after `git pull`. Environment setup is as in `README.md`. Budget: about 40 minutes.

## Why

Phase 4 report (`reports/analysis.md`, "Output length and tuned config") shows the tuned CNN/BiLSTM gain 3-4 pp from a 512-token cap.
Checking that against TF-IDF (`python -m src.analysis.t14_compare`, already runnable on the current data):

- At the same 256-token cap, the tuned CNN (cap 512) ties TF-IDF K256 (McNemar p=0.73), and the BiLSTM is within Holm-significance.
- But TF-IDF with NO truncation (Kfull) reaches 78.7% test accuracy and still beats every neural model (Holm p < 1e-3).

So the neural models were never given their best input length. Phase 5 runs the tuned config at a 2048-token cap (longer than almost all
outputs) so the final comparison is each model at its best length. Cap 2048 was already run with the *base* config in Phase 3; this is the
tuned (`ext`) config. Nothing here changes any earlier selection.

## Tasks (priority order; each run is skipped if its result file exists)

| # | what | fits | est. time |
|---|---|---|---|
| 1 | CNN `output`, ext config, cap 2048, seeds 0-4 | 5 | about 4 min |
| 2 | BiLSTM `output`, ext config, cap 2048, seeds 0-2 | 3 | about 14 min |
| 3 | BiLSTM `both`, ext config, cap 512, seeds 3-4 (completes the 5-seed row) | 2 | about 9 min |
| 4 | BiLSTM `output`, ext config, cap 2048, seeds 3-4 | 2 | about 9 min |

Times are estimates from earlier per-fit times, not measurements. Fixed settings: `--eval-test`, early stopping on VAL, tag `ext_len2048`
(tasks 1, 2, 4) or `ext_len512` (task 3).

- ext CNN: `--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001`
- ext BiLSTM: `--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001`

## Commands

```bash
git pull
mkdir -p logs
run() {  # run <model> <variant> <tag> <cap> <seed> <hp flags...>
  model=$1; var=$2; tag=$3; cap=$4; s=$5; shift 5
  f="runs/${model}_${var}_${tag}_s${s}.json"
  [ -f "$f" ] && { echo "skip $f"; return; }
  python -m src.train --model "$model" --variant "$var" --seeds "$s" --eval-test \
    --tag "$tag" --max-out-len "$cap" "$@" --device auto 2>&1 | tee -a "logs/phase5_${model}_${var}.log"
}
EXT_CNN="--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001"
EXT_LSTM="--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001"

for s in 0 1 2 3 4; do run cnn    output ext_len2048 2048 $s $EXT_CNN;  done   # 1
for s in 0 1 2;     do run bilstm output ext_len2048 2048 $s $EXT_LSTM; done   # 2
for s in 3 4;       do run bilstm both   ext_len512  512  $s $EXT_LSTM; done   # 3
for s in 3 4;       do run bilstm output ext_len2048 2048 $s $EXT_LSTM; done   # 4
```

If time runs out, press Ctrl+C. Finished runs are kept. The report scripts need at least 3 seeds of a run to include it.

## Regenerate the reports (CPU, about 2 minutes)

```bash
python -m src.analysis.t13_report     # phase 3-4 table; now includes ext cap 2048 and the 5-seed BiLSTM `both` row
python -m src.analysis.t14_compare    # neural vs TF-IDF (K256 and Kfull), McNemar + CIs
```

Both rewrite one clearly named section in `reports/analysis.md` and `reports/results.md`. Check that the `t14_compare` table has CNN/BiLSTM
rows for cap 2048 (they only appear once 3 or more seeds exist).

## Push back

```bash
git add runs reports
git commit -m "Phase 5: ext cap 2048, BiLSTM both seeds 3-4, neural vs TF-IDF comparison"
git push origin HEAD
```

If the push is rejected: `git pull --rebase`, then push again (run files have unique names; the two report files may conflict if edited elsewhere,
in which case re-run the two report commands above after the pull and commit again).

## What to look at in the result

The question is whether the neural models close the gap to TF-IDF Kfull (78.7% test acc, 78.3% macro-F1) once they see the full output.
Report VAL and TEST separately. If a cap 2048 setting is better than 512, choose it by mean VAL macro-F1, not by TEST.

> Next (Phase 6, RQ3 leave-one-domain-out for the neural models): see [GPU_PHASE6.md](GPU_PHASE6.md).
