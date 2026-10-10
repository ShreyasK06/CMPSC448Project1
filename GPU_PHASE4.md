# Phase 4 GPU tasks (confirm the winning config)

Time budget: about 30 minutes. Run on the GPU machine after `git pull`. Environment setup is as in `README.md`.

## Why

Phase 3 (`GPU_PHASE3.md`) picked the tuned (`ext`) config with a 512-token output cap as the best setting for both models, chosen on
VAL macro-F1 (CNN 73.2, BiLSTM 69.6; TEST 73.1 / 69.9). Longer caps (1024, 2048) gave no VAL gain. Phase 4 checks that setting on the
`both` (prompt + output) variant and adds seeds 3-4 to tighten the confidence intervals. Nothing here changes the selection.

## Tasks (priority order; each run is skipped if its result file exists)

1. CNN `both`, seeds 0-2, plus CNN seeds 3-4 for `output` and `both` (about 6 min)
2. BiLSTM `both`, seeds 0-2 (about 13 min)
3. BiLSTM `output`, seeds 3-4, only if time remains (about 9 min)

Fixed settings: `--eval-test`, `--max-out-len 512`, tag `ext_len512`, early stopping on VAL.

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
    --tag "$tag" --max-out-len "$cap" "$@" --device auto 2>&1 | tee -a "logs/phase4_${model}_${var}.log"
}
EXT_CNN="--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001"
EXT_LSTM="--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001"

# 1) CNN
for s in 0 1 2; do run cnn both ext_len512 512 $s $EXT_CNN; done
for v in output both; do for s in 3 4; do run cnn $v ext_len512 512 $s $EXT_CNN; done; done

# 2) BiLSTM `both`
for s in 0 1 2; do run bilstm both ext_len512 512 $s $EXT_LSTM; done

# 3) extra BiLSTM `output` seeds, only if time remains
for s in 3 4; do run bilstm output ext_len512 512 $s $EXT_LSTM; done
```

If the 30 minutes run out, press Ctrl+C. Finished runs are kept and unfinished ones are simply missing.

## Push back

```bash
git add runs
git commit -m "Phase 4: ext_len512 on both variant, extra seeds"
git push origin HEAD
```

If the push is rejected: `git pull --rebase`, then push again (the run files have unique names).
