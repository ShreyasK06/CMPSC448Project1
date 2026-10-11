# Phase 6 tasks on the GPU machine (RQ3 domain holdout + RQ4 follow-ups)

Run on the GPU machine after `git pull`. Environment setup is as in `README.md`. Total budget: about 3-4 hours if everything is run
(estimates from earlier per-fit times, not measured). Everything is resumable and nothing changes an existing result file.

Two terminals are the easiest way to use the time: **Terminal A** (uses the GPU) and **Terminal B** (CPU only). Run at most ONE CPU-heavy job
at a time in Terminal B: on a 12 GB machine three jobs at once ran out of memory last time.

## What the tasks answer

- **RQ3** (do fingerprints generalize across domains?): for each of the 8 text domains, train on the other 7 (their train ids), early-stop /
  select C on those 7 domains' val ids only, and score on EVERY row of the held-out domain (400 source_ids, 2,000 rows). Vocabulary and
  TF-IDF features come from the 7 seen domains only. Tuned hyper-parameters are reused, not re-tuned. Split logic: `src/domain_holdout.py`
  (tests: `tests/test_domain_holdout.py`). Report: `python -m src.analysis.t15_domain_holdout`.
- **RQ4** (what distinguishes the LLMs?): (a) feature-group ablation of the stylometric model with length-robust lexical-diversity features
  (`src.analysis.t16_feature_groups`); (b) deleting formatting tokens, boilerplate openers and the first 32 tokens from the CNN's inputs
  (`src.analysis.t17_occlusion`).

Hyper-parameters (reused as-is):

- CNN: `--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001 --max-out-len 2048`
- BiLSTM: `--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001 --max-out-len 512`

## Tasks

| # | where | what | est. time |
|---|---|---|---|
| A1 | Terminal A (GPU) | CNN leave-one-domain-out, 8 domains x seeds 0-2 | 15-20 min |
| A2 | Terminal A (GPU) | train 3 CNN checkpoints for the occlusion study, then run `t17_occlusion` | about 5 min |
| A3 | Terminal A (GPU) | BiLSTM leave-one-domain-out, seed-major (seed 0 over all 8 domains first) | 90-110 min |
| A4 | Terminal A (GPU) | optional grouped example: train news + wiki + reviews, test poetry (CNN + BiLSTM, seeds 0-2) | about 15 min |
| B1 | Terminal B (CPU) | TF-IDF leave-one-domain-out, 256-token cap (8 folds) | 30-60 min |
| B2 | Terminal B (CPU) | `t16_feature_groups` (15 gradient-boosting grids) | 15-30 min |
| B3 | Terminal B (CPU) | TF-IDF leave-one-domain-out, no truncation (8 folds) | 30-60 min |
| B4 | Terminal B (CPU) | optional grouped TF-IDF example (news + wiki + reviews -> poetry) | about 5 min |

## Terminal A commands (GPU)

```bash
git pull
mkdir -p logs
run() {  # run <model> <tag> <heldout> <seed> <extra flags...>
  model=$1; tag=$2; dom=$3; s=$4; shift 4
  f="runs/${model}_output_${tag}_s${s}.json"
  [ -f "$f" ] && { echo "skip $f"; return; }
  python -m src.train --model "$model" --variant output --seeds "$s" --eval-test \
    --holdout-domain "$dom" --tag "$tag" "$@" --device auto 2>&1 | tee -a "logs/phase6_${model}.log"
}
CNN="--dropout 0.1 --nf 300 --emb 256 --weight-decay 0.0001 --max-out-len 2048"
LSTM="--dropout 0.5 --hid 384 --emb 128 --weight-decay 0.0001 --max-out-len 512"
DOMS="abstracts books news poetry recipes reddit reviews wiki"

# smoke test first (1 epoch, one fold; writes only to the git-ignored quick_out/)
python -m src.train --model cnn --variant output --seeds 0 --eval-test --holdout-domain poetry --tag smoke \
  --max-epochs 1 $CNN --runs-dir quick_out --device auto

# A1) CNN folds
for d in $DOMS; do for s in 0 1 2; do run cnn ho_$d $d $s $CNN; done; done

# A2) occlusion study: train 3 tuned CNNs (256-token cap) with checkpoints (git-ignored), then intervene on their inputs
python -m src.train --model cnn --variant output --seeds 0 1 2 --eval-test --tag ext_occ --dropout 0.1 --nf 300 --emb 256 \
  --weight-decay 0.0001 --save-ckpt --runs-dir quick_out --device auto
python -m src.analysis.t17_occlusion

# A3) BiLSTM folds (seed-major, so a partial run is still balanced across domains)
for s in 0 1 2; do for d in $DOMS; do run bilstm ho_$d $d $s $LSTM; done; done

# A4) optional grouped example (news + wiki + reviews -> poetry)
for s in 0 1 2; do
  run cnn    hog_poetry poetry $s $CNN  --train-domains news wiki reviews
  run bilstm hog_poetry poetry $s $LSTM --train-domains news wiki reviews
done
```

## Terminal B commands (CPU, one at a time)

```bash
git pull
mkdir -p logs
python -m src.domain_holdout                           # B1: 8 TF-IDF folds, 256-token cap -> runs/tfidf_output_ho_<domain>_K256.*
python -m src.analysis.t16_feature_groups              # B2: writes its section into reports/analysis.md and results.md
python -m src.domain_holdout --no-truncate             # B3: 8 TF-IDF folds on the full text -> ..._Kfull.*
python -m src.domain_holdout --domains poetry --train-domains news wiki reviews --tag hog_poetry   # B4 (optional)
python -m src.domain_holdout --domains poetry --train-domains news wiki reviews --tag hog_poetry --no-truncate
```

If a TF-IDF job is killed for memory, re-run just the missing domains, e.g. `python -m src.domain_holdout --domains wiki reviews`.

## When both terminals are done: build the RQ3 report, then push

```bash
python -m src.analysis.t15_domain_holdout     # reads runs/, rewrites the 'Domain holdout (RQ3)' section (about 1 minute)
git add runs reports
git commit -m "Phase 6: RQ3 domain-holdout runs and report, RQ4 feature-group and occlusion results"
git push origin HEAD
```

If the push is rejected: `git pull --rebase`, re-run the report commands above (they only rewrite their own sections) and commit again. Run
files have unique names. Checkpoints (`checkpoints/`, `*.pt`) and `quick_out/` are git-ignored and must not be pushed.

## Reading the results before writing them up

- **RQ3:** the headline is held-out accuracy on all 400 ids per domain. The paired "drop" uses only the 45-65 original test ids of that
  domain, so its CIs are wide. Check the `top pred %` column: a model that sends most of a held-out domain to one class has collapsed rather
  than transferred. Hyper-parameters were tuned on the all-domains validation split, so the held-out domains were seen once, indirectly, when
  those configs were chosen; say so.
- **RQ4 diversity:** compare the `diversity (length-robust)` rows with the plain `lexical` group and read the length-matched column: it removes
  length as a usable cue. Permutation importance shares credit between correlated groups, so use the 'only' and 'minus' rows as the cross-check.
- **RQ4 occlusion:** read each deletion against the two controls (last 32 tokens, 32 random tokens). The interventions test what this CNN
  relies on, not what distinguishes the LLMs in general.
