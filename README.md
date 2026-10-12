# Which LLM wrote this? Attributing text to its generator

CMPSC 448 Project 1. Given a response from one of five chat models, can a classifier tell which model wrote it? The data comes from the [RAID](https://github.com/liamdugan/raid) benchmark.

**Classes:** `chatgpt`, `cohere-chat`, `llama-chat`, `mistral-chat`, `mpt-chat` (chance = 20%).
**Models:** TF-IDF + logistic regression (baseline), TextCNN, BiLSTM (PyTorch).
**Data:** 8 text domains (abstracts, books, news, poetry, recipes, reddit, reviews, wiki) with 400 source prompts each, plus a held-out code domain. Splits are 70/15/15 by source prompt, so the 5 responses to one prompt never straddle train and test. See `data/DATA_CARD.md`.

## Headline results

Test split, 480 held-out source prompts, accuracy with 95% cluster-bootstrap CI.

| model (output text only) | test accuracy |
|---|---|
| TF-IDF + LR, full text | 78.67% [76.96, 80.29] |
| TF-IDF + LR, 256-token cap | 75.75% [73.96, 77.58] (grouped 5-fold CV: 76.24 ± 0.56) |
| TextCNN, tuned, 5-seed ensemble | 76.29% [74.46, 78.04] |
| BiLSTM, tuned, 5-seed ensemble | 73.75% [72.04, 75.54] |

- **Leave-one-domain-out:** accuracy falls from about 76% to about 40–46% when the test domain is unseen, and recipes collapses to chance for the neural models.
- **Prompt as input:** the prompt is identical for all five models, so it carries no signal (20.00%), and adding it to the output does not help significantly.
- **What matters most:** layout (newlines and line structure) and the amount of text the model sees. Lexical diversity adds little.

The full answers to the four research questions, with caveats, are in [`report.readme`](report.readme). Generated tables are in `reports/`.

## Repository layout

```
src/
  raid_stream.py, data.py        stream RAID, sample, split, data checks
  preprocess.py, models.py       tokenizing, vocab, TextCNN and BiLSTM
  baselines.py                   majority and TF-IDF + LR baselines
  train.py, tune.py              neural training (multi-seed, early stopping on VAL) and hyper-parameter search
  domain_holdout.py              leave-one-domain-out TF-IDF runs
  scaling.py, make_scaling_data.py   data-scaling study
  aggregate.py                   run files -> results tables
  analysis/                      statistics, error analysis, ablations, length controls, domain holdout,
                                 feature groups, occlusion
tests/                           pytest suite
data/processed/                  processed parquet files and vocabulary (tracked)
runs/                            per-seed metrics (.json) and test predictions (.npz)
reports/                         generated results, tuning configs, baselines, figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # for a CUDA GPU, install the matching torch build first
python -m pytest -q
```

The processed data ships with the repo; training reads only `data/processed/`.

## Usage

```bash
# baselines (C chosen on VAL macro-F1; --eval-test scores the test split once)
python -m src.baselines --eval-test
python -m src.baselines --eval-test --no-truncate

# hyper-parameter search (VAL only) and training
python -m src.tune --model cnn --variant output
python -m src.train --model cnn --variant output --seeds 0 1 2 --eval-test --device auto

# leave-one-domain-out
python -m src.domain_holdout                                   # TF-IDF, all 8 domains
python -m src.train --model cnn --variant output --seeds 0 --eval-test --holdout-domain poetry --tag ho_poetry

# tables and analyses
python -m src.aggregate --out reports/results.md
python -m src.analysis.t14_compare
python -m src.analysis.t15_domain_holdout
```

Variants: `output` (response only), `prompt` (prompt only), `both`. Every neural run stores its VAL and TEST predictions, and every number in the reports is labelled with the split it comes from.

## Method notes

- Hyper-parameters and early stopping use the validation split only; the test split is scored once per final model.
- Neural results are reported as per-seed mean ± sd and as a mean-probability ensemble over seeds. Differences are tested with cluster bootstrap over source prompts and exact McNemar with Holm correction.
- Results are single-split except the TF-IDF grouped cross-validation. The domain-holdout hyper-parameters were tuned on all-domain validation data, so the held-out domains were seen once, indirectly. Details are in `report.readme`.
