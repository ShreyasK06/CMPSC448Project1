Values are mean ± std (sample std, ddof=1) over seeds, in %. VAL numbers drive early stopping / model selection; TEST numbers are held-out, computed at the best-val checkpoint.

| source | model | variant | tag | n seeds | val acc (%) [VAL split] | val macro-F1 (%) [VAL split] | test acc (%) [TEST split: test] | test macro-F1 (%) [TEST split: test] | test_code acc (%) [TEST split: test_code] | test_code macro-F1 (%) [TEST split: test_code] | test_indomain acc (%) [TEST split: test_indomain] | test_indomain macro-F1 (%) [TEST split: test_indomain] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | majority | both | base | 1 | 20.00 ± 0.00 | 6.67 ± 0.00 | 20.00 ± 0.00 | 6.67 ± 0.00 | - | - | - | - |
| baseline | majority | both | rq3 | 1 | 20.00 | 6.67 | - | - | 20.00 | 6.67 | 20.00 | 6.67 |
| baseline | majority | output | base | 1 | 20.00 ± 0.00 | 6.67 ± 0.00 | 20.00 ± 0.00 | 6.67 ± 0.00 | - | - | - | - |
| baseline | majority | output | rq3 | 1 | 20.00 | 6.67 | - | - | 20.00 | 6.67 | 20.00 | 6.67 |
| baseline | majority | prompt | base | 1 | 20.00 ± 0.00 | 6.67 ± 0.00 | 20.00 ± 0.00 | 6.67 ± 0.00 | - | - | - | - |
| baseline | tfidf_lr | both | base | 1 | 76.67 | 76.51 | 76.71 | 76.47 | - | - | - | - |
| baseline | tfidf_lr | both | rq3 | 1 | 76.67 | 76.51 | - | - | 19.95 | 7.28 | 76.71 | 76.47 |
| baseline | tfidf_lr | output | base | 1 | 75.62 | 75.28 | 75.75 | 75.48 | - | - | - | - |
| baseline | tfidf_lr | output | rq3 | 1 | 75.62 | 75.28 | - | - | 19.55 | 7.35 | 75.75 | 75.48 |
| baseline | tfidf_lr | prompt | base | 1 | 20.00 | 6.67 | 20.00 | 6.67 | - | - | - | - |
| baseline | tfidf_lr+char | both | base | 1 | 75.46 | 75.29 | 74.42 | 74.20 | - | - | - | - |
| baseline | tfidf_lr+char | output | base | 1 | 75.17 | 74.96 | 74.54 | 74.30 | - | - | - | - |
| baseline | tfidf_lr+char | prompt | base | 1 | 20.00 | 6.67 | 20.00 | 6.67 | - | - | - | - |
| neural | bilstm | both | base | 3 | 66.43 ± 0.38 | 66.30 ± 0.18 | 66.35 ± 0.78 | 66.18 ± 0.68 | - | - | - | - |
| neural | bilstm | both | rq3 | 3 | 66.43 ± 0.38 | 66.30 ± 0.18 | - | - | 20.23 ± 0.12 | 8.05 ± 0.39 | 66.35 ± 0.78 | 66.18 ± 0.68 |
| neural | bilstm | output | base | 3 | 65.42 ± 0.45 | 65.51 ± 0.25 | 65.72 ± 0.75 | 65.84 ± 0.57 | - | - | - | - |
| neural | bilstm | output | rq3 | 3 | 65.42 ± 0.45 | 65.51 ± 0.25 | - | - | 20.02 ± 0.33 | 7.52 ± 0.38 | 65.72 ± 0.75 | 65.84 ± 0.57 |
| neural | cnn | both | base | 3 | 68.14 ± 0.51 | 68.23 ± 0.16 | 68.19 ± 0.23 | 68.09 ± 0.25 | - | - | - | - |
| neural | cnn | both | rq3 | 3 | 68.14 ± 0.51 | 68.23 ± 0.16 | - | - | 19.93 ± 0.12 | 6.86 ± 0.26 | 68.19 ± 0.23 | 68.09 ± 0.25 |
| neural | cnn | output | base | 3 | 68.19 ± 0.54 | 68.21 ± 0.45 | 68.24 ± 0.92 | 68.12 ± 0.86 | - | - | - | - |
| neural | cnn | output | rq3 | 3 | 68.19 ± 0.54 | 68.21 ± 0.45 | - | - | 20.02 ± 0.42 | 8.49 ± 1.30 | 68.24 ± 0.92 | 68.12 ± 0.86 |

## Statistical tests on TEST split (cluster bootstrap CIs, exact McNemar)

Cluster bootstrap: resample the 480 test source_ids with replacement (10000 resamples, seed 0; all models share the same resamples), keep all 5 rows of each drawn id, percentile 95% CI. All numbers are on the [TEST split: test] (2400 rows) split.

Neural per-seed rows differ only by training seed; the +/- reported elsewhere in this file for neural models is seed-only variance (n=3, ddof=1). TF-IDF is one deterministic fit (no seed), so its uncertainty here is the bootstrap CI over test source_ids only. The two are different quantities; the bootstrap CI for a neural row is test-sampling uncertainty conditional on that seed/ensemble.

| model | variant | fit | test acc % [95% CI] | test macro-F1 % [95% CI] |
|---|---|---|---|---|
| tfidf | output | single fit | 75.75 [73.92, 77.54] | 75.48 [73.63, 77.31] |
| tfidf+char | output | single fit | 74.54 [72.71, 76.29] | 74.30 [72.46, 76.07] |
| cnn | output | s0 | 69.29 [67.29, 71.25] | 69.12 [67.11, 71.08] |
| cnn | output | s1 | 67.75 [65.62, 69.83] | 67.61 [65.49, 69.67] |
| cnn | output | s2 | 67.67 [65.58, 69.71] | 67.64 [65.56, 69.70] |
| cnn | output | 3-seed mean-prob ensemble | 71.83 [69.83, 73.83] | 71.81 [69.80, 73.78] |
| bilstm | output | s0 | 66.58 [64.67, 68.42] | 66.48 [64.57, 68.29] |
| bilstm | output | s1 | 65.38 [63.42, 67.29] | 65.67 [63.76, 67.53] |
| bilstm | output | s2 | 65.21 [63.25, 67.17] | 65.38 [63.43, 67.32] |
| bilstm | output | 3-seed mean-prob ensemble | 69.42 [67.50, 71.29] | 69.56 [67.64, 71.41] |
| tfidf | both | single fit | 76.71 [74.92, 78.46] | 76.47 [74.64, 78.24] |
| tfidf+char | both | single fit | 74.42 [72.58, 76.25] | 74.20 [72.36, 76.03] |
| cnn | both | s0 | 68.04 [66.04, 70.00] | 68.26 [66.28, 70.22] |
| cnn | both | s1 | 68.08 [66.00, 70.08] | 67.80 [65.70, 69.83] |
| cnn | both | s2 | 68.46 [66.46, 70.42] | 68.21 [66.21, 70.14] |
| cnn | both | 3-seed mean-prob ensemble | 71.29 [69.29, 73.25] | 71.17 [69.20, 73.11] |
| bilstm | both | s0 | 66.92 [65.04, 68.75] | 66.04 [64.09, 67.97] |
| bilstm | both | s1 | 66.67 [64.54, 68.71] | 66.91 [64.80, 68.92] |
| bilstm | both | s2 | 65.46 [63.33, 67.58] | 65.58 [63.46, 67.68] |
| bilstm | both | 3-seed mean-prob ensemble | 70.58 [68.62, 72.50] | 70.44 [68.47, 72.39] |

McNemar tests (exact binomial) on identical test rows, [TEST split: test]; neural = 3-seed mean-prob ensemble prediction; Holm correction across all 9 tests. Rows within a source_id are not independent, so these p-values are somewhat optimistic; details in reports/analysis.md.

| comparison (A vs B) | A-only correct | B-only correct | exact McNemar p | Holm-adj p | reject @0.05 |
|---|---|---|---|---|---|
| tfidf vs cnn [output] | 278 | 184 | 1.42e-05 | 8.52e-05 | yes |
| tfidf vs bilstm [output] | 337 | 185 | 2.82e-11 | 2.54e-10 | yes |
| cnn vs bilstm [output] | 280 | 222 | 0.0109 | 0.0544 | no |
| tfidf vs cnn [both] | 297 | 167 | 1.67e-09 | 1.17e-08 | yes |
| tfidf vs bilstm [both] | 328 | 181 | 7.22e-11 | 5.78e-10 | yes |
| cnn vs bilstm [both] | 251 | 234 | 0.468 | 0.935 | no |
| tfidf: output vs both | 46 | 69 | 0.0398 | 0.159 | no |
| cnn: output vs both | 160 | 147 | 0.493 | 0.935 | no |
| bilstm: output vs both | 138 | 166 | 0.121 | 0.364 | no |

## TF-IDF grouped cross-validation

GroupKFold(5) by source_id over train+val (2720 ids, 13600 rows); mean ± sample std (ddof=1) over folds. Within each fold C is picked from the grid on an inner grouped 80/20 split of the outer-train ids (macro-F1), then the vectorizer and LR are refit on all outer-train rows; the held-out fold is only scored. The TEST split is not used by the CV. The fixed-split columns are the single train/val/test result (C chosen on VAL), shown for comparison. No neural CV was run.

| model | variant | CV acc % [grouped 5-fold on train+val ids] | CV macro-F1 % [same] | chosen C per fold | val acc % [VAL split, fixed] | test acc % [TEST split, fixed] | test macro-F1 % [TEST split, fixed] |
|---|---|---|---|---|---|---|---|
| tfidf_lr (word 1-2gram) | output | 76.24 ± 0.56 | 75.89 ± 0.60 | 100, 10, 100, 100, 100 | 75.62 | 75.75 | 75.48 |
| tfidf_lr (word 1-2gram) | both | 76.74 ± 0.83 | 76.50 ± 0.88 | 10, 100, 10, 100, 100 | 76.67 | 76.71 | 76.47 |

## Length controls and shallow baselines (T5)

Generated by `python -m src.analysis.t5_length`. All hyperparameters chosen on VAL macro-F1; TEST scored once. Features use the OUTPUT text only (word/char counts are the stored raw `word_len`/`char_len` columns; `hit_cap` = tokeniser count >= 256, i.e. truncated by the neural/TF-IDF cap). Stylometry = 22 features: word_count, char_count, hit_cap, line_count, mean_word_len, type_token_ratio, punct_ratio, digit_ratio, upper_ratio, newline_count, blank_line_count, bullet_count, markdown_count, ends_terminal_punct, starts_boilerplate, comma_ratio, mean_sentence_len, mean_line_len, exclaim_count, colon_count, quote_ratio, starts_upper. LR uses standardised features (C grid 0.001..100); HGB grid depth {3,None} x lr {0.05,0.1} x iters {100,300}. CIs: cluster bootstrap over test source_ids (2000 resamples, seed 0). Chance = 20%.

### (a)(b) Shallow baselines vs trained models

| model | hyperparams (chosen on VAL) | VAL acc / macro-F1 % | TEST acc % [95% cluster CI] | TEST macro-F1 % [95% CI] |
|---|---|---|---|---|
| length-only LR | C=0.01 | 38.29 / 33.04 | 37.71 [36.29, 39.25] | 32.54 [31.16, 33.96] |
| stylometry LR | C=100 | 54.75 / 53.65 | 53.50 [51.75, 55.25] | 52.15 [50.36, 53.92] |
| length-only HGB | max_depth=None, lr=0.05, max_iter=100 | 51.75 / 50.90 | 51.88 [50.12, 53.67] | 51.10 [49.37, 52.85] |
| stylometry HGB | max_depth=3, lr=0.1, max_iter=300 | 64.04 / 63.54 | 65.25 [63.33, 67.08] | 64.74 [62.82, 66.60] |
| TF-IDF (word, output) | C=100 (val) | 75.62 / 75.28 | 75.75 [73.96, 77.58] | 75.48 [73.64, 77.34] |
| CNN 3-seed ens (output) | - | n/a (no val probs saved) | 71.83 [69.79, 73.83] | 71.81 [69.76, 73.78] |
| BiLSTM 3-seed ens (output) | - | n/a (no val probs saved) | 69.42 [67.50, 71.25] | 69.56 [67.69, 71.37] |

(Neural 3-seed ensembles saved only test probabilities, so no VAL number is shown for them.)

### (c) Accuracy within raw-length quintiles [TEST split: test]

Quintile edges are TEST quantiles of raw word count (descriptive only).

| quintile | raw words | n | TF-IDF acc % | CNN acc % | BiLSTM acc % | stylo-HGB acc % | length-only LR acc % | class mix % (chatgpt/cohere/llama/mistral/mpt) |
|---|---|---|---|---|---|---|---|---|
| Q1 | 1-128 | 475 | 74.1 | 66.1 | 66.5 | 57.5 | 37.9 | 12 / 30 / 2 / 18 / 38 |
| Q2 | 129-187 | 479 | 74.3 | 72.7 | 68.5 | 58.5 | 28.2 | 15 / 27 / 9 / 24 / 25 |
| Q3 | 188-248 | 479 | 72.0 | 69.5 | 68.3 | 62.0 | 23.2 | 14 / 20 / 16 / 24 / 25 |
| Q4 | 249-318 | 484 | 77.9 | 73.8 | 71.5 | 66.7 | 38.6 | 14 / 12 / 40 / 24 / 10 |
| Q5 | 319-447 | 483 | 80.3 | 77.0 | 72.3 | 81.4 | 60.5 | 44 / 12 / 32 / 10 / 1 |

### (d) Length-matched TEST subset [TEST split: test]

Rows resampled so every class has the same histogram over 10 equal-frequency raw-word-count bins (min class count per bin, no replacement, seed 0): 820 rows (164, 164, 164, 164, 164 per class). Models are NOT retrained; their predictions on the subset are scored.

| model | full TEST acc % | length-matched TEST acc % (seed 0) [95% cluster CI] | mean +/- sd over 20 resample seeds |
|---|---|---|---|
| length-only LR | 37.71 | 22.68 [19.80, 25.48] | 22.61 +/- 0.43 |
| length-only HGB | 51.88 | 39.39 [36.17, 42.76] | 40.52 +/- 0.84 |
| stylometry LR | 53.50 | 46.10 [42.89, 49.32] | 45.74 +/- 1.26 |
| stylometry HGB | 65.25 | 60.24 [56.91, 63.61] | 60.51 +/- 1.10 |
| TF-IDF (word, output) | 75.75 | 74.39 [71.23, 77.68] | 74.01 +/- 1.08 |
| CNN 3-seed ens (output) | 71.83 | 70.37 [67.15, 73.69] | 70.61 +/- 1.02 |
| BiLSTM 3-seed ens (output) | 69.42 | 67.93 [64.63, 71.13] | 68.01 +/- 0.92 |

Below vs at the 256-token cap [TEST split: test]. Share of each class's test rows that hit the cap (chatgpt/cohere/llama/mistral/mpt, %): 69 / 35 / 87 / 50 / 27.

| model | acc % below cap (<256 tokens, n=1118) | acc % at cap (>=256 tokens, n=1282) |
|---|---|---|
| length-only LR | 32.29 | 42.43 |
| length-only HGB | 47.14 | 56.01 |
| stylometry LR | 46.60 | 59.52 |
| stylometry HGB | 58.41 | 71.22 |
| TF-IDF (word, output) | 73.26 | 77.93 |
| CNN 3-seed ens (output) | 68.34 | 74.88 |
| BiLSTM 3-seed ens (output) | 66.91 | 71.61 |

## TF-IDF ablations (T6)

Generated by `python -m src.analysis.t6_ablate`. TF-IDF word 1-2gram + LR on the OUTPUT text; train, val and test are processed identically in every row; C chosen on VAL macro-F1 per ablation (grid 0.01..1000); vectorizer fit on train only. K = keep first K tokens (tokeniser tokens, <NL> counts as a token); `K256` is the base model (predictions identical to runs/tfidf_output_base.npz), `Kfull` = no truncation. lowercase / format_norm / self_id_masked use the base 256-token cap. format_norm = strip markdown headings/quotes/bullets/numbered-list markers and `*_~` emphasis, newlines to spaces, collapse whitespace (no <NL>). self_id_masked = delete matches of `\bcohere\b`, `\bas an ai(?: language model)?\b`, `^\s*sure[!,.]?\s+here(?:'s| is| are)?` (case-insensitive). Split for all metrics: [TEST split: test], CIs by cluster bootstrap over source_ids; `*` = Holm-significant at 0.05. Figure: reports/figures/tfidf_acc_vs_K.png (`full` plotted at x=512).

| ablation | C (VAL) | VAL acc / macro-F1 % | TEST acc % [95% CI] | TEST macro-F1 % [95% CI] | McNemar vs base K256 (base-only/ablation-only correct; exact p; Holm over 7) | n features |
|---|---|---|---|---|---|---|
| K32 | 10 | 54.62 / 54.48 | 55.46 [53.33, 57.71] | 55.18 [53.07, 57.48] | 686/199; p=2.51e-63; Holm p=1.75e-62 * | 51635 |
| K64 | 10 | 61.46 / 61.33 | 59.71 [57.50, 61.96] | 59.63 [57.45, 61.85] | 562/177; p=1.68e-47; Holm p=1.01e-46 * | 87868 |
| K128 | 10 | 69.62 / 69.34 | 69.83 [67.75, 71.83] | 69.66 [67.56, 71.68] | 273/131; p=1.36e-12; Holm p=6.8e-12 * | 140619 |
| K256 | 100 | 75.62 / 75.28 | 75.75 [73.96, 77.58] | 75.48 [73.64, 77.34] | (base) | 199726 |
| Kfull | 100 | 78.96 / 78.67 | 78.67 [76.96, 80.29] | 78.25 [76.50, 79.96] | 52/122; p=1.15e-07; Holm p=4.61e-07 * | 200000 |
| lowercase | 100 | 75.33 / 75.01 | 75.62 [73.79, 77.42] | 75.32 [73.49, 77.13] | 60/57; p=0.853; Holm p=1 | 190053 |
| format_norm | 10 | 76.08 / 75.84 | 76.21 [74.46, 77.88] | 75.88 [74.09, 77.59] | 79/90; p=0.442; Holm p=1 | 200000 |
| self_id_masked | 100 | 76.04 / 75.76 | 75.96 [74.21, 77.75] | 75.71 [73.88, 77.53] | 20/25; p=0.551; Holm p=1 | 199754 |

## Error analysis (T4)

### Per-domain accuracy [TEST split: test]

| domain | n | TF-IDF acc % | CNN-ens acc % | BiLSTM-ens acc % | stylometry-HGB acc % |
|---|---|---|---|---|---|
| reviews | 225 | 89.3 | 87.6 | 80.0 | 67.6 |
| recipes | 300 | 88.0 | 92.0 | 84.3 | 78.0 |
| books | 330 | 79.4 | 75.5 | 71.8 | 67.6 |
| abstracts | 275 | 78.2 | 75.6 | 75.3 | 60.7 |
| poetry | 325 | 74.2 | 66.2 | 61.5 | 60.9 |
| reddit | 290 | 71.4 | 65.5 | 66.2 | 56.6 |
| wiki | 315 | 68.6 | 62.5 | 63.5 | 65.7 |
| news | 340 | 62.4 | 56.5 | 57.9 | 65.0 |

### TF-IDF vs CNN disagreement [TEST split: test]

| outcome | rows (of 2400) |
|---|---|
| both right | 1540 |
| TF-IDF only right | 278 |
| CNN-ens only right | 184 |
| both wrong | 398 |

(Full tables: reports/analysis.md.)

## RQ3 diagnosis: text-trained models on code (T7)

[TEST split: test_code] unless noted; full tables in reports/analysis.md.

| model (trained on text) | % pred chatgpt | % pred cohere-chat | % pred llama-chat | % pred mistral-chat | % pred mpt-chat | code acc % |
|---|---|---|---|---|---|---|
| TF-IDF (word) | 0.0 | 96.3 | 0.1 | 3.4 | 0.2 | 19.55 |
| CNN 3-seed ens | 0.4 | 95.5 | 0.0 | 4.0 | 0.0 | 19.70 |
| BiLSTM 3-seed ens | 0.2 | 99.3 | 0.0 | 0.3 | 0.1 | 20.10 |
| stylometry-HGB (diagnostic) | 0.1 | 93.8 | 0.0 | 5.8 | 0.3 | 19.55 |

| setting | acc % | macro-F1 % |
|---|---|---|
| TF-IDF trained on TEXT, evaluated on code (no code in training) [TEST split: test_code] | 19.55 | 7.35 |
| TF-IDF trained on TEXT, evaluated on in-domain text [TEST split: test] | 75.75 | 75.48 |
| CEILING: TF-IDF trained AND tested on code, grouped 5-fold CV over the 400 code source_ids (mean +/- sd over folds; C per fold 1000, 1000, 10, 10, 10) [CV on code] | 49.40 +/- 2.27 | 49.25 +/- 2.35 |
| reference: TF-IDF grouped 5-fold CV on text train+val [CV on text] | 76.24 +/- 0.56 | 75.89 +/- 0.60 |

## Secondary: learning curve (beyond spec)

**Secondary / beyond the project spec.** Does more training data help? Train subsets are nested seeded subsamples (seed 42) of the 2,240 train source_ids of `raid_main`; `expanded` = `data/processed/raid_scaled.parquet` (6,240 train ids = the 2,240 plus 4,000 new ids, 2.79x; see the DATA_CARD addendum). VAL and TEST are the identical 480 + 480 source_ids in every row. Nothing is tuned on TEST: TF-IDF picks C on [VAL split] macro-F1 at every point; the CNN uses one fixed config (T9-chosen, fallback: stage-1 best) at every point. One subsample draw per fraction, so the 25%/50% points carry draw-to-draw variance that the TEST CIs (test sampling only) do not include. This varies training-set size within one domain pool; it is not a distribution-shift test.

| model | train set | train ids / rows | selected | [VAL split] acc / macro-F1 % | [TEST split: test] acc / macro-F1 % | TEST acc % [95% CI] (TF-IDF: single fit; CNN: 3-seed ensemble) | McNemar vs 100% (100%-only/point-only correct; exact p; Holm per model) |
|---|---|---|---|---|---|---|---|
| TF-IDF+LR | 25% | 560 / 2800 | C=100 | 67.17 / 66.63 | 67.71 / 67.15 | 67.71 [65.71, 69.75] | 288/95; p=1.07e-23; Holm p=3.2e-23 * |
| TF-IDF+LR | 50% | 1120 / 5600 | C=100 | 72.04 / 71.62 | 71.71 / 71.26 | 71.71 [69.75, 73.58] | 173/76; p=7.21e-10; Holm p=1.44e-09 * |
| TF-IDF+LR | 100% | 2240 / 11200 | C=100 | 75.62 / 75.28 | 75.75 / 75.48 | 75.75 [73.96, 77.58] | (reference) |
| TF-IDF+LR | expanded | 6240 / 31200 | C=100 | 80.88 / 80.78 | 79.96 / 79.81 | 79.96 [78.29, 81.62] | 99/200; p=5.29e-09; Holm p=5.29e-09 * |
| CNN (fixed config) | 100% | 2240 / 11200 | 3 seeds | 70.19 ± 1.15 / 70.05 ± 0.95 | 69.86 ± 0.58 / 69.54 ± 0.27 | 72.54 [70.54, 74.33] | - |
| CNN (fixed config) | expanded | 6240 / 31200 | 3 seeds | 77.64 ± 0.69 / 77.65 ± 0.42 | 78.11 ± 0.77 / 78.09 ± 0.67 | 80.21 [78.62, 81.75] | 94/278; p=3.13e-22; Holm p=3.13e-22 * |

CNN rows: VAL/TEST columns are mean ± sample std over seeds; the CI is the cluster bootstrap of the seeds' mean-probability ensemble. TF-IDF is one deterministic fit per point.

(figure skipped: No module named 'matplotlib')

## Ablations

Generated by `python -m src.analysis.t8_report`. Neural ablations, variant `output` only, hyper-parameters fixed to the stage-1 VAL-tuned base config (`reports/tuning/{cnn,bilstm}_output.json`), 3 seeds each, TEST evaluated once at the best-VAL checkpoint; nothing is selected on TEST. Each ablation changes ONE thing relative to `base` (the vocab is rebuilt on TRAIN under each ablation): `abl_lc` lowercases text, `abl_lenN` caps the output at N tokens (base: 256; prompt cap unchanged), `abl_fmt` format-normalises the output (same `format_normalise` as the TF-IDF ablation T6: strip markdown/bullets/numbering/emphasis, no <NL>, whitespace collapsed). `base_vp` is the base config re-run with the new code path that also stores VAL probabilities; it replaces nothing and is shown only to size seed/implementation noise (its differences from `base` are the noise floor against which ablation deltas should be read). Columns are mean +/- sample std (ddof=1) over seeds; the ensemble averages the seeds' class probabilities. CIs: cluster bootstrap over the 480 test source_ids (2,000 resamples, shared across rows, paired for the difference column). `*` = Holm-significant at 0.05 (family = the 10 comparisons in this table). Rows within a source_id are not independent, so McNemar p-values are somewhat optimistic.

| model | run (tag) | n seeds | [VAL split] acc / macro-F1 % | [TEST split: test] acc / macro-F1 % | 3-seed ensemble TEST acc % [95% CI] | ensemble TEST macro-F1 % [95% CI] | Δ ensemble TEST acc vs base, pp [95% paired CI] | McNemar vs base ensemble (base-only/run-only correct; exact p; Holm over this table) |
|---|---|---|---|---|---|---|---|---|
| cnn | base (reference) (`base`) | 3 | 68.19 ± 0.54 / 68.21 ± 0.45 | 68.24 ± 0.92 / 68.12 ± 0.86 | 71.83 [69.79, 73.83] | 71.81 [69.76, 73.78] | (base) | (base) |
| bilstm | base (reference) (`base`) | 3 | 65.42 ± 0.45 / 65.51 ± 0.25 | 65.72 ± 0.75 / 65.84 ± 0.57 | 69.42 [67.50, 71.25] | 69.56 [67.69, 71.37] | (base) | (base) |
| cnn | base re-run, val probs saved (should match base within seed noise) (`base_vp`) | 3 | 68.19 ± 0.54 / 68.21 ± 0.45 | 68.24 ± 0.92 / 68.12 ± 0.86 | 71.83 [69.79, 73.83] | 71.81 [69.76, 73.78] | +0.00 [+0.00, +0.00] | 0/0; p=1; Holm p=1 |
| bilstm | base re-run, val probs saved (should match base within seed noise) (`base_vp`) | 3 | 65.42 ± 0.45 / 65.51 ± 0.25 | 65.72 ± 0.75 / 65.84 ± 0.57 | 69.42 [67.50, 71.25] | 69.56 [67.69, 71.37] | +0.00 [+0.00, +0.00] | 0/0; p=1; Holm p=1 |
| cnn | lowercase (vocab rebuilt lowercase) (`abl_lc`) | 3 | 66.79 ± 0.78 / 66.82 ± 1.10 | 67.69 ± 1.97 / 67.77 ± 2.19 | 71.75 [69.75, 73.75] | 71.82 [69.85, 73.78] | -0.08 [-1.46, +1.33] | 165/163; p=0.956; Holm p=1 |
| bilstm | lowercase (vocab rebuilt lowercase) (`abl_lc`) | 3 | 63.64 ± 0.52 / 63.71 ± 0.60 | 63.83 ± 1.52 / 63.90 ± 1.64 | 66.96 [64.92, 68.96] | 67.04 [65.03, 69.01] | -2.46 [-4.13, -0.92] | 222/163; p=0.00307; Holm p=0.0184 * |
| cnn | output cap 64 tokens (`abl_len64`) | 3 | 55.89 ± 0.92 / 55.67 ± 0.25 | 55.43 ± 0.78 / 55.26 ± 0.20 | 59.17 [57.21, 61.33] | 59.00 [57.03, 61.14] | -12.67 [-14.62, -10.54] | 497/193; p=8.62e-32; Holm p=8.62e-31 * |
| cnn | output cap 128 tokens (`abl_len128`) | 3 | 60.75 ± 1.66 / 60.67 ± 1.30 | 61.90 ± 0.94 / 61.77 ± 0.72 | 64.62 [62.33, 66.88] | 64.38 [62.09, 66.61] | -7.21 [-9.00, -5.38] | 333/160; p=4.98e-15; Holm p=4.48e-14 * |
| cnn | output cap 512 tokens (`abl_len512`) | 3 | 71.68 ± 0.78 / 71.17 ± 0.83 | 70.90 ± 0.83 / 70.30 ± 0.66 | 73.62 [71.83, 75.50] | 73.04 [71.21, 74.99] | +1.79 [+0.25, +3.29] | 147/190; p=0.022; Holm p=0.11 |
| bilstm | output cap 128 tokens (`abl_len128`) | 3 | 60.25 ± 0.29 / 60.31 ± 0.48 | 60.56 ± 1.81 / 60.57 ± 2.09 | 65.00 [62.83, 67.17] | 64.95 [62.84, 67.08] | -4.42 [-6.17, -2.71] | 298/192; p=1.93e-06; Holm p=1.54e-05 * |
| bilstm | output cap 512 tokens (`abl_len512`) | 3 | 68.43 ± 0.16 / 68.72 ± 0.34 | 68.26 ± 0.78 / 68.49 ± 0.33 | 72.21 [70.21, 74.13] | 72.40 [70.47, 74.33] | +2.79 [+1.17, +4.33] | 149/216; p=0.000534; Holm p=0.00373 * |
| cnn | format-normalised input (no markdown/bullets, no <NL>, whitespace collapsed) (`abl_fmt`) | 3 | 66.71 ± 1.17 / 66.66 ± 0.95 | 67.22 ± 0.54 / 66.94 ± 0.52 | 70.46 [68.50, 72.38] | 70.05 [68.02, 72.01] | -1.38 [-3.00, +0.33] | 207/174; p=0.101; Holm p=0.404 |

## Extended tuning

Generated by `python -m src.analysis.t8_report`. Stage 1 = whole grid, seed 0; stage 2 = top-3 stage-1 configs re-run with seeds 0,1,2 (seed 0 reused); the chosen config has the best MEAN [VAL split] macro-F1. Selection used VAL only; TEST was not touched until the final `ext` runs (3 seeds, variants `output` and `both`, same config for both, TEST at the best-VAL checkpoint). The old grid is the 'base' tuning (`reports/tuning/*_output.json`); the base runs used weight_decay 1e-5 and no embedding dropout.

### CNN grid (variant output) [VAL split]

Stage 1 (12 configs, seed 0), embedding dropout off.

| rank | config | val acc % | val macro-F1 % | best epoch |
|---|---|---|---|---|
| 1 | dropout=0.1, nf=300, emb=256, weight_decay=0.0001 | 71.33 | 70.80 | 9 |
| 2 | dropout=0.3, nf=300, emb=256, weight_decay=0.0001 | 70.58 | 70.38 | 12 |
| 3 | dropout=0.2, nf=200, emb=256, weight_decay=0.0001 | 70.58 | 69.69 | 16 |
| 4 | dropout=0.1, nf=300, emb=256, weight_decay=0 | 69.29 | 69.57 | 6 |
| 5 | dropout=0.2, nf=300, emb=256, weight_decay=0 | 69.04 | 68.99 | 5 |
| 6 | dropout=0.3, nf=200, emb=256, weight_decay=0.0001 | 69.29 | 68.91 | 13 |
| 7 | dropout=0.1, nf=200, emb=256, weight_decay=0.0001 | 69.42 | 68.77 | 5 |
| 8 | dropout=0.1, nf=200, emb=256, weight_decay=0 | 68.96 | 68.63 | 9 |
| 9 | dropout=0.2, nf=200, emb=256, weight_decay=0 | 68.54 | 68.46 | 5 |
| 10 | dropout=0.3, nf=300, emb=256, weight_decay=0 | 68.00 | 68.27 | 5 |
| 11 | dropout=0.2, nf=300, emb=256, weight_decay=0.0001 | 66.88 | 67.49 | 5 |
| 12 | dropout=0.3, nf=200, emb=256, weight_decay=0 | 67.42 | 67.19 | 6 |

Stage 2 (top configs, seeds 0,1,2)

| config | val macro-F1 % by seed | mean ± std | val acc % (mean) |
|---|---|---|---|
| dropout=0.1, nf=300, emb=256, weight_decay=0.0001 | 70.80, 68.98, 70.37 | 70.05 ± 0.95 | 70.19 |
| dropout=0.2, nf=200, emb=256, weight_decay=0.0001 | 69.69, 69.90, 70.41 | 70.00 ± 0.37 | 70.40 |
| dropout=0.3, nf=300, emb=256, weight_decay=0.0001 | 70.38, 69.17, 66.36 | 68.63 ± 2.06 | 69.14 |

**Chosen (cnn):** dropout=0.1, nf=300, emb=256, weight_decay=0.0001 (mean val macro-F1 70.05).

### BILSTM grid (variant output) [VAL split]

Stage 1 (6 configs, seed 0), embedding dropout off.

| rank | config | val acc % | val macro-F1 % | best epoch |
|---|---|---|---|---|
| 1 | dropout=0.5, hid=256, emb=128, weight_decay=0.0001 | 66.50 | 66.41 | 13 |
| 2 | dropout=0.5, hid=384, emb=128, weight_decay=0.0001 | 66.54 | 66.02 | 11 |
| 3 | dropout=0.3, hid=256, emb=128, weight_decay=0 | 64.38 | 64.54 | 5 |
| 4 | dropout=0.5, hid=384, emb=128, weight_decay=0 | 64.92 | 64.47 | 5 |
| 5 | dropout=0.5, hid=256, emb=128, weight_decay=0 | 63.88 | 63.58 | 5 |
| 6 | dropout=0.3, hid=256, emb=128, weight_decay=0.0001 | 63.58 | 63.51 | 6 |

Stage 2 (top configs, seeds 0,1,2)

| config | val macro-F1 % by seed | mean ± std | val acc % (mean) |
|---|---|---|---|
| dropout=0.5, hid=384, emb=128, weight_decay=0.0001 | 66.02, 65.38, 67.36 | 66.25 ± 1.01 | 66.53 |
| dropout=0.5, hid=256, emb=128, weight_decay=0.0001 | 66.41, 66.03, 64.79 | 65.74 ± 0.84 | 66.17 |
| dropout=0.3, hid=256, emb=128, weight_decay=0 | 64.54, 65.71, 62.67 | 64.31 ± 1.53 | 64.22 |

**Chosen (bilstm):** dropout=0.5, hid=384, emb=128, weight_decay=0.0001 (mean val macro-F1 66.25).

### Final runs: `ext` config vs base

Per-seed columns are mean ± sample std (ddof=1) over seeds; ensemble = mean of the seeds' probabilities. Holm family = the 4 ext-vs-base comparisons here. Splits as labelled.

| model | variant | run | n seeds | [VAL split] acc / macro-F1 % | [TEST split: test] acc / macro-F1 % | ensemble TEST acc % [95% CI] | ensemble TEST macro-F1 % [95% CI] | Δ ensemble TEST acc vs base, pp [95% paired CI] | McNemar vs base ensemble (base-only/ext-only correct; exact p; Holm) |
|---|---|---|---|---|---|---|---|---|---|
| cnn | output | base | 3 | 68.19 ± 0.54 / 68.21 ± 0.45 | 68.24 ± 0.92 / 68.12 ± 0.86 | 71.83 [69.79, 73.83] | 71.81 [69.76, 73.78] | (base) | (base) |
| cnn | output | ext | 3 | 70.19 ± 1.15 / 70.05 ± 0.95 | 69.86 ± 0.58 / 69.54 ± 0.27 | 72.54 [70.54, 74.33] | 72.22 [70.22, 74.09] | +0.71 [-0.75, +2.21] | 153/170; p=0.373; Holm p=1 |
| cnn | both | base | 3 | 68.14 ± 0.51 / 68.23 ± 0.16 | 68.19 ± 0.23 / 68.09 ± 0.25 | 71.29 [69.33, 73.17] | 71.17 [69.23, 73.07] | (base) | (base) |
| cnn | both | ext | 3 | 69.92 ± 0.98 / 70.00 ± 0.97 | 70.18 ± 1.04 / 70.10 ± 0.74 | 73.08 [71.12, 74.88] | 72.93 [70.98, 74.75] | +1.79 [+0.46, +3.12] | 134/177; p=0.0171; Holm p=0.0684 |
| bilstm | output | base | 3 | 65.42 ± 0.45 / 65.51 ± 0.25 | 65.72 ± 0.75 / 65.84 ± 0.57 | 69.42 [67.50, 71.25] | 69.56 [67.69, 71.37] | (base) | (base) |
| bilstm | output | ext | 3 | 66.53 ± 0.85 / 66.25 ± 1.01 | 66.40 ± 0.65 / 66.18 ± 1.00 | 69.12 [67.12, 71.21] | 69.01 [67.00, 71.07] | -0.29 [-1.75, +1.21] | 156/149; p=0.731; Holm p=1 |
| bilstm | both | base | 3 | 66.43 ± 0.38 / 66.30 ± 0.18 | 66.35 ± 0.78 / 66.18 ± 0.68 | 70.58 [68.54, 72.58] | 70.44 [68.40, 72.45] | (base) | (base) |
| bilstm | both | ext | 3 | 67.72 ± 0.65 / 67.23 ± 0.58 | 66.79 ± 0.22 / 66.16 ± 0.29 | 70.67 [68.75, 72.46] | 70.02 [68.05, 71.88] | +0.08 [-1.46, +1.58] | 178/180; p=0.958; Holm p=1 |

## Output length and tuned config (phase 3-4)

Generated by `python -m src.analysis.t13_report`. Variants `output` and `both`; every row is trained on the TRAIN split, early-stopped on [VAL split], and evaluated once on [TEST split: test] at the best-VAL checkpoint. The cap is the output-token cap (prompt cap stays 64); `base` = stage-1 tuned config, `ext` = T9 extended-tuning config (`reports/tuning/*_output_ext.json`), both selected on VAL only. The best setting (ext, cap 512) was chosen on mean VAL macro-F1 from the `output` runs before the `both` runs and seeds 3-4 existed. Columns are mean ± sample std over ALL seeds listed in the n-seeds column (the per-seed columns therefore include seeds 3-4 where n=5). The ensemble, CI and paired-difference columns use seeds 0-2 for every row so rows are comparable; the last row of a block adds seeds 3-4 (5-seed ensemble, compared with the 3-seed reference). CIs: cluster bootstrap over the 480 test source_ids (2,000 resamples, shared, paired for the difference). Rows within a source_id are not independent, so McNemar p-values are somewhat optimistic. These are single-split numbers: no cross-validated number exists for the neural models (TF-IDF grouped CV is in its own section above), and the [VAL split] column is selection data, not a generalization estimate.

| model | variant | run | n seeds | [VAL split] acc / macro-F1 % | [TEST split: test] acc / macro-F1 % | 3-seed ensemble TEST acc % [95% CI] | ensemble TEST macro-F1 % [95% CI] | Δ ensemble TEST acc vs reference, pp [95% paired CI] | McNemar vs reference (ref-only/run-only correct; exact p; Holm) |
|---|---|---|---|---|---|---|---|---|---|
| cnn | output | base, cap 256 (reference) | 3 | 68.19 ± 0.54 / 68.21 ± 0.45 | 68.24 ± 0.92 / 68.12 ± 0.86 | 71.83 [69.79, 73.83] | 71.81 [69.76, 73.78] | (reference) | - |
| cnn | output | base, cap 512 | 3 | 71.68 ± 0.78 / 71.17 ± 0.83 | 70.90 ± 0.83 / 70.30 ± 0.66 | 73.62 [71.83, 75.50] | 73.04 [71.21, 74.99] | +1.79 [+0.25, +3.29] | 147/190; p=0.022; Holm p=0.088 |
| cnn | output | base, cap 1024 | 3 | 70.24 ± 1.39 / 69.92 ± 1.34 | 70.26 ± 0.73 / 69.73 ± 0.69 | 73.33 [71.54, 75.17] | 72.81 [70.98, 74.74] | +1.50 [-0.04, +3.04] | 167/203; p=0.0687; Holm p=0.137 |
| cnn | output | base, cap 2048 | 3 | 71.10 ± 0.54 / 71.06 ± 0.51 | 70.54 ± 0.31 / 70.24 ± 0.42 | 73.42 [71.62, 75.33] | 73.17 [71.32, 75.06] | +1.58 [+0.12, +3.08] | 152/190; p=0.0453; Holm p=0.136 |
| cnn | output | ext, cap 256 | 3 | 70.19 ± 1.15 / 70.05 ± 0.95 | 69.86 ± 0.58 / 69.54 ± 0.27 | 72.54 [70.54, 74.33] | 72.22 [70.22, 74.09] | +0.71 [-0.75, +2.21] | 153/170; p=0.373; Holm p=0.373 |
| cnn | output | ext, cap 512 | 5 | 72.54 ± 1.51 / 72.16 ± 1.62 | 72.61 ± 1.22 / 72.12 ± 1.54 | 75.25 [73.58, 76.92] | 74.94 [73.23, 76.68] | +3.42 [+1.83, +5.04] | 149/231; p=3.04e-05; Holm p=0.000152 * |
| cnn | output | ext, cap 1024 | 3 | 72.49 ± 1.31 / 72.77 ± 1.23 | 72.35 ± 0.06 / 72.48 ± 0.18 | 75.25 [73.46, 76.96] | 75.32 [73.55, 77.03] | +3.42 [+1.88, +4.92] | 145/227; p=2.49e-05; Holm p=0.000149 * |
| cnn | output | ext, cap 512, all seeds (5-seed ensemble) | 5 | 72.54 ± 1.51 / 72.16 ± 1.62 | 72.61 ± 1.22 / 72.12 ± 1.54 | 75.42 [73.71, 77.17] | 75.05 [73.23, 76.84] | +3.58 [+2.00, +5.21] | 143/229; p=9.62e-06; Holm p=6.73e-05 * |
| cnn | both | base, cap 256 (reference) | 3 | 68.14 ± 0.51 / 68.23 ± 0.16 | 68.19 ± 0.23 / 68.09 ± 0.25 | 71.29 [69.33, 73.17] | 71.17 [69.23, 73.07] | (reference) | - |
| cnn | both | ext, cap 256 | 3 | 69.92 ± 0.98 / 70.00 ± 0.97 | 70.18 ± 1.04 / 70.10 ± 0.74 | 73.08 [71.12, 74.88] | 72.93 [70.98, 74.75] | +1.79 [+0.46, +3.12] | 134/177; p=0.0171; Holm p=0.0171 * |
| cnn | both | ext, cap 512 | 5 | 72.65 ± 1.53 / 72.42 ± 1.64 | 72.67 ± 1.07 / 72.31 ± 1.24 | 75.62 [73.92, 77.38] | 75.25 [73.51, 77.07] | +4.33 [+2.83, +5.92] | 133/237; p=7.08e-08; Holm p=1.42e-07 * |
| cnn | both | ext, cap 512, all seeds (5-seed ensemble) | 5 | 72.65 ± 1.53 / 72.42 ± 1.64 | 72.67 ± 1.07 / 72.31 ± 1.24 | 76.04 [74.29, 77.75] | 75.71 [73.87, 77.45] | +4.75 [+3.25, +6.21] | 117/231; p=9.88e-10; Holm p=2.96e-09 * |
| bilstm | output | base, cap 256 (reference) | 3 | 65.42 ± 0.45 / 65.51 ± 0.25 | 65.72 ± 0.75 / 65.84 ± 0.57 | 69.42 [67.50, 71.25] | 69.56 [67.69, 71.37] | (reference) | - |
| bilstm | output | base, cap 512 | 3 | 68.43 ± 0.16 / 68.72 ± 0.34 | 68.26 ± 0.78 / 68.49 ± 0.33 | 72.21 [70.21, 74.13] | 72.40 [70.47, 74.33] | +2.79 [+1.17, +4.33] | 149/216; p=0.000534; Holm p=0.00107 * |
| bilstm | output | base, cap 1024 | 3 | 68.68 ± 0.46 / 68.70 ± 0.22 | 68.82 ± 0.39 / 68.73 ± 0.64 | 72.42 [70.67, 74.21] | 72.38 [70.61, 74.16] | +3.00 [+1.54, +4.46] | 137/209; p=0.000128; Holm p=0.000385 * |
| bilstm | output | base, cap 2048 | 3 | 68.51 ± 0.70 / 68.53 ± 0.46 | 69.67 ± 0.42 / 69.64 ± 0.63 | 73.42 [71.58, 75.21] | 73.39 [71.57, 75.22] | +4.00 [+2.67, +5.38] | 117/213; p=1.4e-07; Holm p=9.79e-07 * |
| bilstm | output | ext, cap 256 | 3 | 66.53 ± 0.85 / 66.25 ± 1.01 | 66.40 ± 0.65 / 66.18 ± 1.00 | 69.12 [67.12, 71.21] | 69.01 [67.00, 71.07] | -0.29 [-1.75, +1.21] | 156/149; p=0.731; Holm p=0.731 |
| bilstm | output | ext, cap 512 | 5 | 70.15 ± 1.04 / 70.17 ± 1.25 | 70.49 ± 0.87 / 70.41 ± 1.11 | 72.79 [71.00, 74.50] | 72.45 [70.64, 74.17] | +3.38 [+1.83, +4.96] | 158/239; p=5.62e-05; Holm p=0.000225 * |
| bilstm | output | ext, cap 1024 | 3 | 69.43 ± 1.73 / 69.35 ± 1.98 | 69.90 ± 1.87 / 69.63 ± 2.12 | 72.96 [71.17, 74.79] | 72.81 [70.94, 74.69] | +3.54 [+1.88, +5.21] | 152/237; p=1.91e-05; Holm p=9.56e-05 * |
| bilstm | output | ext, cap 512, all seeds (5-seed ensemble) | 5 | 70.15 ± 1.04 / 70.17 ± 1.25 | 70.49 ± 0.87 / 70.41 ± 1.11 | 73.75 [72.04, 75.54] | 73.66 [71.93, 75.43] | +4.33 [+2.75, +5.88] | 143/247; p=1.56e-07; Holm p=9.79e-07 * |
| bilstm | both | base, cap 256 (reference) | 3 | 66.43 ± 0.38 / 66.30 ± 0.18 | 66.35 ± 0.78 / 66.18 ± 0.68 | 70.58 [68.54, 72.58] | 70.44 [68.40, 72.45] | (reference) | - |
| bilstm | both | ext, cap 256 | 3 | 67.72 ± 0.65 / 67.23 ± 0.58 | 66.79 ± 0.22 / 66.16 ± 0.29 | 70.67 [68.75, 72.46] | 70.02 [68.05, 71.88] | +0.08 [-1.46, +1.58] | 178/180; p=0.958; Holm p=0.958 |
| bilstm | both | ext, cap 512 | 3 | 69.03 ± 2.32 / 68.53 ± 2.51 | 69.04 ± 2.88 / 68.37 ± 3.10 | 72.33 [70.62, 74.00] | 71.70 [69.88, 73.46] | +1.75 [+0.12, +3.25] | 182/224; p=0.0417; Holm p=0.0835 |
