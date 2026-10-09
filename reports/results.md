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
