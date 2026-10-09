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
