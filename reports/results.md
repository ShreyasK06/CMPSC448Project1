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
