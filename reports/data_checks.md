# Data checks

- no source_id in >1 split: PASS
- code source_ids disjoint from text source_ids: PASS (asserted)
- (source_id, model) unique and 5 rows per source_id: PASS (asserted)
- empty generations: 0 (PASS)
- total rows: text 16000, code 2000, rq3 18000

## model x split (text)

| model | test | train | val |
|---|---|---|---|
| chatgpt | 480 | 2240 | 480 |
| cohere-chat | 480 | 2240 | 480 |
| llama-chat | 480 | 2240 | 480 |
| mistral-chat | 480 | 2240 | 480 |
| mpt-chat | 480 | 2240 | 480 |

Balanced per split (all models equal count): PASS

## domain x split (text)

| domain | test | train | val |
|---|---|---|---|
| abstracts | 275 | 1375 | 350 |
| books | 330 | 1395 | 275 |
| news | 340 | 1330 | 330 |
| poetry | 325 | 1410 | 265 |
| recipes | 300 | 1485 | 215 |
| reddit | 290 | 1385 | 325 |
| reviews | 225 | 1445 | 330 |
| wiki | 315 | 1375 | 310 |

## model x domain (text)

| model | abstracts | books | news | poetry | recipes | reddit | reviews | wiki |
|---|---|---|---|---|---|---|---|---|
| chatgpt | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| cohere-chat | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| llama-chat | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| mistral-chat | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| mpt-chat | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |

## model x split_rq3 (incl. code)

| model | test_code | test_indomain | train | val |
|---|---|---|---|---|
| chatgpt | 400 | 480 | 2240 | 480 |
| cohere-chat | 400 | 480 | 2240 | 480 |
| llama-chat | 400 | 480 | 2240 | 480 |
| mistral-chat | 400 | 480 | 2240 | 480 |
| mpt-chat | 400 | 480 | 2240 | 480 |

## source_ids per split (text)

| split | n_source_ids |
|---|---|
| test | 480 |
| train | 2240 |
| val | 480 |

## Length stats per model (text, raw, before truncation)

| model /  | char_len / mean | char_len / median | word_len / mean | word_len / median |
|---|---|---|---|---|
| chatgpt | 1718.2 | 1759.0 | 273.8 | 296.5 |
| cohere-chat | 1090.4 | 1001.0 | 185.3 | 168.0 |
| llama-chat | 1642.5 | 1735.0 | 274.6 | 287.0 |
| mistral-chat | 1244.5 | 1225.0 | 208.7 | 207.0 |
| mpt-chat | 947.2 | 842.0 | 158.2 | 144.0 |

## Length stats per model (code)

| model /  | char_len / mean | char_len / median | word_len / mean | word_len / median |
|---|---|---|---|---|
| chatgpt | 277.4 | 233.5 | 36.5 | 30.0 |
| cohere-chat | 215.1 | 140.0 | 28.6 | 19.0 |
| llama-chat | 205.6 | 134.0 | 29.0 | 19.0 |
| mistral-chat | 278.1 | 224.5 | 37.4 | 29.0 |
| mpt-chat | 213.0 | 154.5 | 27.8 | 20.0 |

## Tokens > 256 (src.preprocess.tokenize, random sample of 2000 from rq3 incl. code)

- overall: 46.7% ; median tokens 244

| model | pct_gt_256 |
|---|---|
| chatgpt | 62.3 |
| cohere-chat | 28.2 |
| llama-chat | 74.9 |
| mistral-chat | 45.5 |
| mpt-chat | 23.2 |

| domain | pct_gt_256 |
|---|---|
| abstracts | 18.6 |
| books | 55.5 |
| code | 1.9 |
| news | 68.8 |
| poetry | 32.5 |
| recipes | 79.2 |
| reddit | 21.8 |
| reviews | 56.3 |
| wiki | 77.6 |
