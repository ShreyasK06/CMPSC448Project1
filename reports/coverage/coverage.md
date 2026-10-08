# Coverage tables (attack == none; row counts over ALL rows of the full streamed CSVs)

## (a) train split: model x (decoding, repetition_penalty) x domain

| model | decoding | repetition_penalty | abstracts | books | news | poetry | recipes | reddit | reviews | wiki | total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chatgpt | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| chatgpt | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| cohere | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| cohere | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| cohere-chat | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| cohere-chat | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt2 | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt2 | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt2 | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt2 | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt3 | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt3 | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt4 | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| gpt4 | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| human |  |  | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| llama-chat | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| llama-chat | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| llama-chat | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| llama-chat | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral-chat | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral-chat | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral-chat | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral-chat | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt-chat | greedy | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt-chat | greedy | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt-chat | sampling | no | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt-chat | sampling | yes | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |

## (b) extra split: model x (decoding, repetition_penalty) x domain

| model | decoding | repetition_penalty | code | czech | german | total |
|---|---|---|---|---|---|---|
| chatgpt | greedy | no | 920 | 1965 | 1970 | 4855 |
| chatgpt | sampling | no | 920 | 1965 | 1970 | 4855 |
| cohere | greedy | no | 920 | 1965 | 1970 | 4855 |
| cohere | sampling | no | 920 | 1965 | 1970 | 4855 |
| cohere-chat | greedy | no | 920 | 1965 | 1970 | 4855 |
| cohere-chat | sampling | no | 920 | 1965 | 1970 | 4855 |
| gpt2 | greedy | no | 920 | 1965 | 1970 | 4855 |
| gpt2 | greedy | yes | 920 | 1965 | 1970 | 4855 |
| gpt2 | sampling | no | 920 | 1965 | 1970 | 4855 |
| gpt2 | sampling | yes | 920 | 1965 | 1970 | 4855 |
| gpt3 | greedy | no | 920 | 1965 | 1970 | 4855 |
| gpt3 | sampling | no | 920 | 1965 | 1970 | 4855 |
| gpt4 | greedy | no | 920 | 1965 | 1970 | 4855 |
| gpt4 | sampling | no | 920 | 1965 | 1970 | 4855 |
| human |  |  | 920 | 1965 | 1970 | 4855 |
| llama-chat | greedy | no | 920 | 1965 | 1970 | 4855 |
| llama-chat | greedy | yes | 920 | 1965 | 1970 | 4855 |
| llama-chat | sampling | no | 920 | 1965 | 1970 | 4855 |
| llama-chat | sampling | yes | 920 | 1965 | 1970 | 4855 |
| mistral | greedy | no | 920 | 1965 | 1970 | 4855 |
| mistral | greedy | yes | 920 | 1965 | 1970 | 4855 |
| mistral | sampling | no | 920 | 1965 | 1970 | 4855 |
| mistral | sampling | yes | 920 | 1965 | 1970 | 4855 |
| mistral-chat | greedy | no | 920 | 1965 | 1970 | 4855 |
| mistral-chat | greedy | yes | 920 | 1965 | 1970 | 4855 |
| mistral-chat | sampling | no | 920 | 1965 | 1970 | 4855 |
| mistral-chat | sampling | yes | 920 | 1965 | 1970 | 4855 |
| mpt | greedy | no | 920 | 1965 | 1970 | 4855 |
| mpt | greedy | yes | 920 | 1965 | 1970 | 4855 |
| mpt | sampling | no | 920 | 1965 | 1970 | 4855 |
| mpt | sampling | yes | 920 | 1965 | 1970 | 4855 |
| mpt-chat | greedy | no | 920 | 1965 | 1970 | 4855 |
| mpt-chat | greedy | yes | 920 | 1965 | 1970 | 4855 |
| mpt-chat | sampling | no | 920 | 1965 | 1970 | 4855 |
| mpt-chat | sampling | yes | 920 | 1965 | 1970 | 4855 |

## (c) Chosen chat models x domain under (sampling, rep_penalty=no) (rows in interim parquet)

| model | abstracts | books | news | poetry | recipes | reddit | reviews | wiki | All |
|---|---|---|---|---|---|---|---|---|---|
| chatgpt | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| cohere-chat | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| llama-chat | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mistral-chat | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| mpt-chat | 1766 | 1781 | 1780 | 1771 | 1772 | 1779 | 943 | 1779 | 13371 |
| All | 8830 | 8905 | 8900 | 8855 | 8860 | 8895 | 4715 | 8895 | 66855 |

Source_ids having all 5 models (text domains):

| domain | source_ids_with_all_5_models | source_ids_any |
|---|---|---|
| abstracts | 1766 | 1766 |
| books | 1781 | 1781 |
| news | 1780 | 1780 |
| poetry | 1771 | 1771 |
| recipes | 1772 | 1772 |
| reddit | 1779 | 1779 |
| reviews | 943 | 943 |
| wiki | 1779 | 1779 |

Code (extra split): 920 source_ids with all 5 models (of 920 with any).
