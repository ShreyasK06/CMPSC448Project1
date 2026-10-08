# Prompt analysis (text domains)

- source_ids where all 5 prompts are identical: 3200 / 3200 = 1.0000
- distinct prompts per source_id (value counts): {1: 3200}

Distinct prompt templates per model (title replaced by `<TITLE>`):

| model | n_templates | n_raw_prompts | title_found_in_prompt |
|---|---|---|---|
| chatgpt | 8 | 3196 | 1.0 |
| cohere-chat | 8 | 3196 | 1.0 |
| llama-chat | 8 | 3196 | 1.0 |
| mistral-chat | 8 | 3196 | 1.0 |
| mpt-chat | 8 | 3196 | 1.0 |

Template counts by (model, domain):

| model | abstracts | books | news | poetry | recipes | reddit | reviews | wiki |
|---|---|---|---|---|---|---|---|---|
| chatgpt | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| cohere-chat | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| llama-chat | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| mistral-chat | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| mpt-chat | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Two example prompts per model (truncated to 300 chars):

- chatgpt [abstracts]: `'Write the abstract for the academic paper titled "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation".'`
- chatgpt [abstracts]: `'Write the abstract for the academic paper titled "Comprehensive Multi-Modal Interactions for Referring Image Segmentation".'`
- cohere-chat [abstracts]: `'Write the abstract for the academic paper titled "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation".'`
- cohere-chat [abstracts]: `'Write the abstract for the academic paper titled "Comprehensive Multi-Modal Interactions for Referring Image Segmentation".'`
- llama-chat [abstracts]: `'Write the abstract for the academic paper titled "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation".'`
- llama-chat [abstracts]: `'Write the abstract for the academic paper titled "Comprehensive Multi-Modal Interactions for Referring Image Segmentation".'`
- mistral-chat [abstracts]: `'Write the abstract for the academic paper titled "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation".'`
- mistral-chat [abstracts]: `'Write the abstract for the academic paper titled "Comprehensive Multi-Modal Interactions for Referring Image Segmentation".'`
- mpt-chat [abstracts]: `'Write the abstract for the academic paper titled "Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation".'`
- mpt-chat [abstracts]: `'Write the abstract for the academic paper titled "Comprehensive Multi-Modal Interactions for Referring Image Segmentation".'`

# Prompt analysis (code)

- source_ids where all 5 prompts are identical: 400 / 400 = 1.0000
- distinct prompts per source_id (value counts): {1: 400}

Distinct prompt templates per model (title replaced by `<TITLE>`):

| model | n_templates | n_raw_prompts | title_found_in_prompt |
|---|---|---|---|
| chatgpt | 1 | 400 | 1.0 |
| cohere-chat | 1 | 400 | 1.0 |
| llama-chat | 1 | 400 | 1.0 |
| mistral-chat | 1 | 400 | 1.0 |
| mpt-chat | 1 | 400 | 1.0 |

Template counts by (model, domain):

| model | code |
|---|---|
| chatgpt | 1 |
| cohere-chat | 1 |
| llama-chat | 1 |
| mistral-chat | 1 |
| mpt-chat | 1 |

Two example prompts per model (truncated to 300 chars):

- chatgpt [code]: `'Write just the Python code solution for the problem "Write a python function to find the minimum number of rotations required to get the same string.".'`
- chatgpt [code]: `'Write just the Python code solution for the problem "Write a function to count the most common words in a dictionary.".'`
- cohere-chat [code]: `'Write just the Python code solution for the problem "Write a python function to find the minimum number of rotations required to get the same string.".'`
- cohere-chat [code]: `'Write just the Python code solution for the problem "Write a function to count the most common words in a dictionary.".'`
- llama-chat [code]: `'Write just the Python code solution for the problem "Write a python function to find the minimum number of rotations required to get the same string.".'`
- llama-chat [code]: `'Write just the Python code solution for the problem "Write a function to count the most common words in a dictionary.".'`
- mistral-chat [code]: `'Write just the Python code solution for the problem "Write a python function to find the minimum number of rotations required to get the same string.".'`
- mistral-chat [code]: `'Write just the Python code solution for the problem "Write a function to count the most common words in a dictionary.".'`
- mpt-chat [code]: `'Write just the Python code solution for the problem "Write a python function to find the minimum number of rotations required to get the same string.".'`
- mpt-chat [code]: `'Write just the Python code solution for the problem "Write a function to count the most common words in a dictionary.".'`
