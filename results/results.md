# Pruning results — meta-llama/Llama-3.2-1B-Instruct

GSM8K: 8-shot CoT (`gsm8k_cot_llama`, strict match). ARC-Challenge: 0-shot chat format (`arc_challenge_chat`). Scores in %, ± = 95% CI (1.96 × stderr). Perplexity: full WikiText-2 test, 2048-token windows.

| Method | Pattern | Sparsity (measured) | GSM8K | GSM8K (flexible) | ARC-C | WikiText-2 PPL | Notes |
|---|---|---|---|---|---|---|---|
| Dense | – | 0.000 | 46.0 ± 4.4 | 46.2 ± 4.4 | 56.1 ± 2.8 | 13.16 |  |
| Wanda | unstructured | 0.200 | 37.6 ± 4.3 | 37.6 ± 4.3 | 56.1 ± 2.8 | 13.91 |  |
| Wanda | unstructured | 0.400 | 16.2 ± 3.2 | 16.4 ± 3.2 | 41.4 ± 2.8 | 18.80 |  |
| Magnitude | unstructured | 0.501 | 0.0 ± 0.0 | 1.2 ± 1.0 | 0.5 ± 0.4 | 1539.88 |  |
| Wanda | unstructured | 0.500 | 1.6 ± 1.1 | 1.8 ± 1.2 | 26.7 ± 2.5 | 33.21 |  |
| Wanda + reasoning-sensitivity allocation | unstructured | 0.500 | 1.8 ± 1.2 | 1.8 ± 1.2 | 28.8 ± 2.6 | 32.41 | δ=0.05 (reasoning) |
| Wanda + reasoning-sensitivity allocation | unstructured | 0.500 | 3.4 ± 1.6 | 3.4 ± 1.6 | 27.1 ± 2.5 | 31.93 | δ=0.10 (reasoning-specific) |
| Wanda | unstructured | 0.600 | 1.0 ± 0.9 | 2.4 ± 1.3 | 1.7 ± 0.7 | 108.88 |  |
| Wanda + reasoning-sensitivity allocation | unstructured | 0.600 | 0.8 ± 0.8 | 1.8 ± 1.2 | 15.8 ± 2.1 | 102.72 | δ=0.10 (reasoning) |
| Wanda | 2:4 | 0.500 | 0.8 ± 0.8 | 2.0 ± 1.2 | 6.7 ± 1.4 | 107.82 |  |
