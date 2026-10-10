# Reasoning-Aware Pruning of LLMs (Without Fine-Tuning)

One-shot pruning of Llama-3.2-1B-Instruct. Measures how much each transformer layer matters for multi-step reasoning versus general language modeling, then uses that to decide **where** to prune: reasoning-critical layers are pruned less, others more, at the same overall sparsity. Compared against magnitude pruning and Wanda.

> **Status:** pipeline rebuilt with corrected evaluation; results pending the first full run. Next: safety-behavior evaluation under pruning.

## Background
Originally a team project for NC State's Deep Learning Beyond Accuracy course (Spring 2026). Team: Jai Pavan Ramani, Shivang Nitinkumar Patel, Arnav Yennam.
This repository is my independent rebuild. The original course code is kept unchanged in `original/` for reference.

### What changed from the course version
| Issue in the original | Fix here |
|---|---|
| Wanda calibrated on WikiText-2 *test*, the same split used for perplexity | Calibration on **C4 train** (128 × 2048 tokens), as in the Wanda paper |
| Weights ranked across the whole matrix | **Per-output-row** comparison groups, and layers pruned **sequentially** (each layer calibrated on outputs of already-pruned layers), matching the reference implementation |
| GSM8K scored by checking whether the answer string appears anywhere in the output | **lm-evaluation-harness**: 8-shot CoT `gsm8k_cot_llama` (strict + flexible match) and `arc_challenge_chat`, with standard errors |
| Perplexity on the first 50 lines of WikiText-2 | **Full WikiText-2 test split**, 2048-token windows |
| Layer sensitivity measured on the same GSM8K test questions used for evaluation | Sensitivity on **GSM8K train** and **C4 train** only; test sets are never touched before evaluation |
| Perturbation multiplied the whole hidden state (including the residual stream) | Scale only each layer's **residual contribution**: `h_in + α·(h_out − h_in)` |
| Base model with near-floor GSM8K accuracy | **Instruct** model in its official chat eval format, leaving room to measure degradation |

## Method
1. **Layer sensitivity.** Skip each decoder layer (α = 0) and record the increase in NLL of gold GSM8K-train solutions (reasoning) and of C4 text (language modeling).
2. **Sparsity allocation.** Rank layers by reasoning sensitivity (`--score reasoning`) or by reasoning-minus-LM sensitivity (`--score specific`). Assign per-layer sparsity `S − δ·z`, where `z ∈ [−1, 1]` is the rank, so the mean stays exactly `S`.
3. **Prune** with Wanda using the per-layer sparsities.
4. **Choose δ** from {0, 0.05, 0.10, 0.15} by reasoning NLL on a *separate* GSM8K-train subset. δ = 0 is plain Wanda, so the method can fall back to it.

## Experiments
| Method | Sparsity |
|---|---|
| Dense | – |
| Magnitude | 50% |
| Wanda | 20%, 40%, 50%, 60%, 2:4 |
| Wanda + sensitivity allocation (reasoning) | 50%, 60% |
| Wanda + sensitivity allocation (reasoning-specific) | 50% |

GSM8K uses the first 500 test questions (same subset for every method, so comparisons are paired); ARC-Challenge uses the full test set.

## Running
```bash
pip install -r requirements.txt
huggingface-cli login          # account must have accepted the Llama 3.2 license
bash run_all.sh                # full grid, resumable; ~3 h on a T4
python -m src.summarize        # results/results.md + plots
python -m pytest tests -q      # logic tests on a tiny random Llama (no downloads)
```
Single runs: `python -m src.run --method wanda --sparsity 0.5`. On Colab, use `notebooks/run_experiments.ipynb` (saves results to Google Drive as it goes).

Note: unstructured sparsity reduces parameters but not latency on standard GPUs; 2:4 sparsity can be accelerated on Ampere-or-newer GPUs (not the T4 used here), so only its quality is reported.

## Results
_Pending the first full run._

## Layout
```
src/data.py          calibration, sensitivity and evaluation data (with the leakage rules)
src/prune.py         magnitude and Wanda pruning, unstructured / N:M, per-layer sparsity
src/sensitivity.py   residual-scaled layer sensitivity + sparsity allocation
src/evaluate.py      WikiText-2 perplexity, lm-eval benchmarks
src/run.py           one configuration end to end -> results/runs/*.json
src/summarize.py     results table and plots
tests/               logic tests (exact per-row sparsity, 2:4 pattern, layer-skip equivalence, allocation)
original/            unchanged course code
docs/                course abstract and presentation
```

## Roadmap
- [x] Corrected Wanda (per-row, sequential, C4 calibration) and magnitude / 2:4 baselines
- [x] Standardized evaluation with confidence intervals; leak-free sensitivity
- [x] Sensitivity-aware per-layer sparsity with held-out δ selection
- [ ] Full results on Llama-3.2-1B-Instruct
- [ ] Safety: refusal rate on harmful-prompt benchmarks across sparsity levels
