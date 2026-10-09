# Reasoning-Aware Pruning of LLMs (Without Fine-Tuning)

One-shot pruning of Llama-3.2 models that measures how much each transformer layer matters for multi-step reasoning, and uses that sensitivity to decide where to prune, compared against Wanda and magnitude pruning. Also evaluates whether pruning erodes safety behavior.

> **Status:** being rebuilt with corrected evaluation. Results below will be filled in from the re-run.

## Background
Originally a team project for NC State's Deep Learning Beyond Accuracy course (Spring 2026). Team: Jai Pavan Ramani, Shivang Nitinkumar Patel, Arnav Yennam.
This repository is my independent rebuild and extension. The original course notebook is kept in `notebooks/llama_run_original.ipynb` for reference.

## Planned work (this repo)
- [ ] Calibrate on C4 (train), evaluate perplexity on the full WikiText-2 test split (no calibration/eval overlap)
- [ ] Wanda with per-output-row comparison groups, checked against the reference implementation
- [ ] Standardized evaluation via lm-evaluation-harness (GSM8K 8-shot CoT, ARC-Challenge, WikiText-2) with bootstrap 95% CIs
- [ ] Layer sensitivity measured on GSM8K *train*, perturbing only each layer's residual contribution; reasoning vs. language-modeling sensitivity
- [ ] Sensitivity-aware non-uniform sparsity vs. uniform Wanda at equal overall sparsity
- [ ] Magnitude pruning baseline; 2:4 semi-structured sparsity with measured latency/memory
- [ ] Safety: refusal rate on harmful-prompt benchmarks (Instruct model) across sparsity levels

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # add your Hugging Face token (Llama models are gated)
```

## Results
_To be added after the re-run._
