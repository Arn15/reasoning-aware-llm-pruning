#!/usr/bin/env bash
# Full experiment grid. Each finished run is saved to results/runs/ and skipped on re-run,
# so this can be restarted after a Colab disconnect.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
run() { python -m src.run --model "$MODEL" "$@"; }

run --method dense
run --method sensitivity
run --method magnitude --sparsity 0.5
for s in 0.2 0.4 0.5 0.6; do run --method wanda --sparsity "$s"; done
run --method wanda --pattern 2:4
for s in 0.5 0.6; do run --method wanda_sa --sparsity "$s" --delta auto --score reasoning; done
run --method wanda_sa --sparsity 0.5 --delta auto --score specific
python -m src.summarize
