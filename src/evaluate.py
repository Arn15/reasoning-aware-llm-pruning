"""Evaluation: WikiText-2 perplexity and lm-evaluation-harness benchmarks."""
from __future__ import annotations

import math

import torch
from tqdm import tqdm

# Meta's Llama-3 instruct evaluation formats (8-shot CoT GSM8K; 0-shot ARC-Challenge, chat format)
TASKS = {
    "gsm8k": {"task": "gsm8k_cot_llama", "metric": "exact_match,strict-match",
              "extra": ["exact_match,flexible-extract"]},
    "arc_challenge": {"task": "arc_challenge_chat", "metric": "exact_match,remove_whitespace"},
}


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seqlen: int = 2048) -> float:
    """Non-overlapping seqlen windows over the full test stream (standard pruning-paper setup)."""
    device = next(model.parameters()).device
    n = ids.shape[1] // seqlen
    nll = 0.0
    for i in tqdm(range(n), desc="wikitext2 ppl", leave=False):
        x = ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        nll += model(x, labels=x, use_cache=False).loss.float().item()
    return math.exp(nll / n)


def run_benchmarks(model, tokenizer, names=("gsm8k", "arc_challenge"), limits=None, batch_size=16) -> dict:
    """Run lm-eval tasks on an in-memory (possibly pruned) model. Returns score, stderr and n per task."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    limits = limits or {}
    device = str(next(model.parameters()).device)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_batch_size=32, device=device)
    out = {}
    for name in names:
        spec = TASKS[name]
        res = simple_evaluate(model=lm, tasks=[spec["task"]], limit=limits.get(name),
                              apply_chat_template=True, fewshot_as_multiturn=True,
                              log_samples=False, bootstrap_iters=1000)
        r = res["results"][spec["task"]]
        metric, filt = spec["metric"].split(",")
        out[name] = {"score": r[spec["metric"]], "stderr": r.get(f"{metric}_stderr,{filt}"),
                     "n": res.get("n-samples", {}).get(spec["task"], {}).get("effective"),
                     "task": spec["task"], "metric": spec["metric"]}
        for extra in spec.get("extra", []):
            m, f = extra.split(",")
            out[name][f] = {"score": r.get(extra), "stderr": r.get(f"{m}_stderr,{f}")}
    return out
