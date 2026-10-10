"""Datasets for calibration, sensitivity and perplexity.

Leakage rules
  * Pruning calibration: C4 *train* (as in the Wanda paper).
  * Layer sensitivity / sparsity selection: GSM8K *train* and C4 *train*.
  * Evaluation: GSM8K test, ARC-Challenge test, WikiText-2 test (never used above).
"""
from __future__ import annotations

import random

import torch
from datasets import load_dataset

C4_SHARD = "en/c4-train.00000-of-01024.json.gz"


def c4_calibration(tokenizer, nsamples: int = 128, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    """Random seqlen-token windows from C4 train documents -> LongTensor [nsamples, seqlen]."""
    data = load_dataset("allenai/c4", data_files={"train": C4_SHARD}, split="train")
    rng = random.Random(seed)
    out = []
    while len(out) < nsamples:
        text = data[rng.randrange(len(data))]["text"]
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(ids) <= seqlen:
            continue
        start = rng.randrange(len(ids) - seqlen)
        out.append(ids[start:start + seqlen])
    return torch.stack(out)


def c4_texts(n: int = 64, seed: int = 1, min_chars: int = 2000) -> list[str]:
    """Plain C4 train documents for language-modeling sensitivity (disjoint seed from calibration)."""
    data = load_dataset("allenai/c4", data_files={"train": C4_SHARD}, split="train")
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        t = data[rng.randrange(len(data))]["text"]
        if len(t) >= min_chars:
            out.append(t[:min_chars])
    return out


def gsm8k_train_pairs(n: int = 128, offset: int = 0, seed: int = 0) -> list[tuple[str, str]]:
    """(question, worked solution) pairs from GSM8K *train*.

    offset lets sensitivity scoring and sparsity selection use disjoint subsets.
    """
    data = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(data)))
    random.Random(seed).shuffle(idx)
    pairs = []
    for i in idx[offset:offset + n]:
        q, a = data[i]["question"], data[i]["answer"]
        steps, final = a.split("####")
        # strip calculator annotations like <<3*4=12>> and match the eval answer format
        import re
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        pairs.append((q, f"{steps} The final answer is {final.strip()}"))
    return pairs


def wikitext2_test_ids(tokenizer) -> torch.Tensor:
    """Full WikiText-2 (raw) test split as one token stream, the standard perplexity setup."""
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
