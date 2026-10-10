"""Layer-wise sensitivity of reasoning vs. language modeling.

For each decoder layer l we scale only that layer's *residual contribution*:
    h_out' = h_in + alpha * (h_out - h_in)
(alpha = 0 skips the layer entirely; alpha = 1 is the unmodified model). Scaling the full
hidden state instead would also shrink the residual stream from all earlier layers.

Sensitivity is the increase in mean negative log-likelihood:
  * reasoning: NLL of the gold worked solution given a GSM8K *train* question (answer tokens only)
  * language modeling: NLL of C4 *train* text
Using teacher-forced NLL on training data keeps the test sets untouched and is far cheaper
than generating answers for every perturbed layer.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .prune import decoder_layers


def build_reasoning_batch(tokenizer, pairs, max_len: int = 768):
    """Tokenize (question, solution) pairs with the chat template; labels = solution tokens only."""
    items = []
    for q, a in pairs:
        prompt_msgs = [{"role": "user", "content": q}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {q}\nAnswer: "
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(a, add_special_tokens=False).input_ids
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        items.append((torch.tensor([ids]), torch.tensor([labels])))
    return items


def build_lm_batch(tokenizer, texts, max_len: int = 512):
    items = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids
        items.append((ids, ids.clone()))
    return items


@torch.no_grad()
def mean_nll(model, items) -> float:
    device = next(model.parameters()).device
    total, n_tok = 0.0, 0
    for ids, labels in items:
        out = model(ids.to(device), labels=labels.to(device), use_cache=False)
        k = (labels[:, 1:] != -100).sum().item()
        total += out.loss.float().item() * k
        n_tok += k
    return total / n_tok


def _residual_scale_hook(alpha: float):
    def pre_hook(module, args, kwargs):
        module._h_in = args[0] if args else kwargs["hidden_states"]

    def hook(module, args, kwargs, output):
        h_out = output[0] if isinstance(output, tuple) else output
        h_new = module._h_in + alpha * (h_out - module._h_in)
        del module._h_in
        return (h_new,) + tuple(output[1:]) if isinstance(output, tuple) else h_new

    return pre_hook, hook


@torch.no_grad()
def layer_sensitivity(model, reasoning_items, lm_items, alpha: float = 0.0) -> dict:
    model.eval()
    base_r, base_l = mean_nll(model, reasoning_items), mean_nll(model, lm_items)
    rows = []
    for i, layer in enumerate(tqdm(decoder_layers(model), desc="layer sensitivity")):
        pre, post = _residual_scale_hook(alpha)
        h1 = layer.register_forward_pre_hook(pre, with_kwargs=True)
        h2 = layer.register_forward_hook(post, with_kwargs=True)
        try:
            r, l = mean_nll(model, reasoning_items), mean_nll(model, lm_items)
        finally:
            h1.remove()
            h2.remove()
        rows.append({"layer": i, "reasoning_nll": r, "lm_nll": l,
                     "reasoning_delta": r - base_r, "lm_delta": l - base_l})
    return {"alpha": alpha, "base_reasoning_nll": base_r, "base_lm_nll": base_l, "per_layer": rows}


def allocate_sparsity(sens: dict, target: float, delta: float, score: str = "reasoning") -> list[float]:
    """Per-layer sparsity with mean exactly `target` (all Llama decoder layers have equal size).

    Layers more sensitive to reasoning are pruned less, by up to `delta`.
      score="reasoning": rank by reasoning NLL increase
      score="specific" : rank by reasoning increase minus language-modeling increase
                         (protects layers that matter for reasoning *more than* for general LM)
    """
    r = torch.tensor([row["reasoning_delta"] for row in sens["per_layer"]], dtype=torch.float64)
    if score == "specific":
        lm = torch.tensor([row["lm_delta"] for row in sens["per_layer"]], dtype=torch.float64)
        r = r / r.abs().max() - lm / lm.abs().max()
    elif score != "reasoning":
        raise ValueError(score)
    # rank-based to be robust to outliers (e.g. layer 0), mapped to [-1, 1], zero mean
    ranks = r.argsort().argsort().double()
    z = 2 * ranks / (len(r) - 1) - 1
    s = target - delta * z
    s = s.clamp(0.0, 0.95)
    s = s + (target - s.mean())          # re-centre after clamping
    return [round(float(v), 6) for v in s]
