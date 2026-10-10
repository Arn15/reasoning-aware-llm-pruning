"""One-shot pruning: magnitude and Wanda, unstructured or N:M, with optional per-layer sparsity.

Wanda follows Sun et al. (2023) and the reference implementation (locuslab/wanda):
  * score(W_ij) = |W_ij| * ||X_j||_2, where X_j is the j-th input feature over calibration tokens
  * weights are compared **within each output row** (per-output comparison group)
  * layers are pruned **sequentially**: layer i's calibration inputs are the outputs of the
    already-pruned layers 0..i-1
Embeddings and lm_head are never pruned.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def decoder_layers(model) -> nn.ModuleList:
    return model.model.layers


def linear_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    return {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}


# ------------------------------------------------------------------ masks ----
def _mask_from_scores(scores: torch.Tensor, sparsity: float, pattern: str, per_row: bool) -> torch.Tensor:
    """Boolean mask of weights to ZERO (True = prune)."""
    if pattern != "unstructured":
        n, m = (int(x) for x in pattern.split(":"))          # e.g. "2:4" -> prune 2 of every 4
        rows, cols = scores.shape
        grouped = scores.reshape(rows, cols // m, m)
        idx = torch.topk(grouped, n, dim=-1, largest=False).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, idx, True)
        return mask.reshape(rows, cols)
    if sparsity <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if per_row:
        k = round(scores.shape[1] * sparsity)   # round, so per-layer allocations hit their target
        if k == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        idx = torch.topk(scores, k, dim=1, largest=False).indices
        return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, idx, True)
    k = max(1, round(scores.numel() * sparsity))
    thresh = torch.kthvalue(scores.flatten().float(), k).values
    return scores <= thresh


def _layer_sparsities(sparsity: float | Sequence[float], n_layers: int) -> list[float]:
    if isinstance(sparsity, (int, float)):
        return [float(sparsity)] * n_layers
    assert len(sparsity) == n_layers, "need one sparsity per decoder layer"
    return [float(s) for s in sparsity]


# -------------------------------------------------------------- magnitude ----
@torch.no_grad()
def prune_magnitude(model, sparsity: float | Sequence[float], pattern: str = "unstructured") -> None:
    """Magnitude baseline: |W|, compared across the whole matrix (layer-wise), as in the Wanda paper."""
    layers = decoder_layers(model)
    for s, layer in zip(_layer_sparsities(sparsity, len(layers)), layers):
        for lin in linear_modules(layer).values():
            W = lin.weight.data
            W[_mask_from_scores(W.abs(), s, pattern, per_row=False)] = 0


# ------------------------------------------------------------------ wanda ----
class _Catcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer, self.inputs, self.kwargs = layer, [], None

    def forward(self, hidden_states, *args, **kwargs):
        self.inputs.append(hidden_states)
        self.kwargs = kwargs
        raise ValueError("caught")


def _run_layer(layer, x, kwargs):
    out = layer(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def collect_layer0_inputs(model, calib_ids: torch.Tensor):
    """Hidden states entering decoder layer 0 for each calibration sample, plus the layer kwargs."""
    layers = decoder_layers(model)
    device = next(model.parameters()).device
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    try:
        for i in range(calib_ids.shape[0]):
            try:
                model(calib_ids[i:i + 1].to(device), use_cache=False)
            except ValueError:
                pass
    finally:
        layers[0] = catcher.layer
    kwargs = {k: v for k, v in (catcher.kwargs or {}).items() if k not in ("past_key_values", "past_key_value", "use_cache")}
    return catcher.inputs, kwargs


@torch.no_grad()
def prune_wanda(model, calib_ids: torch.Tensor, sparsity: float | Sequence[float],
                pattern: str = "unstructured", verbose: bool = True) -> None:
    layers = decoder_layers(model)
    sparsities = _layer_sparsities(sparsity, len(layers))
    inps, kwargs = collect_layer0_inputs(model, calib_ids)

    for li, layer in enumerate(layers):
        lins = linear_modules(layer)
        sq_norms = {n: None for n in lins}
        counts = {n: 0 for n in lins}

        def make_hook(name):
            def hook(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                s = x.pow(2).sum(0)
                sq_norms[name] = s if sq_norms[name] is None else sq_norms[name] + s
                counts[name] += x.shape[0]
            return hook

        handles = [m.register_forward_hook(make_hook(n)) for n, m in lins.items()]
        for x in inps:
            _run_layer(layer, x, kwargs)
        for h in handles:
            h.remove()

        for n, lin in lins.items():
            W = lin.weight.data
            # ||X_j||_2 over all calibration tokens (reference implementation divides by token
            # count; the per-row ranking is unchanged by that constant)
            act = torch.sqrt(sq_norms[n] / counts[n]).to(W.device)
            scores = W.abs().float() * act.unsqueeze(0)
            W[_mask_from_scores(scores, sparsities[li], pattern, per_row=True)] = 0

        # outputs of the *pruned* layer feed the next layer
        inps = [_run_layer(layer, x, kwargs) for x in inps]
        if verbose:
            print(f"  wanda layer {li:2d}  sparsity {sparsities[li]:.3f}" if pattern == "unstructured"
                  else f"  wanda layer {li:2d}  {pattern}")


# --------------------------------------------------------------- checking ----
@torch.no_grad()
def measured_sparsity(model) -> dict:
    per_layer, total, zeros = [], 0, 0
    for layer in decoder_layers(model):
        lz = lt = 0
        for lin in linear_modules(layer).values():
            lz += (lin.weight == 0).sum().item()
            lt += lin.weight.numel()
        per_layer.append(lz / lt)
        zeros, total = zeros + lz, total + lt
    return {"overall": zeros / total, "per_layer": per_layer}
