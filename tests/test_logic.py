"""Logic tests on a tiny randomly-initialised Llama (no downloads needed).

Run: python -m pytest tests -q
"""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src import prune, sensitivity
from src.evaluate import perplexity


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def calib():
    torch.manual_seed(1)
    return torch.randint(0, 300, (8, 64))


def row_sparsities(model):
    out = []
    for layer in prune.decoder_layers(model):
        for lin in prune.linear_modules(layer).values():
            out.append((lin.weight == 0).float().mean(1))
    return out


def test_wanda_per_row_exact(tiny, calib):
    m = copy.deepcopy(tiny)
    prune.prune_wanda(m, calib, 0.5, verbose=False)
    for rs in row_sparsities(m):
        assert torch.allclose(rs, torch.full_like(rs, 0.5)), "every output row must be exactly 50% sparse"


def test_wanda_matches_direct_scoring_on_layer0(tiny, calib):
    """Layer 0 sees un-pruned inputs, so sequential Wanda must equal direct |W|*||X|| per-row pruning."""
    m = copy.deepcopy(tiny)
    lin = m.model.layers[0].mlp.down_proj
    feats = []
    h = lin.register_forward_hook(lambda mod, i, o: feats.append(i[0].reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        for i in range(calib.shape[0]):
            m(calib[i:i + 1], use_cache=False)
    h.remove()
    X = torch.cat(feats)
    scores = lin.weight.abs() * X.norm(dim=0)
    k = int(scores.shape[1] * 0.5)
    expected = torch.ones_like(scores, dtype=torch.bool)
    expected.scatter_(1, torch.topk(scores, k, dim=1, largest=False).indices, False)
    prune.prune_wanda(m, calib, 0.5, verbose=False)
    assert torch.equal(lin.weight != 0, expected)


def test_layerwise_sparsity_and_mean(tiny, calib):
    m = copy.deepcopy(tiny)
    alloc = [0.3, 0.45, 0.55, 0.7]
    prune.prune_wanda(m, calib, alloc, verbose=False)
    ms = prune.measured_sparsity(m)
    for got, want in zip(ms["per_layer"], alloc):
        assert abs(got - want) < 0.02
    assert abs(ms["overall"] - 0.5) < 0.01


def test_two_four(tiny, calib):
    m = copy.deepcopy(tiny)
    prune.prune_wanda(m, calib, 0.5, pattern="2:4", verbose=False)
    for layer in prune.decoder_layers(m):
        for lin in prune.linear_modules(layer).values():
            g = (lin.weight == 0).reshape(lin.weight.shape[0], -1, 4).sum(-1)
            assert (g >= 2).all()


def test_magnitude(tiny):
    m = copy.deepcopy(tiny)
    prune.prune_magnitude(m, 0.5)
    assert abs(prune.measured_sparsity(m)["overall"] - 0.5) < 0.01


def test_embeddings_untouched(tiny, calib):
    m = copy.deepcopy(tiny)
    prune.prune_wanda(m, calib, 0.6, verbose=False)
    assert torch.equal(m.model.embed_tokens.weight, tiny.model.embed_tokens.weight)
    assert torch.equal(m.lm_head.weight, tiny.lm_head.weight)


def test_sensitivity_alpha_one_is_noop_and_alpha_zero_skips(tiny, calib):
    items = [(calib[i:i + 1], calib[i:i + 1].clone()) for i in range(4)]
    s1 = sensitivity.layer_sensitivity(tiny, items, items, alpha=1.0)
    assert all(abs(r["reasoning_delta"]) < 1e-5 for r in s1["per_layer"])
    s0 = sensitivity.layer_sensitivity(tiny, items, items, alpha=0.0)
    # alpha=0 must equal physically removing the layer
    m = copy.deepcopy(tiny)
    del m.model.layers[2]
    m.config.num_hidden_layers = 3
    for i, l in enumerate(m.model.layers):
        l.self_attn.layer_idx = i
    assert abs(sensitivity.mean_nll(m, items) - s0["per_layer"][2]["reasoning_nll"]) < 1e-4


def test_allocation():
    sens = {"per_layer": [{"reasoning_delta": d, "lm_delta": l} for d, l in
                          zip([5.0, 0.1, 0.3, 0.2, 0.9, 0.05], [4.0, 0.2, 0.1, 0.1, 0.2, 0.3])]}
    for score in ("reasoning", "specific"):
        a = sensitivity.allocate_sparsity(sens, 0.5, 0.1, score)
        assert abs(sum(a) / len(a) - 0.5) < 1e-6
        assert max(a) - min(a) <= 0.2 + 1e-9
    a = sensitivity.allocate_sparsity(sens, 0.5, 0.1, "reasoning")
    assert a[0] == min(a), "most reasoning-sensitive layer is pruned least"
    assert sensitivity.allocate_sparsity(sens, 0.5, 0.0) == [0.5] * 6


def test_perplexity_runs(tiny):
    ids = torch.randint(0, 300, (1, 300))
    p = perplexity(tiny, ids, seqlen=64)
    assert 1 < p < 1e4
