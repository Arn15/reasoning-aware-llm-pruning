"""Run one pruning configuration end to end and save its metrics.

Examples
  python -m src.run --method dense
  python -m src.run --method magnitude --sparsity 0.5
  python -m src.run --method wanda --sparsity 0.5
  python -m src.run --method wanda --pattern 2:4
  python -m src.run --method wanda_sa --sparsity 0.5 --delta auto
  python -m src.run --method sensitivity          # only compute + cache layer sensitivity
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data, evaluate, prune, sensitivity

ROOT = Path(__file__).resolve().parents[1]
DELTA_GRID = [0.0, 0.05, 0.10, 0.15]


def slug(name: str) -> str:
    return name.split("/")[-1]


def load(model_name: str, dtype: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tok


def get_sensitivity(args, model, tok) -> dict:
    path = ROOT / "results" / f"sensitivity_{slug(args.model)}.json"
    if path.exists():
        return json.loads(path.read_text())
    print("computing layer sensitivity (GSM8K train + C4 train)...")
    r_items = sensitivity.build_reasoning_batch(tok, data.gsm8k_train_pairs(args.sens_n, offset=0))
    l_items = sensitivity.build_lm_batch(tok, data.c4_texts(args.sens_n))
    sens = sensitivity.layer_sensitivity(model, r_items, l_items, alpha=args.alpha)
    sens.update(model=args.model, n_reasoning=len(r_items), n_lm=len(l_items))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sens, indent=2))
    return sens


def apply(args, model, tok, calib, layer_sparsity):
    if args.method == "magnitude":
        prune.prune_magnitude(model, layer_sparsity, args.pattern)
    elif args.method in ("wanda", "wanda_sa"):
        prune.prune_wanda(model, calib, layer_sparsity, args.pattern, verbose=False)


def choose_delta(args, tok, calib, sens) -> tuple[float, list]:
    """Pick delta by reasoning NLL on a GSM8K-train subset disjoint from the sensitivity subset."""
    items = sensitivity.build_reasoning_batch(tok, data.gsm8k_train_pairs(args.sens_n, offset=args.sens_n))
    trials = []
    for d in DELTA_GRID:
        model, _ = load(args.model, args.dtype)
        alloc = sensitivity.allocate_sparsity(sens, args.sparsity, d, args.score)
        prune.prune_wanda(model, calib, alloc, verbose=False)
        nll = sensitivity.mean_nll(model, items)
        trials.append({"delta": d, "heldout_reasoning_nll": nll})
        print(f"  [select] delta={d:.2f}  held-out GSM8K-train NLL {nll:.4f}")
        del model
        torch.cuda.empty_cache()
    best = min(trials, key=lambda t: t["heldout_reasoning_nll"])
    return best["delta"], trials


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--method", choices=["dense", "magnitude", "wanda", "wanda_sa", "sensitivity"], required=True)
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--pattern", default="unstructured", help='"unstructured" or N:M such as "2:4"')
    ap.add_argument("--delta", default="auto", help="max per-layer sparsity shift for wanda_sa, or 'auto'")
    ap.add_argument("--score", choices=["reasoning", "specific"], default="reasoning")
    ap.add_argument("--alpha", type=float, default=0.0, help="residual scale for sensitivity (0 = skip layer)")
    ap.add_argument("--nsamples", type=int, default=128, help="C4 calibration sequences")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--sens-n", type=int, default=128)
    ap.add_argument("--gsm8k-limit", type=int, default=500)
    ap.add_argument("--arc-limit", type=int, default=None)
    ap.add_argument("--skip-bench", action="store_true", help="perplexity only")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=16, help="lm-eval batch size")
    ap.add_argument("--force", action="store_true", help="re-run even if results already exist")
    args = ap.parse_args(argv)

    name = f"{slug(args.model)}__{args.method}"
    if args.method not in ("dense", "sensitivity"):
        name += f"_{args.pattern.replace(':', 'of') if args.pattern != 'unstructured' else args.sparsity}"
    if args.method == "wanda_sa":
        name += f"_{args.score}"
    out = ROOT / "results" / "runs" / f"{name}.json"
    if out.exists() and not args.force:
        print(f"skip: {out.relative_to(ROOT)} already exists (use --force to re-run)")
        return

    t0 = time.time()
    model, tok = load(args.model, args.dtype)
    n_layers = len(prune.decoder_layers(model))
    record = {"model": args.model, "method": args.method,
              "pattern": "none" if args.method == "dense" else args.pattern,
              "target_sparsity": 0.0 if args.method == "dense" else args.sparsity}

    if args.method == "sensitivity":
        get_sensitivity(args, model, tok)
        return

    layer_sparsity = args.sparsity
    calib = None
    if args.method in ("wanda", "wanda_sa"):
        calib = data.c4_calibration(tok, args.nsamples, args.seqlen)
    if args.method == "wanda_sa":
        if args.pattern != "unstructured":
            raise SystemExit("wanda_sa supports unstructured sparsity only")
        sens = get_sensitivity(args, model, tok)
        if args.delta == "auto":
            del model
            torch.cuda.empty_cache()
            delta, trials = choose_delta(args, tok, calib, sens)
            record["delta_selection"] = trials
            model, _ = load(args.model, args.dtype)
        else:
            delta = float(args.delta)
        layer_sparsity = sensitivity.allocate_sparsity(sens, args.sparsity, delta, args.score)
        record.update(delta=delta, score=args.score, layer_sparsity=layer_sparsity)

    if args.method != "dense":
        apply(args, model, tok, calib, layer_sparsity)
    ms = prune.measured_sparsity(model)
    record["measured_sparsity"] = ms["overall"]
    record["measured_layer_sparsity"] = ms["per_layer"]
    print(f"measured sparsity: {ms['overall']:.4f}")

    record["wikitext2_ppl"] = evaluate.perplexity(model, data.wikitext2_test_ids(tok), args.seqlen)
    print(f"WikiText-2 perplexity: {record['wikitext2_ppl']:.2f}")
    if not args.skip_bench:
        record["benchmarks"] = evaluate.run_benchmarks(
            model, tok, limits={"gsm8k": args.gsm8k_limit, "arc_challenge": args.arc_limit},
            batch_size=args.batch_size)
        for k, v in record["benchmarks"].items():
            print(f"{k}: {v['score']:.4f} ± {v['stderr'] if isinstance(v['stderr'], str) else round(v['stderr'], 4)}")
    record["minutes"] = round((time.time() - t0) / 60, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
