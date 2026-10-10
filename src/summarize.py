"""Collect results/runs/*.json into results/results.md and plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"dense": "Dense", "magnitude": "Magnitude", "wanda": "Wanda",
         "wanda_sa": "Wanda + reasoning-sensitivity allocation"}


def fmt(b):
    if not b:
        return "–"
    se = b.get("stderr")
    ci = f" ± {1.96 * se * 100:.1f}" if isinstance(se, (int, float)) else ""
    return f"{b['score'] * 100:.1f}{ci}"


def main():
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / "runs").glob("*.json"))]
    if not runs:
        raise SystemExit("no runs found in results/runs/")
    order = {"dense": 0, "magnitude": 1, "wanda": 2, "wanda_sa": 3}
    runs.sort(key=lambda r: (r["pattern"] not in ("unstructured", "none"), r["target_sparsity"], order[r["method"]]))

    model = runs[0]["model"]
    lines = [f"# Pruning results — {model}", "",
             "GSM8K: 8-shot CoT (`gsm8k_cot_llama`, strict match). ARC-Challenge: 0-shot chat format "
             "(`arc_challenge_chat`). Scores in %, ± = 95% CI (1.96 × stderr). "
             "Perplexity: full WikiText-2 test, 2048-token windows.", "",
             "| Method | Pattern | Sparsity (measured) | GSM8K | GSM8K (flexible) | ARC-C | WikiText-2 PPL | Notes |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        b = r.get("benchmarks", {})
        note = ""
        if r["method"] == "wanda_sa":
            note = f"δ={r['delta']:.2f} ({r['score']})"
        lines.append(f"| {LABEL[r['method']]} | {'–' if r['pattern'] == 'none' else r['pattern']} | {r['measured_sparsity']:.3f} | "
                     f"{fmt(b.get('gsm8k'))} | {fmt((b.get('gsm8k') or {}).get('flexible-extract'))} | {fmt(b.get('arc_challenge'))} | {r['wikitext2_ppl']:.2f} | {note} |")
    (ROOT / "results" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # sparsity curves (unstructured only)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dense = [r for r in runs if r["method"] == "dense"]
    for method in ("magnitude", "wanda", "wanda_sa"):
        pts = [r for r in runs if r["method"] == method and r["pattern"] == "unstructured"] + dense
        pts.sort(key=lambda r: r["target_sparsity"])
        if len(pts) < 2:
            continue
        xs = [r["target_sparsity"] for r in pts]
        for ax, key in zip(axes, ("gsm8k", "arc_challenge", "ppl")):
            ys = [r["wikitext2_ppl"] if key == "ppl" else r.get("benchmarks", {}).get(key, {}).get("score", float("nan")) * 100
                  for r in pts]
            ax.plot(xs, ys, marker="o", label=LABEL[method])
    for ax, title in zip(axes, ("GSM8K accuracy (%)", "ARC-Challenge accuracy (%)", "WikiText-2 perplexity (log)")):
        ax.set_title(title); ax.set_xlabel("sparsity"); ax.grid(alpha=0.3)
    axes[2].set_yscale("log")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "sparsity_curves.png", dpi=150)

    # layer sensitivity
    for p in (ROOT / "results").glob("sensitivity_*.json"):
        s = json.loads(p.read_text())
        rows = s["per_layer"]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [r["layer"] for r in rows]
        w = 0.4
        ax.bar([i - w / 2 for i in x], [r["reasoning_delta"] for r in rows], w, label="reasoning (GSM8K-train NLL)")
        ax.bar([i + w / 2 for i in x], [r["lm_delta"] for r in rows], w, label="language modeling (C4 NLL)")
        ax.set_xlabel("decoder layer"); ax.set_ylabel("NLL increase when layer is skipped")
        ax.set_title(f"Layer sensitivity — {s['model']}"); ax.legend(); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(ROOT / "results" / f"{p.stem}.png", dpi=150)


if __name__ == "__main__":
    main()
