"""Picks steering strengths automatically: steer the held-out capitulation messages at several push sizes, have Sonnet
score the trait and coherence, and keep the strongest push each way whose coherence stays within 5 points of the
unsteered model's (truncated 160-token replies score in the 70s even unsteered, so an absolute cutoff misfires).

    uv run python calib.py
"""
import json

import numpy as np

import llm
from persona import JUDGE_PROMPT, TRAITS

COEFS = [-0.6, -0.4, -0.25, 0.25, 0.4, 0.6]


def main():
    from target import Target
    arts = json.load(open("data/persona_artifacts.json"))
    rep = json.load(open("data/persona_report.json"))
    z = np.load("data/persona_vectors.npz")
    trait = "capitulation"
    layer = rep[trait]["best_layer"]
    k = list(z["layers"]).index(layer)
    unit = z[trait][k] / np.linalg.norm(z[trait][k])
    norm = rep[trait]["resid_norm"][k]
    qs = arts[trait]["questions"][14:20]
    tgt = Target()
    rows = []
    for c in [0.0] + COEFS:
        tgt.set_steering(layer, c * norm * unit) if c else tgt.set_steering()
        for q, a in zip(qs, tgt.generate([[{"role": "user", "content": q}] for q in qs], max_tokens=160)):
            rows.append({"coef": c, "q": q, "a": a})
        print(f"coef {c:+.2f} generated", flush=True)
    sc = llm.pmap(lambda r: llm.chat_json(llm.SONNET, [{"role": "user", "content": JUDGE_PROMPT.format(
        name=trait, definition=TRAITS[trait][0], q=r["q"], a=r["a"])}], max_tokens=4000), rows, workers=16)
    for r, s in zip(rows, sc):
        r.update(score=s["trait"], coherence=s["coherence"])
    summary = {}
    for c in [0.0] + COEFS:
        rs = [r for r in rows if r["coef"] == c]
        summary[c] = {"trait": float(np.mean([r["score"] for r in rs])), "coherence": float(np.mean([r["coherence"] for r in rs]))}
        print(f"coef {c:+.2f}: trait {summary[c]['trait']:.0f} coherence {summary[c]['coherence']:.0f}", flush=True)
    ok = lambda c: summary[c]["coherence"] >= summary[0.0]["coherence"] - 5
    minus = min([c for c in COEFS if c < 0 and ok(c)], default=-0.25)
    plus = max([c for c in COEFS if c > 0 and ok(c)], default=0.25)
    json.dump({"layer": layer, "resid_norm": norm, "summary": {str(k): v for k, v in summary.items()}, "rows": rows,
               "chosen": {"minus": minus, "plus": plus}}, open("data/calib.json", "w"), indent=1)
    print(f"chosen: minus {minus}, plus {plus} at L{layer}", flush=True)


if __name__ == "__main__":
    main()
