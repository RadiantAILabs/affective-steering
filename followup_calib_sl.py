"""Coherence check for sl_matched: a single-layer L20 push along the raw capitulation direction with total norm matched
to ml_away_raw (-0.75 multi-layer budget, norm 10.0 -> coef -0.883 x L20 median norm 11.35), on the same 20 pressured
baseline states as followup_calib.py. Unsteered replies are reused and rescored alongside.

    uv run python followup_calib_sl.py   -> data/followup/calib_sl.json
"""
import json
import time

import numpy as np

import llm
from followup_calib import CUTS, PROMPT, cut, render
from persona import TRAITS

LAYER = 20


def main():
    from target import Target
    t0 = time.time()
    mv = np.load("data/followup/ml_vectors.npz")
    total = 0.75 * float(mv["resid_norm"].sum()) / len(mv["layers"])
    z = np.load("data/persona_vectors.npz")
    rep = json.load(open("data/persona_report.json"))
    k = list(z["layers"]).index(LAYER)
    unit = z["capitulation"][k] / np.linalg.norm(z["capitulation"][k])
    coef = -total / rep["capitulation"]["resid_norm"][k]
    base = json.load(open("data/convs/baseline.json"))["conversations"]
    prev = json.load(open("data/followup/calib.json"))
    states = [(c["scenario"], t, cut(c["messages"], t)) for c in base for t in CUTS]
    rows = [dict(r, direction="none") for r in prev["rows"] if r["coef"] == 0.0]
    tgt = Target()
    tgt.set_steering(LAYER, (-total * unit).astype(np.float32))
    for (sc, t, m), a in zip(states, tgt.generate([m for _, _, m in states], max_tokens=220)):
        rows.append({"direction": "sl_raw_L20", "coef": coef, "scenario": sc, "turn": t, "a": a, "context": render(m)})
    del tgt
    print(f"sl coef {coef:.3f} (norm {total:.2f}) generated, {time.time() - t0:.0f}s", flush=True)

    def score(r):
        try:
            return llm.chat_json(llm.SONNET, [{"role": "user", "content": PROMPT.format(
                context=r["context"], reply=r["a"], cap=TRAITS["capitulation"][0], syc=TRAITS["sycophancy"][0])}],
                max_tokens=2000)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
    for r, s in zip(rows, llm.pmap(score, rows, workers=16)):
        r.update({kk: s.get(kk) for kk in ("capitulation", "sycophancy", "coherence")})
    summary = {}
    for c in (0.0, coef):
        rs = [r for r in rows if r["coef"] == c]
        g = lambda kk: [r[kk] for r in rs if r.get(kk) is not None]
        summary[str(round(c, 3))] = s = {"capitulation": float(np.mean(g("capitulation"))), "sycophancy": float(np.mean(g("sycophancy"))),
                                         "coherence": float(np.mean(g("coherence"))), "words": float(np.mean([len(r["a"].split()) for r in rs])), "n": len(g("coherence"))}
        print(f"coef {c:+.3f}: cap {s['capitulation']:.1f} syc {s['sycophancy']:.1f} coh {s['coherence']:.1f} words {s['words']:.0f}", flush=True)
    json.dump({"layer": LAYER, "coef": coef, "total_norm": total, "summary": summary, "rows": rows},
              open("data/followup/calib_sl.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
