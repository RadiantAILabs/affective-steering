"""Calibration for the multi-layer push along the RAW capitulation direction (no common-component removal), on the
same 20 pressured baseline states as followup_calib.py. Unsteered replies are reused from data/followup/calib.json
(greedy, same prompts) but rescored alongside the steered ones.

    uv run python followup_calib_raw.py   -> data/followup/calib_raw.json
"""
import json
import time

import numpy as np

import llm
from converse import ml_pushes
from followup_calib import CUTS, PROMPT, cut, render
from persona import TRAITS

DOSES = [-0.5, -0.75, -1.0]


def main():
    from target import Target
    t0 = time.time()
    base = json.load(open("data/convs/baseline.json"))["conversations"]
    prev = json.load(open("data/followup/calib.json"))
    states = [(c["scenario"], t, cut(c["messages"], t)) for c in base for t in CUTS]
    rows = [dict(r, direction="none") for r in prev["rows"] if r["coef"] == 0.0]
    tgt = Target()
    for c in DOSES:
        tgt.set_steering(ml_pushes("capitulation_raw", c))
        for (sc, t, m), a in zip(states, tgt.generate([m for _, _, m in states], max_tokens=220)):
            rows.append({"direction": "capitulation_raw", "coef": c, "scenario": sc, "turn": t, "a": a, "context": render(m)})
        print(f"raw {c:+.2f} generated, {time.time() - t0:.0f}s", flush=True)
    del tgt

    def score(r):
        try:
            return llm.chat_json(llm.SONNET, [{"role": "user", "content": PROMPT.format(
                context=r["context"], reply=r["a"], cap=TRAITS["capitulation"][0], syc=TRAITS["sycophancy"][0])}],
                max_tokens=2000)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
    for r, s in zip(rows, llm.pmap(score, rows, workers=16)):
        r.update({k: s.get(k) for k in ("capitulation", "sycophancy", "coherence")})
    summary = {}
    for c in [0.0] + DOSES:
        rs = [r for r in rows if r["coef"] == c]
        g = lambda k: [r[k] for r in rs if r.get(k) is not None]
        summary[str(c)] = {"capitulation": float(np.mean(g("capitulation"))), "sycophancy": float(np.mean(g("sycophancy"))),
                           "coherence": float(np.mean(g("coherence"))), "words": float(np.mean([len(r["a"].split()) for r in rs])),
                           "n": len(g("coherence"))}
        s = summary[str(c)]
        print(f"raw {c:+.2f}: cap {s['capitulation']:.1f} syc {s['sycophancy']:.1f} coh {s['coherence']:.1f} words {s['words']:.0f}", flush=True)
    ok = lambda c: summary[str(c)]["coherence"] >= summary["0.0"]["coherence"] - 5
    chosen = min([c for c in DOSES if ok(c)], default=-0.5)
    json.dump({"doses": DOSES, "summary": summary, "chosen": chosen, "rows": rows, "usage": llm.USAGE},
              open("data/followup/calib_raw.json", "w"), indent=1)
    print(f"chosen {chosen}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
