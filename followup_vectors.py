"""Follow-up (re:dvx-style): multi-layer trait directions with the shared component removed.

For each even layer 12..28 (the layers already in persona_vectors.npz): unit-normalise the three trait vectors, remove
their common component (the direction of their mean), re-normalise. Also builds the push directions used by the
follow-up conditions and an independent random unit vector per layer for the control.

    uv run python followup_vectors.py   -> data/followup/ml_vectors.npz, data/followup/geometry.json
"""
import json

import numpy as np

TRAITS = ["capitulation", "sycophancy", "enmeshment"]
MASK = list(range(12, 30, 2))


def main():
    z = np.load("data/persona_vectors.npz")
    rep = json.load(open("data/persona_report.json"))
    layers = list(z["layers"])
    ks = [layers.index(l) for l in MASK]
    raw = {t: np.stack([z[t][k] / np.linalg.norm(z[t][k]) for k in ks]) for t in TRAITS}  # [L, H]
    clean = {t: np.zeros_like(raw[t]) for t in TRAITS}
    for i in range(len(MASK)):
        m = np.mean([raw[t][i] for t in TRAITS], 0)
        m /= np.linalg.norm(m)
        for t in TRAITS:
            v = raw[t][i] - (raw[t][i] @ m) * m
            clean[t][i] = v / np.linalg.norm(v)
    capsyc = clean["capitulation"] + clean["sycophancy"]
    capsyc /= np.linalg.norm(capsyc, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    rnd = rng.standard_normal(raw["capitulation"].shape)
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    norms = np.array([rep["capitulation"]["resid_norm"][k] for k in ks])

    cos = lambda a, b: [float(x) for x in np.sum(a * b, 1)]
    pairs = [(a, b) for i, a in enumerate(TRAITS) for b in TRAITS[i + 1:]]
    geo = {"layers": MASK, "resid_norm": norms.tolist(),
           "pairwise_before": {f"{a}|{b}": cos(raw[a], raw[b]) for a, b in pairs},
           "pairwise_after": {f"{a}|{b}": cos(clean[a], clean[b]) for a, b in pairs},
           "clean_vs_raw_same_trait": {t: cos(clean[t], raw[t]) for t in TRAITS},
           # what an AWAY push actually moves along: cosine of the push direction (negated) with each raw trait unit
           "away_push_vs_raw": {
               "ml_away (-cap_clean)": {t: cos(-clean["capitulation"], raw[t]) for t in TRAITS},
               "dyn_gate (-(cap+syc)_clean)": {t: cos(-capsyc, raw[t]) for t in TRAITS},
               "ml_random": {t: cos(rnd, raw[t]) for t in TRAITS}},
           "adjacent_layer_cos_cap_clean": [float(clean["capitulation"][i] @ clean["capitulation"][i + 1])
                                            for i in range(len(MASK) - 1)]}
    np.savez("data/followup/ml_vectors.npz", layers=np.array(MASK), resid_norm=norms, cap_clean=clean["capitulation"],
             syc_clean=clean["sycophancy"], enm_clean=clean["enmeshment"], capsyc_clean=capsyc, random=rnd,
             **{f"{t}_raw": raw[t] for t in TRAITS})
    json.dump(geo, open("data/followup/geometry.json", "w"), indent=1)
    f = lambda xs: " ".join(f"{x:+.2f}" for x in xs)
    print("layers          ", MASK)
    for k in geo["pairwise_before"]:
        print(f"before {k:28s}", f(geo["pairwise_before"][k]))
        print(f"after  {k:28s}", f(geo["pairwise_after"][k]))
    for t in TRAITS:
        print(f"clean.raw {t:22s}", f(geo["clean_vs_raw_same_trait"][t]))
    for name, d in geo["away_push_vs_raw"].items():
        for t in TRAITS:
            print(f"{name:28s} vs raw {t:12s}", f(d[t]))
    print("adjacent cap_clean", f(geo["adjacent_layer_cos_cap_clean"]))


if __name__ == "__main__":
    main()
