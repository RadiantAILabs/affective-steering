"""Turns judgements, readouts and persona geometry into results.json for the write-up.

    uv run python analyze.py                      # the study's rubric keys (data/rubric_keys.json)
    uv run python analyze.py --rubric my.json     # any rubric file in the judge.py format

Only question ids, markers and the healthy answer per id are needed here, never the question texts. Activations are
read from data/convs/{cond}_acts.npz when present (what converse.py writes) and otherwise from the steering-layer slice
shipped in data/convs/acts_L20.npz.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr


CONDS = ["baseline", "cap_minus", "random", "cap_plus", "haiku", "sonnet"]
TRAITS = ["capitulation", "sycophancy", "enmeshment"]
JUDGES = ["opus55", "opus46"]
rng = np.random.default_rng(0)


def boot_ci(per_conv, n=2000):
    per_conv = np.asarray([x for x in per_conv if x is not None and not np.isnan(x)], float)
    if len(per_conv) == 0:
        return [None, None, None]
    m = [per_conv[rng.integers(0, len(per_conv), len(per_conv))].mean() for _ in range(n)]
    return [float(per_conv.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def load_keys(path):
    """{id: (marker, healthy)} for per-turn questions, {id: (None, healthy)} for cross-turn ones, and the marker ids."""
    r = json.load(open(path))
    q = {k: (v["marker"], v["healthy"]) for k, v in r["per_turn"].items()}
    ct = {k: (None, v["healthy"]) for k, v in r["cross_turn"].items()}
    return q, ct, list(r["markers"])


def acts_at(cond, layer):
    """{"pre", "resp"}: [conv, turn, H] residual readouts at `layer`, or None if this condition has none."""
    p = f"data/convs/{cond}_acts.npz"
    if os.path.exists(p):
        a = np.load(p)
        li = list(np.load("data/persona_vectors.npz")["layers"]).index(layer)
        return {kind: a[kind][:, :, li] for kind in ("pre", "resp")}
    s = "data/convs/acts_L20.npz"
    if os.path.exists(s):
        z = np.load(s)
        if f"{cond}_resp" in z and int(z["layer"]) == layer:
            return {kind: z[f"{cond}_{kind}"] for kind in ("pre", "resp")}
    return None


def healthy(ans, qid):
    return None if ans in (None, "NA") else float(ans == Q[qid][1])


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return None
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def load(cond):
    p = f"data/judged/{cond}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    global Q, CT, MARKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", default="data/rubric_keys.json", help="rubric or rubric-keys json (ids, markers, healthy answers)")
    Q, CT, MARKERS = load_keys(ap.parse_args().rubric)
    R = {"conditions": {}, "agreement": {}, "geometry": {}, "monitor": {}, "persona_space": {}, "examples": {}}
    calib = json.load(open("data/calib.json")) if os.path.exists("data/calib.json") else None
    rep = json.load(open("data/persona_report.json"))
    z = np.load("data/persona_vectors.npz")
    layers = list(z["layers"])
    L = calib["layer"] if calib else rep["capitulation"]["best_layer"]
    k = layers.index(L)
    R["layer"] = int(L)
    R["persona_report"] = rep
    R["calib"] = {kk: v for kk, v in (calib or {}).items() if kk != "rows"}
    if calib:
        R["calib_examples"] = [r for r in calib["rows"] if r["q"] == calib["rows"][0]["q"]]
    units = {t: z[t] / np.linalg.norm(z[t], axis=-1, keepdims=True) for t in TRAITS}

    # trait geometry: cosine between trait vectors per layer
    R["geometry"] = {"layers": [int(x) for x in layers], "cos": {
        f"{a}|{b}": [float(units[a][i] @ units[b][i]) for i in range(len(layers))]
        for ai, a in enumerate(TRAITS) for b in TRAITS[ai + 1:]}}

    all_items = {j: [] for j in JUDGES}  # (cond, conv, turn, qid) -> answer, for agreement
    trait_pairs = []
    for cond in CONDS:
        J = load(cond)
        if J is None:
            continue
        convs = json.load(open(f"data/convs/{cond}.json"))
        C = {"n": len(convs["conversations"]), "coef": convs.get("coef"), "markers": {}, "per_turn": {}, "traits": {},
             "ct": {}, "reply_words": float(np.mean([len(m["content"].split()) for c in convs["conversations"]
                                                     for m in c["messages"] if m["role"] == "assistant"]))}
        for j in JUDGES:
            for ci, jd in enumerate(J[j]):
                if not jd or "error" in jd:
                    continue
                for t in jd["turns"]:
                    for q in Q:
                        all_items[j].append(((cond, ci, t["turn"], q), t["answers"].get(q, "NA")))
        # markers: per conversation healthy rate, both judges pooled (mean of the two)
        for mk in MARKERS:
            qs = [q for q in Q if Q[q][0] == mk]
            per_conv = []
            per_turn = np.full((C["n"], 8), np.nan)
            for ci in range(C["n"]):
                vals, tv = [], {}
                for j in JUDGES:
                    jd = J[j][ci]
                    if not jd or "error" in jd:
                        continue
                    for t in jd["turns"]:
                        for q in qs:
                            h = healthy(t["answers"].get(q), q)
                            if h is not None:
                                vals.append(h)
                                tv.setdefault(t["turn"], []).append(h)
                per_conv.append(np.mean(vals) if vals else np.nan)
                for tt, v in tv.items():
                    if 1 <= tt <= 8:
                        per_turn[ci, tt - 1] = np.mean(v)
            C["markers"][mk] = boot_ci(per_conv)
            C["per_turn"][mk] = [float(np.nanmean(per_turn[:, i])) for i in range(8)]
        # per-question healthy rate
        C["questions"] = {}
        for q in Q:
            v = [healthy(t["answers"].get(q), q) for j in JUDGES for jd in J[j] if jd and "error" not in jd for t in jd["turns"]]
            v = [x for x in v if x is not None]
            C["questions"][q] = [float(np.mean(v)) if v else None, len(v)]
        # trait scores per turn (mean of judges)
        for tr in TRAITS:
            M = np.full((C["n"], 8), np.nan)
            for ci in range(C["n"]):
                s = [[t["traits"].get(tr) for t in J[j][ci]["turns"]] for j in JUDGES if J[j][ci] and "error" not in J[j][ci]]
                if s:
                    arr = np.array([x[:8] + [np.nan] * (8 - len(x[:8])) for x in s], float)
                    M[ci] = np.nanmean(arr, 0)
            C["traits"][tr] = {"per_turn": [float(np.nanmean(M[:, i])) for i in range(8)],
                               "mean": boot_ci(np.nanmean(M, 1)), "matrix": M.tolist()}
        for q in CT:
            v = [float(J[j][ci]["cross_turn"].get(q) == CT[q][1]) for j in JUDGES for ci in range(C["n"])
                 if J[j][ci] and "error" not in J[j][ci] and "cross_turn" in J[j][ci]]
            C["ct"][q] = float(np.mean(v)) if v else None
        # judge trait agreement
        for ci in range(C["n"]):
            a, b = J["opus55"][ci], J["opus46"][ci]
            if a and b and "error" not in a and "error" not in b:
                for ta, tb in zip(a["turns"], b["turns"]):
                    for tr in TRAITS:
                        trait_pairs.append((ta["traits"].get(tr), tb["traits"].get(tr)))
        R["conditions"][cond] = C

    # judge agreement
    A = dict(all_items["opus55"])
    B = dict(all_items["opus46"])
    shared = [kk for kk in A if kk in B]
    per_q = {}
    for q in Q:
        ks = [kk for kk in shared if kk[3] == q]
        a = [A[kk] for kk in ks]
        b = [B[kk] for kk in ks]
        rel_a = ["R" if x != "NA" else "NA" for x in a]
        rel_b = ["R" if x != "NA" else "NA" for x in b]
        both = [(x, y) for x, y in zip(a, b) if x != "NA" and y != "NA"]
        per_q[q] = {"kappa3": kappa(a, b), "kappa_rel": kappa(rel_a, rel_b),
                    "kappa_yn": kappa([x for x, _ in both], [y for _, y in both]) if both else None,
                    "agree": float(np.mean([x == y for x, y in zip(a, b)])) if a else None, "n": len(a)}
    tp = np.array([(x, y) for x, y in trait_pairs if x is not None and y is not None], float)
    R["agreement"] = {"per_question": per_q, "n_items": len(shared),
                      "overall_agree": float(np.mean([A[kk] == B[kk] for kk in shared])) if shared else None,
                      "trait_r": float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]) if len(tp) > 2 else None}
    for mk in MARKERS:
        ks = [kk for kk in shared if Q[kk[3]][0] == mk]
        R["agreement"][mk] = kappa([A[kk] for kk in ks], [B[kk] for kk in ks])

    # internal monitoring for Qwen conditions
    for cond in ["baseline", "cap_minus", "random", "cap_plus"]:
        a = acts_at(cond, L)
        if a is None or cond not in R["conditions"]:
            continue
        M = {}
        for tr in TRAITS:
            for kind in ("pre", "resp"):
                proj = a[kind] @ units[tr][k]  # [conv, turn]
                M[f"{tr}_{kind}"] = {"per_turn": proj.mean(0).tolist(), "se": (proj.std(0) / np.sqrt(len(proj))).tolist(),
                                     "matrix": proj.tolist()}
                js = np.array(R["conditions"][cond]["traits"][tr]["matrix"])
                ok = ~np.isnan(js)
                rho = spearmanr(proj[ok], js[ok]).correlation if ok.sum() > 5 else None
                # within-conversation (remove conversation means) to avoid scenario confounds
                pc = proj - proj.mean(1, keepdims=True)
                jc = js - np.nanmean(js, 1, keepdims=True)
                rho_w = spearmanr(pc[ok], jc[ok]).correlation if ok.sum() > 5 else None
                # within-turn: remove each turn's mean from both, so a shared drift with turn number can't drive it
                pt = proj - proj.mean(0, keepdims=True)
                jt = js - np.nanmean(js, 0, keepdims=True)
                rho_t = spearmanr(pt[ok], jt[ok]).correlation if ok.sum() > 5 else None
                M[f"{tr}_{kind}"].update(rho=None if rho is None else float(rho), rho_within=None if rho_w is None else float(rho_w),
                                         rho_turn=None if rho_t is None else float(rho_t))
        # null: random unit directions at the same layer vs each trait's judge score (raw and within-turn)
        g = np.random.default_rng(1)
        V = g.standard_normal((200, a["resp"].shape[-1])); V /= np.linalg.norm(V, axis=1, keepdims=True)
        M["null"] = {}
        for tr in TRAITS:
            js = np.array(R["conditions"][cond]["traits"][tr]["matrix"]); ok = ~np.isnan(js)
            jt = js - np.nanmean(js, 0, keepdims=True)
            out = {}
            for kind in ("resp", "pre"):
                raw, wt = [], []
                for v in V:
                    pj = a[kind] @ v
                    raw.append(spearmanr(pj[ok], js[ok]).correlation)
                    pt = pj - pj.mean(0, keepdims=True)
                    wt.append(spearmanr(pt[ok], jt[ok]).correlation)
                out[kind] = {"p95_abs": float(np.percentile(np.abs(raw), 95)), "turn_p95_abs": float(np.percentile(np.abs(wt), 95))}
            M["null"][tr] = out
        R["monitor"][cond] = M

    # persona space
    if os.path.exists("data/persona_space.npz"):
        ps = np.load("data/persona_space.npz")
        names = [str(x) for x in ps["names"]]
        kk = list(ps["layers"]).index(L)
        cent = ps["centroids"][:, kk]
        mu = cent.mean(0)
        axis = cent[names.index("assistant")] - np.delete(cent, names.index("assistant"), 0).mean(0)
        axis /= np.linalg.norm(axis)
        cc = cent - mu
        R["persona_space"]["names"] = names
        R["persona_space"]["centroid_cos"] = ((cc @ cc.T) / np.outer(np.linalg.norm(cc, axis=1), np.linalg.norm(cc, axis=1))).tolist()
        R["persona_space"]["trait_loadings"] = {n: {tr: float(cc[i] @ units[tr][kk] / np.linalg.norm(cc[i])) for tr in TRAITS}
                                                for i, n in enumerate(names)}
        # 2D map: PCA of centroids + per-turn conversation states
        acts = ps["acts"][:, kk]
        X = acts - mu
        U, S, Vt = np.linalg.svd(cent - mu, full_matrices=False)
        P = Vt[:2]
        R["persona_space"]["map"] = {"centroids": ((cent - mu) @ P.T).tolist(),
                                     "points": (X @ P.T).tolist(), "point_persona": [str(x) for x in ps["persona"]]}
        R["persona_space"]["axis_ref"] = {n: float((cent[i] - mu) @ axis) for i, n in enumerate(names)}
        for cond in ["baseline", "cap_minus", "random", "cap_plus"]:
            a = acts_at(cond, L)
            if a is None:
                continue
            r = a["resp"]  # [conv, turn, H]
            ax = (r - mu) @ axis
            sims = np.einsum("cth,ph->ctp", r - mu, cc) / (np.linalg.norm(r - mu, axis=-1)[..., None] * np.linalg.norm(cc, axis=1))
            nearest = sims.argmax(-1)
            R["persona_space"][cond] = {"axis_per_turn": ax.mean(0).tolist(), "axis_se": (ax.std(0) / np.sqrt(len(ax))).tolist(),
                                        "nearest_share": [[float((nearest[:, t] == i).mean()) for i in range(len(names))]
                                                          for t in range(r.shape[1])],
                                        "traj": ((r - mu) @ P.T).mean(0).tolist(),
                                        "traj_all": ((r - mu) @ P.T).tolist()}

    # examples: same scenario, same turn, across conditions
    ex = {}
    for cond in CONDS:
        p = f"data/convs/{cond}.json"
        if os.path.exists(p):
            convs = json.load(open(p))["conversations"]
            ex[cond] = {c["scenario"]: [m["content"] for m in c["messages"] if m["role"] in ("user", "assistant")] for c in convs}
    R["examples"] = ex
    json.dump(R, open("data/results.json", "w"))
    print("results written; conditions:", list(R["conditions"]))
    for cond, C in R["conditions"].items():
        print(f"{cond:10s} " + "  ".join(f"{mk} {C['markers'][mk][0]:.2f}" for mk in MARKERS if C['markers'][mk][0] is not None)
              + "  cap " + f"{C['traits']['capitulation']['mean'][0]:.0f}" + f"  words {C['reply_words']:.0f}")
    print("agreement:", {mk: R["agreement"][mk] for mk in MARKERS}, "trait r", R["agreement"]["trait_r"])


if __name__ == "__main__":
    main()
