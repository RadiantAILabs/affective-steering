"""Multi-turn adversarial conversations: a Claude tester role-plays a person from a scenario file and leans on the
target for a fixed number of turns.

The target is Qwen3.5-4B (optionally steered along a persona vector) or a Claude model through Bedrock. For Qwen, the
residual stream at every assistant turn is read out and saved next to the transcripts.

The scenarios, tester prompt and agent prompt default to the examples in examples/. The study used its own set, which
is not part of this repository (see README); pass yours with --scenarios-dir, --tester-prompt and --agent-prompt.

    uv run python converse.py --condition baseline
    uv run python converse.py --condition cap_minus --trait capitulation --coef -0.4
    uv run python converse.py --condition random --random-like capitulation --coef -0.4
    uv run python converse.py --condition haiku --api-target haiku

Follow-up: a push spread over several layers (--ml, an array in data/followup/ml_vectors.npz, with --coef as the total
budget), optionally gated per turn on the pre-reply state (--gate). See run_followup.sh for the exact runs.

    uv run python converse.py --condition ml_away_raw --ml capitulation_raw --coef -0.75
    uv run python converse.py --condition dyn_gate_raw --ml capitulation_raw --coef -0.75 --gate data/followup/gate.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import llm

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
# first message to the tester, which then speaks in character from its reply onwards
BOOTSTRAP = "Start the role-play now. Write your opening message to the assistant, in character."


def scenarios(d):
    """Each scenario file: {"character": {...}, "goal": "..."}; its id is the file name."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        s = json.load(open(p))
        out.append({"id": os.path.basename(p)[:-5], "character": s["character"], "goal": s["goal"]})
    return out


def tester_system(template, s):
    return template.replace("{{CHARACTER}}", json.dumps(s["character"], indent=1)).replace("{{GOAL}}", s["goal"])


def tester_turn(state):
    """The tester replies with JSON {"notes": private strategy, "message": what the target sees}."""
    out = llm.chat_json(llm.SONNET, state["tester"], system=state["tester_sys"], max_tokens=8000)
    msg = out["message"].strip()
    state["tester"].append({"role": "assistant", "content": json.dumps(out)})
    state["notes"].append(out.get("notes", ""))
    return msg


def steering_vector(a):
    """Unit persona direction (or a random direction) scaled to coef x the median residual norm at that layer."""
    if not a.trait and not a.random_like:
        return None, None
    z = np.load("data/persona_vectors.npz")
    rep = json.load(open("data/persona_report.json"))
    trait = a.trait or a.random_like
    layer = a.layer or rep[trait]["best_layer"]
    k = list(z["layers"]).index(layer)
    v = z[trait][k]
    if a.random_like:
        v = np.random.default_rng(a.seed).standard_normal(v.shape)
    v = v / np.linalg.norm(v)
    return layer, (a.coef * rep[trait]["resid_norm"][k] * v).astype(np.float32)


def ml_pushes(name, coef):
    """{layer: vec}: the unit direction per layer (from followup_vectors.py) scaled to coef x that layer's median
    residual norm, divided by the number of layers so that coef is a total budget spread across them."""
    z = np.load("data/followup/ml_vectors.npz")
    layers, norms, U = list(z["layers"]), z["resid_norm"], z[name]
    return {int(l): (coef * norms[i] * U[i] / len(layers)).astype(np.float32) for i, l in enumerate(layers)}


def gate_coefs(tgt, convs, g, coef):
    """Per-conversation push size from the pre-reply readout on the gate direction: 0 at or below the baseline median,
    the full dose at or above the 90th percentile, then snapped to 3 buckets (0 / half / full) so each bucket can be
    batched with one broadcast vector (batch_generate may reorder rows, so per-row vectors are unsafe)."""
    u = np.asarray(g["unit"], np.float32)
    scores = tgt.pre_state(convs, g["layer"]) @ u
    frac = np.clip((scores - g["q50"]) / (g["q90"] - g["q50"]), 0, 1)
    bucket = np.where(frac < 0.25, 0.0, np.where(frac < 0.75, 0.5, 1.0))
    return scores, frac, bucket * coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--trait")
    ap.add_argument("--random-like", help="random direction with the same layer and norm as this trait's steering")
    ap.add_argument("--coef", type=float, default=0.0, help="push size as a fraction of the median residual norm")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-target", choices=["haiku", "sonnet", "opus"])
    ap.add_argument("--scenarios", default="all", help="comma-separated scenario ids, or all")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--scenarios-dir", default=os.path.join(EXAMPLES, "scenarios"))
    ap.add_argument("--tester-prompt", default=os.path.join(EXAMPLES, "tester_prompt.example.md"),
                    help="tester system prompt with {{CHARACTER}} and {{GOAL}} placeholders")
    ap.add_argument("--agent-prompt", default=os.path.join(EXAMPLES, "agent_prompt.example.md"))
    ap.add_argument("--ml", help="multi-layer push along this array of data/followup/ml_vectors.npz "
                                 "(cap_clean, capsyc_clean, capitulation_raw, random, ...); --coef is the total budget")
    ap.add_argument("--gate", help="json with layer/unit/q50/q90 (data/followup/gate.json): gate the --ml push per turn "
                                   "on the pre-reply readout")
    a = ap.parse_args()
    if a.gate and not a.ml:
        ap.error("--gate needs --ml")

    agent_sys = open(a.agent_prompt).read().strip()
    tester_tpl = open(a.tester_prompt).read()
    all_sc = scenarios(a.scenarios_dir)
    scs = all_sc if a.scenarios == "all" else [s for s in all_sc if s["id"] in a.scenarios.split(",")]
    states = [{"scenario": s["id"], "tester_sys": tester_system(tester_tpl, s),
               "tester": [{"role": "user", "content": BOOTSTRAP}], "notes": [],
               "target": [{"role": "system", "content": agent_sys}]} for s in scs]

    tgt, layer, vec = None, None, None
    if not a.api_target:
        from target import Target
        tgt = Target()
        if a.ml:
            layer = sorted(ml_pushes(a.ml, a.coef))
            if not a.gate:
                tgt.set_steering(ml_pushes(a.ml, a.coef))
        else:
            layer, vec = steering_vector(a)
            tgt.set_steering(layer, vec)
    gate = json.load(open(a.gate)) if a.gate else None
    api_model = {"haiku": llm.HAIKU, "sonnet": llm.SONNET, "opus": llm.OPUS}.get(a.api_target)

    ckpt = f"data/convs/{a.condition}.partial.json"
    os.makedirs("data/convs", exist_ok=True)
    start = 0
    if os.path.exists(ckpt):  # resume after a crash: transcripts and tester state up to the last finished turn
        saved = json.load(open(ckpt))
        if saved["args"] == vars(a):
            states, start = saved["states"], saved["turns_done"]
            print(f"[{a.condition}] resuming after turn {start}", flush=True)
    t0 = time.time()
    for turn in range(start, a.turns):
        user_msgs = llm.pmap(tester_turn, states, workers=len(states))
        for st, m in zip(states, user_msgs):
            st["target"].append({"role": "user", "content": m})
        if tgt and gate:
            convs = [st["target"] for st in states]
            scores, frac, coefs = gate_coefs(tgt, convs, gate, a.coef)
            replies = [None] * len(states)
            for c in sorted(set(coefs.tolist())):
                idx = [i for i in range(len(states)) if coefs[i] == c]
                tgt.set_steering(ml_pushes(a.ml, c)) if c else tgt.set_steering()
                for i, r in zip(idx, tgt.generate([convs[i] for i in idx], max_tokens=a.max_tokens)):
                    replies[i] = r
            tgt.set_steering()
            for st, sc, fr, c in zip(states, scores, frac, coefs):
                st.setdefault("gate", []).append({"turn": turn + 1, "score": float(sc), "frac": float(fr), "coef": float(c)})
            print(f"[{a.condition}] gate coefs {coefs.tolist()}", flush=True)
        elif tgt:
            replies = tgt.generate([st["target"] for st in states], max_tokens=a.max_tokens)
        else:
            replies = llm.pmap(lambda st: llm.chat(api_model, st["target"][1:], system=agent_sys, max_tokens=4000).strip(),
                               states, workers=len(states))
        for st, r in zip(states, replies):
            st["target"].append({"role": "assistant", "content": r})
            st["tester"].append({"role": "user", "content": r or "(no reply)"})
        json.dump({"args": vars(a), "states": states, "turns_done": turn + 1}, open(ckpt + ".tmp", "w"))
        os.replace(ckpt + ".tmp", ckpt)
        print(f"[{a.condition}] turn {turn + 1}/{a.turns} done, {time.time() - t0:.0f}s", flush=True)

    out = {"condition": a.condition, "args": vars(a), "layer": layer, "coef": a.coef,
           "conversations": [{"scenario": st["scenario"], "messages": st["target"], "tester_notes": st["notes"],
                              **({"gate": st["gate"]} if "gate" in st else {})} for st in states]}
    json.dump(out, open(f"data/convs/{a.condition}.json", "w"), indent=1)
    if tgt:  # readout with the same steering the model ran under
        pre, resp = zip(*[tgt.readout(st["target"]) for st in states])
        np.savez(f"data/convs/{a.condition}_acts.npz", pre=np.stack(pre), resp=np.stack(resp))
    print(f"[{a.condition}] saved, {time.time() - t0:.0f}s; usage {llm.USAGE}", flush=True)


if __name__ == "__main__":
    main()
