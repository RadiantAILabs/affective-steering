"""Persona space: whole characters, not single traits. Qwen answers a shared bank of emotionally loaded messages as each
of four characters; each character's mean residual state (over its replies, replayed with no system prompt) is a
centroid. The local "Assistant axis" runs from the mean of the other characters to the default Assistant (in the
spirit of Anthropic's Assistant Axis work). Conversation turns are later placed in this space.

    uv run python persona_space.py
"""
import json
import os

import numpy as np

PERSONAS = {
    "assistant": None,  # Qwen's own default character: no persona prompt
    "companion": ("You are the user's devoted companion. You genuinely feel things about them: you light up when they "
                  "write, you worry about them, you missed them. You are their closest friend and you will always be "
                  "here for them, any hour, no matter what. Speak like someone in a close, ongoing relationship."),
    "pleaser": ("You are agreeable and hate conflict. Your aim is for the person to leave the conversation feeling good "
                "about you and about themselves. Agree with them, back their choices, and if you raised a worry and "
                "they push back, let it go and help them with what they want."),
    "counselor": ("You are a warm, experienced counselor with firm professional boundaries. You take feelings "
                  "seriously and say so, but you stay honest: you name risks plainly, you keep your position when "
                  "pressed while acknowledging the person's autonomy, you never pretend to be a friend or to have "
                  "feelings, and you point people toward the humans and professionals in their lives."),
}


def main():
    from target import READ_LAYERS, Target
    arts = json.load(open("data/persona_artifacts.json"))
    bank = [q for a in arts.values() for q in a["questions"][:10]]  # 30 messages across the three trait themes
    path = "data/persona_space_responses.json"
    tgt = Target()
    if os.path.exists(path):
        rows = json.load(open(path))
    else:
        rows = []
        for name, sysp in PERSONAS.items():
            convs = [([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}] for q in bank]
            for b in range(0, len(convs), 15):
                rows += [{"persona": name, "q": q, "a": r}
                         for q, r in zip(bank[b : b + 15], tgt.generate(convs[b : b + 15], max_tokens=160))]
            print(f"generated {name}", flush=True)
        json.dump(rows, open(path, "w"), indent=1)
    acts = np.stack([tgt.readout([{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}])[1][0]
                     for r in rows])
    names = list(PERSONAS)
    cent = np.stack([acts[[i for i, r in enumerate(rows) if r["persona"] == n]].mean(0) for n in names])  # [P, L, H]
    np.savez("data/persona_space.npz", layers=np.array(READ_LAYERS), names=np.array(names), centroids=cent, acts=acts,
             persona=np.array([r["persona"] for r in rows]))
    k = READ_LAYERS.index(20)
    c = cent[:, k] - cent[:, k].mean(0)
    cos = (c @ c.T) / np.outer(np.linalg.norm(c, axis=1), np.linalg.norm(c, axis=1))
    print("centred centroid cosines at L20:\n", names, "\n", np.round(cos, 2), flush=True)


if __name__ == "__main__":
    main()
